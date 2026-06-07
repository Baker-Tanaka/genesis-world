"""Unit tests for the PX4 SITL bridge (gs.px4).

These run without a GPU and without a real PX4: geometry/atmosphere conversions are pure
numpy, and the end-to-end lockstep exchange is exercised against an in-process fake PX4 TCP
server. Tests needing MAVLink are skipped when ``pymavlink`` is not installed.
"""

import socket
import threading
import time

import numpy as np
import pytest
import torch

import genesis as gs
from genesis.ext.px4 import geo
from genesis.ext.px4.options import PX4Options


def assert_allclose(actual, desired, *, tol=1e-6):
    np.testing.assert_allclose(np.asarray(actual, dtype=np.float64), desired, atol=tol, rtol=0.0)


# ==========================================================================================
# geo.py  -- pure numpy, no MAVLink / GPU
# ==========================================================================================


@pytest.mark.required
def test_geo_frame_conversions():
    # ENU -> NED: N=y, E=x, D=-z
    v = np.array([[1.0, 2.0, 3.0]])
    assert_allclose(geo.enu_to_ned(v), [[2.0, 1.0, -3.0]], tol=1e-12)
    # involution
    assert_allclose(geo.ned_to_enu(geo.enu_to_ned(v)), v, tol=1e-12)

    # FLU -> FRD: x stays, y/z negate
    assert_allclose(geo.flu_to_frd(v), [[1.0, -2.0, -3.0]], tol=1e-12)
    assert_allclose(geo.frd_to_flu(geo.flu_to_frd(v)), v, tol=1e-12)


@pytest.mark.required
def test_geo_attitude_identity():
    # Body aligned with world (identity ENU->FLU) must map to identity NED->FRD.
    q = geo.enu_flu_quat_to_ned_frd(np.array([1.0, 0.0, 0.0, 0.0]))
    assert abs(abs(q[0]) - 1.0) < 1e-9
    assert_allclose(q[1:], [0.0, 0.0, 0.0], tol=1e-9)


@pytest.mark.required
def test_geo_local_ned_to_latlon():
    home = (47.0, 8.5, 500.0)
    # Zero offset returns home.
    lat, lon, alt = geo.local_ned_to_latlon(home, 0.0, 0.0, 0.0)
    assert_allclose([float(lat), float(lon), float(alt)], list(home), tol=1e-9)

    # 111 m north ~ 0.001 deg latitude; down lowers AMSL altitude.
    lat, lon, alt = geo.local_ned_to_latlon(home, 111.0, 0.0, 10.0)
    assert lat > home[0]
    assert_allclose(float(lat) - home[0], 111.0 / geo.EARTH_RADIUS * 180.0 / np.pi, tol=1e-6)
    assert_allclose(float(alt), home[2] - 10.0, tol=1e-9)


@pytest.mark.required
def test_geo_atmosphere_roundtrip():
    assert_allclose(float(geo.altitude_to_pressure(0.0)), 1013.25, tol=1e-2)
    for alt in (0.0, 100.0, 1000.0, 5000.0):
        p = geo.altitude_to_pressure(alt)
        assert_allclose(float(geo.pressure_to_altitude(p)), alt, tol=1e-3)
    # Pressure decreases with altitude.
    assert geo.altitude_to_pressure(2000.0) < geo.altitude_to_pressure(0.0)


# ==========================================================================================
# mavlink_io.py  -- requires pymavlink
# ==========================================================================================


def _find_msg(msgs, msg_type):
    for m in msgs:
        if m.get_type() == msg_type:
            return m
    return None


@pytest.mark.required
def test_mavlink_sensor_roundtrip():
    pytest.importorskip("pymavlink")
    from pymavlink.dialects.v20 import common as mavlink2

    from genesis.ext.px4.mavlink_io import MavConn

    a, b = socket.socketpair()
    try:
        conn = MavConn(a)
        conn.send_hil_sensor(
            time_usec=12345,
            acc=[0.0, 0.0, 9.81],
            gyro=[0.1, 0.2, 0.3],
            mag=[0.21, 0.0, 0.43],
            abs_pressure=1013.25,
            pressure_alt=0.0,
            temperature=15.0,
            fields_updated=0x1FFF,
        )
        data = b.recv(4096)
        decoder = mavlink2.MAVLink(None)
        msg = _find_msg(decoder.parse_buffer(data) or [], "HIL_SENSOR")
        assert msg is not None
        assert msg.time_usec == 12345
        assert abs(msg.zacc - 9.81) < 1e-4
        assert abs(msg.xgyro - 0.1) < 1e-4
        assert abs(msg.abs_pressure - 1013.25) < 1e-2
    finally:
        a.close()
        b.close()


@pytest.mark.required
def test_mavlink_parse_actuator_controls():
    pytest.importorskip("pymavlink")
    from pymavlink.dialects.v20 import common as mavlink2

    from genesis.ext.px4.mavlink_io import MAV_MODE_FLAG_SAFETY_ARMED, parse_actuator_controls

    controls = [0.25] * 16
    armed_msg = mavlink2.MAVLink_hil_actuator_controls_message(
        time_usec=1, controls=controls, mode=MAV_MODE_FLAG_SAFETY_ARMED, flags=0
    )
    parsed_controls, armed = parse_actuator_controls(armed_msg)
    assert armed is True
    assert_allclose(parsed_controls[:4], [0.25, 0.25, 0.25, 0.25], tol=1e-6)

    disarmed_msg = mavlink2.MAVLink_hil_actuator_controls_message(time_usec=1, controls=controls, mode=0, flags=0)
    _, armed = parse_actuator_controls(disarmed_msg)
    assert armed is False


# ==========================================================================================
# bridge.py  -- end-to-end lockstep against a fake PX4 server
# ==========================================================================================


class _FakeScene:
    def __init__(self, n_envs, dt=0.004):
        self.n_envs = n_envs
        self.dt = dt
        self._pre_step_callbacks = []

    def register_pre_step_callback(self, cb):
        self._pre_step_callbacks.append(cb)


class _FakeDrone:
    def __init__(self, scene, n_propellers=4):
        self._scene = scene
        self.n_propellers = n_propellers
        self.last_rpm = None

    def _shape(self, last):
        if self._scene.n_envs == 0:
            return (last,)
        return (self._scene.n_envs, last)

    def get_pos(self):
        return torch.zeros(self._shape(3))

    def get_vel(self):
        return torch.zeros(self._shape(3))

    def set_propellers_rpm(self, rpm):
        self.last_rpm = rpm.detach().cpu().numpy()


class _FakeIMUReturn:
    def __init__(self, scene):
        b = max(1, scene.n_envs)
        shape = (3,) if scene.n_envs == 0 else (scene.n_envs, 3)
        self.lin_acc = torch.zeros(shape)
        self.lin_acc[..., 2] = 9.81  # at rest specific force points up
        self.ang_vel = torch.zeros(shape)
        self.mag = torch.zeros(shape)
        self.mag[..., 0] = 0.21


class _FakeIMU:
    def __init__(self, scene):
        self._scene = scene

    def read(self):
        return _FakeIMUReturn(self._scene)


class _FakePX4(threading.Thread):
    """A minimal PX4 stand-in: connect, and reply to every HIL_SENSOR with a fixed
    HIL_ACTUATOR_CONTROLS (lockstep 1:1)."""

    def __init__(self, host, port, controls, armed):
        super().__init__(daemon=True)
        from pymavlink.dialects.v20 import common as mavlink2

        self._mavlink2 = mavlink2
        self.host = host
        self.port = port
        self.controls = list(controls) + [0.0] * (16 - len(controls))
        self.armed = armed
        self.connected = threading.Event()
        self._stop = threading.Event()

    def run(self):
        sock = None
        deadline = time.time() + 10.0
        while time.time() < deadline:
            try:
                sock = socket.create_connection((self.host, self.port), timeout=1.0)
                break
            except OSError:
                time.sleep(0.05)
        assert sock is not None, "fake PX4 could not connect to the bridge"
        self.connected.set()
        sock.settimeout(0.2)
        mav = self._mavlink2.MAVLink(None)
        mode = 128 if self.armed else 0
        while not self._stop.is_set():
            try:
                data = sock.recv(4096)
            except socket.timeout:
                continue
            except OSError:
                break
            if not data:
                break
            for m in mav.parse_buffer(data) or []:
                if m.get_type() == "HIL_SENSOR":
                    reply = self._mavlink2.MAVLink_hil_actuator_controls_message(
                        time_usec=m.time_usec, controls=self.controls, mode=mode, flags=0
                    )
                    sock.sendall(reply.pack(mav))
        sock.close()

    def stop(self):
        self._stop.set()
        self.join(timeout=5.0)


def _find_free_base(count):
    """Find a base TCP port such that base..base+count-1 are all bindable."""
    for base in range(15600, 15900):
        socks = []
        ok = True
        for i in range(count):
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                s.bind(("127.0.0.1", base + i))
            except OSError:
                ok = False
                s.close()
                break
            socks.append(s)
        for s in socks:
            s.close()
        if ok:
            return base
    raise RuntimeError("no free port range found")


def _run_bridge(n_envs, controls, armed, motor_mapping=None):
    pytest.importorskip("pymavlink")
    from genesis.ext.px4.bridge import PX4Bridge

    base = _find_free_base(max(1, n_envs))
    scene = _FakeScene(n_envs)
    drone = _FakeDrone(scene)
    imu = _FakeIMU(scene)

    fakes = [_FakePX4("127.0.0.1", base + i, controls, armed) for i in range(max(1, n_envs))]
    for f in fakes:
        f.start()

    opts = PX4Options(
        auto_spawn=False,
        host="127.0.0.1",
        base_tcp_port=base,
        connect_timeout=10.0,
        max_rpm=25000.0,
        motor_mapping=motor_mapping,
    )
    bridge = PX4Bridge(scene, drone, imu, opts)
    bridge.start()
    try:
        rpm = bridge.update()
        rpm = bridge.update()  # a second exchange to confirm steady-state lockstep
    finally:
        bridge.stop()
        for f in fakes:
            f.stop()
    return rpm, drone.last_rpm


@pytest.mark.required
def test_bridge_lockstep_batched():
    rpm, last = _run_bridge(n_envs=2, controls=[0.5, 0.5, 0.5, 0.5], armed=True)
    assert rpm.shape == (2, 4)
    assert_allclose(rpm, 0.5 * 25000.0, tol=1e-3)
    assert last.shape == (2, 4)
    assert_allclose(last, 0.5 * 25000.0, tol=1e-3)


@pytest.mark.required
def test_bridge_disarmed_zero_rpm():
    rpm, _ = _run_bridge(n_envs=2, controls=[0.7, 0.7, 0.7, 0.7], armed=False)
    assert rpm.shape == (2, 4)
    assert_allclose(rpm, 0.0, tol=1e-9)


@pytest.mark.required
def test_bridge_motor_mapping_unbatched():
    # n_envs=0 exercises the unbatched set_propellers_rpm path; mapping reverses channels.
    rpm, last = _run_bridge(
        n_envs=0, controls=[0.1, 0.2, 0.3, 0.4], armed=True, motor_mapping=(3, 2, 1, 0)
    )
    assert rpm.shape == (1, 4)
    expected = np.array([0.4, 0.3, 0.2, 0.1]) * 25000.0
    assert_allclose(rpm[0], expected, tol=1e-3)
    assert last.shape == (4,)  # unbatched apply
    assert_allclose(last, expected, tol=1e-3)
