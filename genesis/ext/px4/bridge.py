"""PX4Bridge: the simulator side of PX4's lockstep MAVLink interface for Genesis.

Each parallel environment is flown by its own PX4 SITL instance. Every simulation step the
bridge reads the batched drone/IMU state once, converts it into per-instance sensor
messages, streams them to PX4, blocks until every instance returns its motor command
(lockstep), maps those commands to propeller RPMs and lets the physics advance.

Typical use::

    px4 = gs.px4.PX4Bridge(scene, drone, imu, gs.px4.PX4Options(px4_dir=..., ...))
    px4.start()
    for _ in range(N):
        scene.step()        # sensors<->actuators exchanged automatically
    px4.stop()
"""

import selectors
import socket
import time

import numpy as np
import torch

import genesis as gs

from . import geo
from . import mavlink_io as mio
from .mavlink_io import MavConn
from .options import PX4Options
from .process import PX4ProcessManager

_HIL_ACTUATOR_CONTROLS = "HIL_ACTUATOR_CONTROLS"


class PX4Bridge:
    """Connects a Genesis drone (batched over envs) to one PX4 SITL instance per env."""

    def __init__(self, scene, drone, imu, options: PX4Options | None = None):
        self._scene = scene
        self._drone = drone
        self._imu = imu
        self._options = options if options is not None else PX4Options(auto_spawn=False)

        self._B = max(1, scene.n_envs)
        self._n_prop = drone.n_propellers

        self._conns: list[MavConn] = []
        self._listeners: list[socket.socket] = []
        self._selector: selectors.BaseSelector | None = None
        self._proc_mgr = PX4ProcessManager(self._options) if self._options.auto_spawn else None

        self._registered = False
        self._started = False
        self._sim_time = 0.0  # seconds, advanced by dt each exchange
        self._step = 0

        # Decimation intervals (in steps); computed in start() once dt is known.
        self._gps_every = 1
        self._mag_every = 1
        self._baro_every = 1

        # Motor channel -> propeller index map.
        mapping = self._options.motor_mapping
        if mapping is None:
            mapping = tuple(range(self._n_prop))
        if len(mapping) < self._n_prop:
            gs.raise_exception(f"motor_mapping has {len(mapping)} entries but drone has {self._n_prop} propellers.")
        self._motor_mapping = np.asarray(mapping[: self._n_prop], dtype=np.int64)

        # Last commanded RPM (B, n_prop); applied when an instance reports no fresh command.
        self._last_rpm = np.zeros((self._B, self._n_prop), dtype=np.float64)

        # Per-instance flag: True once PX4 has delivered its first actuator command. PX4 only
        # starts emitting commands after it has booted off a continuous HIL_SENSOR stream, so
        # until then we must not block waiting for it (that would deadlock). After the first
        # command we hold strict lockstep for that instance.
        self._established = [False] * self._B

    # ====================================================================== lifecycle
    def start(self) -> "PX4Bridge":
        """Open the simulator TCP servers, (optionally) spawn PX4, connect, and hook step()."""
        if self._started:
            gs.raise_exception("PX4Bridge.start() called twice.")

        dt = self._scene.dt
        # PX4's lockstep IMU integration rejects a zero/non-increasing timestamp (vehicle_imu
        # logs a "timestamp error" and never publishes), so start one dt in rather than at 0.
        self._sim_time = dt
        self._gps_every = max(1, round(1.0 / (self._options.gps_hz * dt))) if self._options.gps_hz > 0 else 1
        self._mag_every = max(1, round(1.0 / (self._options.mag_hz * dt))) if self._options.mag_hz > 0 else 1
        self._baro_every = max(1, round(1.0 / (self._options.baro_hz * dt))) if self._options.baro_hz > 0 else 1

        # 1) Listen on one port per env.
        for i in range(self._B):
            port = self._options.base_tcp_port + i
            lsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            lsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            lsock.bind((self._options.host, port))
            lsock.listen(1)
            lsock.settimeout(self._options.connect_timeout)
            self._listeners.append(lsock)
            gs.logger.info(f"PX4Bridge listening for instance {i} on {self._options.host}:{port}")

        # 2) Spawn PX4 instances (if configured to).
        if self._proc_mgr is not None:
            for i in range(self._B):
                self._proc_mgr.spawn(i)

        # 3) Accept the connections.
        self._selector = selectors.DefaultSelector()
        for i, lsock in enumerate(self._listeners):
            try:
                csock, addr = lsock.accept()
            except socket.timeout:
                self._cleanup_sockets()
                gs.raise_exception(f"Timed out waiting for PX4 instance {i} to connect.")
            csock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            conn = MavConn(csock)
            conn.sock.setblocking(False)
            self._conns.append(conn)
            self._selector.register(conn.sock, selectors.EVENT_READ, data=i)
            gs.logger.info(f"PX4 instance {i} connected from {addr}.")

        # 4) Drive the exchange from the scene's step loop.
        self._scene.register_pre_step_callback(self._on_pre_step)
        self._registered = True
        self._started = True
        return self

    def stop(self) -> None:
        """Tear down: detach callback, close sockets, terminate PX4 processes."""
        if self._registered:
            try:
                self._scene._pre_step_callbacks.remove(self._on_pre_step)
            except (ValueError, AttributeError):
                pass
            self._registered = False
        self._cleanup_sockets()
        if self._proc_mgr is not None:
            self._proc_mgr.terminate_all()
        self._started = False

    def _cleanup_sockets(self) -> None:
        if self._selector is not None:
            self._selector.close()
            self._selector = None
        for conn in self._conns:
            conn.close()
        self._conns.clear()
        for lsock in self._listeners:
            try:
                lsock.close()
            except OSError:
                pass
        self._listeners.clear()

    def __enter__(self) -> "PX4Bridge":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    # ====================================================================== public info
    @property
    def mavlink_endpoints(self) -> list[tuple[str, int]]:
        """UDP offboard endpoints (host, port) PX4 SITL opens for GCS/MAVSDK control,
        one per env (PX4 default ``14540 + instance``). Use these to arm/takeoff/offboard."""
        return [(self._options.host, 14540 + i) for i in range(self._B)]

    # ====================================================================== step driver
    def _on_pre_step(self) -> bool:
        """Pre-step callback: run one lockstep sensor/actuator exchange. Returns False so
        the scene advances physics with the freshly received motor commands."""
        self.update()
        return False  # never veto the physics advance

    def pump(self) -> np.ndarray:
        """Run one sensor<->actuator exchange that advances PX4's clock WITHOUT stepping the
        physics or driving the motors.

        Use while PX4 is still booting/connecting: the drone holds its current (resting) pose,
        PX4 still gets a steady sensor stream and boots its EKF, but the (comparatively
        expensive) physics solve and viewer render are skipped entirely.
        """
        return self.update(apply=False)

    def update(self, apply: bool = True) -> np.ndarray:
        """Perform one sensor->PX4->actuator exchange and apply the resulting RPMs.

        Returns the applied RPM array, shape ``(B, n_prop)``. Can be called manually (with
        ``auto_spawn=False`` and no registered callback) right before ``scene.step()``. With
        ``apply=False`` the motor command is computed but not written to the drone (used by
        :meth:`pump` during boot, where physics is not advancing).
        """
        sensors = self._read_state()
        time_usec = int(round(self._sim_time * 1e6))

        send_gps = (self._step % self._gps_every) == 0
        send_mag = (self._step % self._mag_every) == 0
        send_baro = (self._step % self._baro_every) == 0

        fields = mio.FIELD_ACCEL | mio.FIELD_GYRO
        if send_mag:
            fields |= mio.FIELD_MAG
        if send_baro:
            fields |= mio.FIELD_BARO

        # ---- send sensors to every instance --------------------------------------
        for i, conn in enumerate(self._conns):
            if self._step == 0:
                conn.send_heartbeat()
            conn.send_system_time(time_usec)
            conn.send_hil_sensor(
                time_usec,
                sensors["acc"][i],
                sensors["gyro"][i],
                sensors["mag"][i],
                sensors["abs_pressure"][i],
                sensors["pressure_alt"][i],
                sensors["temperature"][i],
                fields,
            )
            if send_gps:
                conn.send_hil_gps(
                    time_usec,
                    sensors["lat"][i],
                    sensors["lon"][i],
                    sensors["alt"][i],
                    sensors["vn"][i],
                    sensors["ve"][i],
                    sensors["vd"][i],
                )

        # ---- lockstep: wait for every instance's motor command -------------------
        controls, armed = self._recv_actuators()

        # ---- map normalized motor outputs -> propeller RPM -----------------------
        rpm = self._last_rpm
        for i in range(self._B):
            if controls[i] is None:
                continue  # keep previous command for this instance
            u = np.clip(np.asarray(controls[i][: self._n_prop], dtype=np.float64), 0.0, 1.0)
            env_rpm = np.zeros(self._n_prop, dtype=np.float64)
            if not (self._options.arm_gate and not armed[i]):
                env_rpm[self._motor_mapping] = u * self._options.max_rpm
            rpm[i] = env_rpm
        self._last_rpm = rpm

        if apply:
            self._apply_rpm(rpm)

        self._sim_time += self._scene.dt
        self._step += 1
        return rpm

    # ====================================================================== internals
    def _read_state(self) -> dict:
        """Read batched drone + IMU state once and convert to per-instance PX4 sensors."""
        pos = self._to_np(self._drone.get_pos())  # (B,3) ENU world
        vel = self._to_np(self._drone.get_vel())  # (B,3) ENU world
        imu = self._imu.read()
        acc_flu = self._to_np(imu.lin_acc)  # (B,3) body FLU specific force
        gyro_flu = self._to_np(imu.ang_vel)  # (B,3) body FLU rate
        mag_flu = self._to_np(imu.mag)  # (B,3) body FLU field [gauss]

        # Body FLU -> FRD
        acc = geo.flu_to_frd(acc_flu)
        gyro = geo.flu_to_frd(gyro_flu)
        mag = geo.flu_to_frd(mag_flu)

        # World ENU -> NED for position/velocity
        pos_ned = geo.enu_to_ned(pos)
        vel_ned = geo.enu_to_ned(vel)
        north, east, down = pos_ned[:, 0], pos_ned[:, 1], pos_ned[:, 2]
        lat, lon, alt = geo.local_ned_to_latlon(self._options.home, north, east, down)

        alt_amsl = self._options.home[2] + pos[:, 2]  # ENU z is up
        abs_pressure = geo.altitude_to_pressure(alt_amsl)
        temperature = geo.temperature_at(alt_amsl)

        return {
            "acc": acc,
            "gyro": gyro,
            "mag": mag,
            "abs_pressure": abs_pressure,
            "pressure_alt": alt_amsl,
            "temperature": temperature,
            "lat": lat,
            "lon": lon,
            "alt": alt,
            "vn": vel_ned[:, 0],
            "ve": vel_ned[:, 1],
            "vd": vel_ned[:, 2],
        }

    def _recv_actuators(self):
        """Collect one HIL_ACTUATOR_CONTROLS per instance.

        Strict lockstep (block until the command arrives) only applies to instances that have
        already produced their first command. Instances still booting are polled non-blockingly:
        PX4 emits no actuator output until it has booted off a continuous HIL_SENSOR stream, so
        blocking on that first command would deadlock the exchange against PX4's own startup.
        """
        controls: list = [None] * self._B
        armed: list = [False] * self._B
        # Only instances already in lockstep are required to deliver a command this step.
        pending = {i for i in range(self._B) if self._established[i]}

        def consume(events):
            for key, _ in events:
                i = key.data
                try:
                    msgs = self._conns[i].poll_messages()
                except ConnectionError:
                    gs.raise_exception(f"PX4 instance {i} disconnected during lockstep.")
                for msg in msgs:
                    if msg.get_type() == _HIL_ACTUATOR_CONTROLS:
                        controls[i], armed[i] = mio.parse_actuator_controls(msg)
                        self._established[i] = True
                        pending.discard(i)

        # Non-blocking pass: lets still-booting instances report their first command (if any)
        # without forcing us to wait on them.
        consume(self._selector.select(timeout=0))

        # Blocking pass: hold strict lockstep for every instance already producing commands.
        deadline = time.monotonic() + self._options.connect_timeout
        while pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                gs.raise_exception(f"Lockstep timeout: no actuator command from PX4 instances {sorted(pending)}.")
            consume(self._selector.select(timeout=remaining))
        return controls, armed

    def _apply_rpm(self, rpm: np.ndarray) -> None:
        rpm_t = torch.as_tensor(rpm, dtype=gs.tc_float, device=gs.device)
        if self._scene.n_envs == 0:
            self._drone.set_propellers_rpm(rpm_t[0])
        else:
            self._drone.set_propellers_rpm(rpm_t)

    def _to_np(self, t) -> np.ndarray:
        """Tensor/array -> (B, ·) float64 numpy with an explicit batch axis."""
        if isinstance(t, torch.Tensor):
            arr = t.detach().cpu().numpy()
        else:
            arr = np.asarray(t)
        arr = arr.astype(np.float64, copy=False)
        if self._scene.n_envs == 0:
            arr = arr[None]
        return arr
