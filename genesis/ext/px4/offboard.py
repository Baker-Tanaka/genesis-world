"""Offboard (ground-station side) control for PX4 SITL.

Where :class:`~genesis.ext.px4.PX4Bridge` is the *simulator* side of PX4 -- the HIL
sensor<->actuator lockstep -- this module is the *offboard* side: it connects to PX4's
offboard MAVLink endpoint (``udp 14540 + instance``), arms PX4, switches it to OFFBOARD
mode and streams position/velocity setpoints, so a Genesis script can fly the drone with no
external ground station (QGroundControl / MAVSDK).

PX4 only finishes booting -- and only then starts emitting its offboard MAVLink stream --
once the HIL sensor stream is flowing, which in Genesis happens during ``scene.step()``.
:class:`OffboardPilot` therefore steps the scene while waiting for PX4 to connect and while
arming; a bare :class:`OffboardClient` leaves stepping to the caller.

Exposed as ``gs.px4``::

    pilot = gs.px4.OffboardPilot(scene, px4.mavlink_endpoints)
    pilot.wait_until_connected()
    pilot.engage()                       # OFFBOARD + arm, all instances
    pilot.broadcast_position(0, 0, -3)   # fly up to 3 m (NED: down is negative)
"""

import time

import genesis as gs

# SET_POSITION_TARGET_LOCAL_NED ``type_mask`` bits (a set bit means "ignore this field").
_IGNORE_POS = 0b0000_0000_0000_0111
_IGNORE_VEL = 0b0000_0000_0011_1000
_IGNORE_ACC = 0b0000_0001_1100_0000
_IGNORE_YAW = 0b0000_0100_0000_0000
_IGNORE_YAW_RATE = 0b0000_1000_0000_0000

# PX4 custom main mode index for OFFBOARD (see PX4 commander_state / px4_custom_mode.h).
_PX4_MAIN_MODE_OFFBOARD = 6


class OffboardClient:
    """pymavlink offboard client for a single PX4 SITL instance.

    PX4 SITL transmits its offboard MAVLink stream to ``udp 14540 + instance`` (see PX4's
    ``px4-rc.mavlink``), so we bind and listen there (``udpin``). The peer address is learnt
    from PX4's first packet, so :meth:`pump` must observe a heartbeat (see :attr:`connected`)
    before any setpoint is sent. All setpoints are expressed in PX4's local NED frame.
    """

    def __init__(self, host: str, port: int):
        from pymavlink import mavutil

        self._mavutil = mavutil
        self.conn = mavutil.mavlink_connection(f"udpin:{host}:{port}")
        self.target_system = 1
        self.target_component = 1
        self._connected = False
        self._armed = False

    @property
    def connected(self) -> bool:
        """True once PX4's first heartbeat has been received (it has booted)."""
        return self._connected

    @property
    def armed(self) -> bool:
        """Latest PX4 armed state, as of the last :meth:`pump`."""
        return self._armed

    # -------------------------------------------------------------------- recv
    def pump(self) -> None:
        """Drain pending telemetry, tracking PX4's connection and armed state."""
        armed_flag = self._mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
        while True:
            msg = self.conn.recv_match(type="HEARTBEAT", blocking=False)
            if msg is None:
                break
            if not self._connected:
                self.target_system = msg.get_srcSystem()
                self.target_component = msg.get_srcComponent()
                self._connected = True
            self._armed = bool(msg.base_mode & armed_flag)

    # -------------------------------------------------------------------- send
    def _send_setpoint(self, type_mask, pos, vel, yaw, yaw_rate) -> None:
        self.conn.mav.set_position_target_local_ned_send(
            0,  # time_boot_ms (autopilot ignores it for setpoints)
            self.target_system,
            self.target_component,
            self._mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            type_mask,
            pos[0], pos[1], pos[2],
            vel[0], vel[1], vel[2],
            0.0, 0.0, 0.0,  # acceleration (ignored)
            yaw, yaw_rate,
        )  # fmt: skip

    def send_position_ned(self, north: float, east: float, down: float, yaw: float = 0.0) -> None:
        """Stream a position setpoint (NED metres) with a fixed yaw."""
        mask = _IGNORE_VEL | _IGNORE_ACC | _IGNORE_YAW_RATE
        self._send_setpoint(mask, (north, east, down), (0.0, 0.0, 0.0), yaw, 0.0)

    def send_velocity_ned(self, vn: float, ve: float, vd: float, yaw_rate: float = 0.0) -> None:
        """Stream a velocity setpoint (NED m/s) with a yaw rate."""
        mask = _IGNORE_POS | _IGNORE_ACC | _IGNORE_YAW
        self._send_setpoint(mask, (0.0, 0.0, 0.0), (vn, ve, vd), 0.0, yaw_rate)

    def set_offboard_mode(self) -> None:
        """Request PX4 OFFBOARD main mode via MAV_CMD_DO_SET_MODE."""
        m = self._mavutil.mavlink
        self.conn.mav.command_long_send(
            self.target_system, self.target_component,
            m.MAV_CMD_DO_SET_MODE, 0,
            m.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, _PX4_MAIN_MODE_OFFBOARD, 0, 0, 0, 0, 0,
        )  # fmt: skip

    def arm(self) -> None:
        """Request PX4 to arm via MAV_CMD_COMPONENT_ARM_DISARM (param1 = 1)."""
        m = self._mavutil.mavlink
        self.conn.mav.command_long_send(
            self.target_system, self.target_component,
            m.MAV_CMD_COMPONENT_ARM_DISARM, 0,
            1, 0, 0, 0, 0, 0, 0,
        )  # fmt: skip

    def close(self) -> None:
        self.conn.close()


class OffboardPilot:
    """High-level offboard autopilot for one or more PX4 instances over a Genesis scene.

    Holds one :class:`OffboardClient` per parallel env and steps the scene as needed so PX4
    can boot and arm. Setpoints are broadcast to every instance (all drones follow the same
    command); reach individual instances through :attr:`clients` for per-drone control.
    """

    def __init__(self, scene, endpoints):
        self._scene = scene
        self.clients: list[OffboardClient] = [OffboardClient(host, port) for host, port in endpoints]

    def __len__(self) -> int:
        return len(self.clients)

    def __enter__(self) -> "OffboardPilot":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _pump(self) -> None:
        for c in self.clients:
            c.pump()

    def _boot_step(self) -> None:
        """Advance the lockstep exchange as fast as possible, bypassing the viewer.

        PX4's boot/arm is dead time the user does not need to watch, and it costs many sim
        steps (EKF/GPS convergence). Stepping with ``update_visualizer=False`` skips the
        realtime pacer and the (often software) render, so PX4 advances its clock at the
        physics rate instead of being throttled to wall-clock real time.
        """
        self._scene.step(update_visualizer=False)

    def wait_until_connected(self, timeout: float = 60.0) -> None:
        """Step the scene until every PX4 instance has heartbeated (finished booting).

        PX4's lockstep boot needs the HIL sensor stream, which only flows while the scene is
        stepping, so we drive ``scene.step()`` here rather than blocking on a bare socket.
        """
        deadline = time.monotonic() + timeout
        while not all(c.connected for c in self.clients):
            self._pump()
            self._boot_step()
            if time.monotonic() > deadline:
                gs.raise_exception(
                    f"No PX4 heartbeat on the offboard endpoint(s) within {timeout:.0f}s. "
                    "Is PX4 SITL running and connected to the bridge?"
                )
        gs.logger.info(f"All {len(self.clients)} PX4 instance(s) connected on the offboard link.")

    def engage(self, warmup_steps: int = 200, timeout: float = 30.0) -> None:
        """Stream a hover setpoint, then switch every instance to OFFBOARD and arm it.

        PX4 rejects OFFBOARD unless a setpoint stream is already flowing, so we warm up the
        stream first, then re-request mode + arm each step until every instance reports armed.
        Runs uncapped (no viewer) so the EKF/GPS preflight checks clear quickly.
        """
        for _ in range(warmup_steps):
            self.broadcast_velocity(0.0, 0.0, 0.0)
            self._boot_step()

        gs.logger.info("Requesting OFFBOARD mode and arming.")
        deadline = time.monotonic() + timeout
        step = 0
        while not all(c.armed for c in self.clients):
            # Stream setpoints every step (offboard stays alive), but re-request mode/arm only
            # a few times a second -- the loop runs uncapped, so doing it every step would spam
            # PX4 with thousands of commands per second.
            request = (step % 25) == 0
            for c in self.clients:
                c.pump()
                c.send_velocity_ned(0.0, 0.0, 0.0)
                if request and not c.armed:
                    c.set_offboard_mode()
                    c.arm()
            self._boot_step()
            step += 1
            if time.monotonic() > deadline:
                gs.logger.warning("Some PX4 instance(s) did not confirm armed; continuing anyway.")
                break
        else:
            gs.logger.info("PX4 armed and in OFFBOARD.")

    def broadcast_position(self, north: float, east: float, down: float, yaw: float = 0.0) -> None:
        """Pump telemetry and send the same NED position setpoint to every instance."""
        for c in self.clients:
            c.pump()
            c.send_position_ned(north, east, down, yaw)

    def broadcast_velocity(self, vn: float, ve: float, vd: float, yaw_rate: float = 0.0) -> None:
        """Pump telemetry and send the same NED velocity setpoint to every instance."""
        for c in self.clients:
            c.pump()
            c.send_velocity_ned(vn, ve, vd, yaw_rate)

    def close(self) -> None:
        for c in self.clients:
            c.close()
