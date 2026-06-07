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

import numpy as np

import genesis as gs

from . import geo

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

    def disarm(self) -> None:
        """Request PX4 to disarm via MAV_CMD_COMPONENT_ARM_DISARM (param1 = 0)."""
        m = self._mavutil.mavlink
        self.conn.mav.command_long_send(
            self.target_system, self.target_component,
            m.MAV_CMD_COMPONENT_ARM_DISARM, 0,
            0, 0, 0, 0, 0, 0, 0,
        )  # fmt: skip

    def close(self) -> None:
        self.conn.close()


class OffboardPilot:
    """High-level offboard autopilot for one or more PX4 instances over a Genesis scene.

    Holds one :class:`OffboardClient` per parallel env and steps the scene as needed so PX4
    can boot and arm. Setpoints are broadcast to every instance (all drones follow the same
    command); reach individual instances through :attr:`clients` for per-drone control.
    """

    def __init__(self, scene, endpoints, bridge=None):
        self._scene = scene
        self._bridge = bridge
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

    def _fast_step(self) -> None:
        """Step physics with the viewer disabled (skips the realtime pacer + render)."""
        self._scene.step(update_visualizer=False)

    def _pump_step(self) -> None:
        """Advance PX4 while it is booting (disarmed), without physics or rendering.

        With a bridge we :meth:`~PX4Bridge.pump` it -- PX4 still gets its sensor<->actuator
        exchange and boots, while the drone holds its resting pose and the (comparatively
        expensive) physics solve and viewer are skipped. Without a bridge we fall back to a
        viewer-less step. Only safe while PX4 is disarmed: once it commands motors the control
        loop must be closed, so :meth:`engage` and flight use real physics steps.
        """
        if self._bridge is not None:
            self._bridge.pump()
        else:
            self._scene.step(update_visualizer=False)

    def wait_until_connected(self, timeout: float = 60.0, settle_steps: int = 50) -> None:
        """Wait until every PX4 instance has heartbeated (finished booting), physics paused.

        PX4's lockstep boot needs a steady sensor stream, but not a moving vehicle, so once the
        drone has settled (``settle_steps`` real steps to put it at rest and populate the
        sensors) we hold its pose and only :meth:`~PX4Bridge.pump` the MAVLink exchange -- no
        physics, no render -- which is far cheaper per step than a full ``scene.step()``.
        """
        if self._bridge is not None:
            for _ in range(settle_steps):
                self._scene.step(update_visualizer=False)

        gs.logger.info("Waiting for PX4 to boot and connect (physics paused)...")
        deadline = time.monotonic() + timeout
        while not all(c.connected for c in self.clients):
            self._pump()
            self._pump_step()
            if time.monotonic() > deadline:
                gs.raise_exception(
                    f"No PX4 heartbeat on the offboard endpoint(s) within {timeout:.0f}s. "
                    "Is PX4 SITL running and connected to the bridge?"
                )
        gs.logger.info(f"PX4 connected ({len(self.clients)} instance(s)).")

    def engage(self, warmup_steps: int = 200, timeout: float = 30.0) -> None:
        """Stream a hover setpoint, then switch every instance to OFFBOARD and arm it.

        PX4 rejects OFFBOARD unless a setpoint stream is already flowing, so we warm up the
        stream first, then re-request mode + arm each step until every instance reports armed.
        Runs with real physics (the drone responds to PX4 once armed) but no viewer, so it is
        fast while keeping the control loop closed.
        """
        for _ in range(warmup_steps):
            self.broadcast_velocity(0.0, 0.0, 0.0)
            self._fast_step()

        gs.logger.info("Arming and switching to OFFBOARD...")
        deadline = time.monotonic() + timeout
        step = 0
        while not all(c.armed for c in self.clients):
            # Stream setpoints every step (offboard stays alive), but re-request mode/arm only
            # a few times a second -- doing it every step would spam PX4 with commands.
            request = (step % 25) == 0
            for c in self.clients:
                c.pump()
                c.send_velocity_ned(0.0, 0.0, 0.0)
                if request and not c.armed:
                    c.set_offboard_mode()
                    c.arm()
            self._fast_step()
            step += 1
            if time.monotonic() > deadline:
                gs.logger.warning("Some PX4 instance(s) did not confirm armed; continuing anyway.")
                break
        else:
            gs.logger.info("Armed; OFFBOARD engaged.")

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

    def fly_waypoints(
        self,
        drone,
        waypoints_enu,
        arrival_radius: float = 0.3,
        max_steps: int = 200_000,
        yaw: float = 0.0,
        hold: bool = True,
        check_every: int = 10,
    ) -> int:
        """Fly a list of local-ENU waypoints, advancing when every drone is within range.

        Each waypoint is converted to PX4 local NED and streamed as a position setpoint;
        arrival is judged in Genesis ENU from ``drone.get_pos()`` so it is independent of
        PX4's internal origin. With several instances the route advances only once *all* are
        within ``arrival_radius``. Returns the number of steps taken.

        Parameters
        ----------
        drone : DroneEntity
            The flown drone, used only for arrival detection via ``get_pos()``.
        waypoints_enu : sequence of (x, y, z)
            Target positions in the local ENU frame (metres relative to the start).
        hold : bool
            If True keep streaming the final waypoint until ``max_steps``; if False return as
            soon as it is reached.
        check_every : int
            Test arrival every N steps. Each test reads ``drone.get_pos()`` (a GPU<->CPU sync),
            so checking at ~25 Hz instead of every step keeps the step loop lighter.
        """
        targets_ned = [geo.enu_to_ned(np.asarray(wp, dtype=np.float64)) for wp in waypoints_enu]
        targets_enu = [np.asarray(wp, dtype=np.float64) for wp in waypoints_enu]
        last = len(waypoints_enu) - 1
        idx = 0
        reached_final = False
        gs.logger.info(f"Flying waypoint 0/{last}: {tuple(targets_enu[0])} (ENU).")
        for step in range(max_steps):
            ned = targets_ned[idx]
            self.broadcast_position(ned[0], ned[1], ned[2], yaw)
            self._scene.step()

            if step % check_every:
                continue
            # drone.get_pos() is (3,) for n_envs=0, else (B, 3).
            pos = drone.get_pos().detach().cpu().numpy().reshape(-1, 3)
            if np.all(np.linalg.norm(pos - targets_enu[idx], axis=1) < arrival_radius):
                if idx < last:
                    idx += 1
                    gs.logger.info(
                        f"Reached waypoint {idx - 1}/{last}; flying to {idx}/{last}: {tuple(targets_enu[idx])} (ENU)."
                    )
                elif not reached_final:
                    reached_final = True
                    gs.logger.info(f"Reached final waypoint {last}/{last}.")
                    if not hold:
                        return step + 1
        return max_steps

    def disarm(self) -> None:
        """Request every instance to disarm."""
        for c in self.clients:
            c.disarm()

    def close(self) -> None:
        for c in self.clients:
            c.close()
