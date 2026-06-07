"""MAVLink encode/decode helpers for the PX4 simulator (HIL) link.

This module owns the wire protocol with PX4: it builds the sensor messages the simulator
must stream (``HIL_SENSOR``, ``HIL_GPS``, ``SYSTEM_TIME``, ``HEARTBEAT``) and parses the
``HIL_ACTUATOR_CONTROLS`` PX4 sends back. ``pymavlink`` is imported lazily so that merely
importing :mod:`genesis` never requires it.

One :class:`MavConn` wraps a single connected TCP socket plus its own MAVLink codec; the
bridge holds one per parallel env.
"""

import socket

# ``fields_updated`` bitmask for HIL_SENSOR (which sensor groups carry fresh data).
FIELD_ACCEL = 0x0007  # xacc, yacc, zacc
FIELD_GYRO = 0x0038  # xgyro, ygyro, zgyro
FIELD_MAG = 0x01C0  # xmag, ymag, zmag
FIELD_ABS_PRESSURE = 0x0200
FIELD_PRESSURE_ALT = 0x0800
FIELD_TEMPERATURE = 0x1000
FIELD_BARO = FIELD_ABS_PRESSURE | FIELD_PRESSURE_ALT | FIELD_TEMPERATURE

MAV_MODE_FLAG_SAFETY_ARMED = 128


def _load_mavlink():
    """Import and return the MAVLink v2 ``common`` dialect module (lazy)."""
    try:
        from pymavlink.dialects.v20 import common as mavlink2
    except ImportError as e:
        raise ImportError(
            "PX4 bridge requires 'pymavlink'. Install it with `pip install pymavlink` "
            "or `pip install genesis-world[px4]`."
        ) from e
    return mavlink2


class MavConn:
    """A MAVLink codec bound to one connected stream socket."""

    def __init__(self, sock: socket.socket, src_system: int = 1, src_component: int = 1):
        self._mav_mod = _load_mavlink()
        self.sock = sock
        self.sock.setblocking(True)
        # file=None: we never let MAVLink touch the socket directly; we pack/sendall ourselves.
        self.mav = self._mav_mod.MAVLink(None, srcSystem=src_system, srcComponent=src_component)
        self._recv_buf = bytearray()

    @property
    def fileno(self) -> int:
        return self.sock.fileno()

    # -------------------------------------------------------------------- send
    def _send(self, msg) -> None:
        self.sock.sendall(msg.pack(self.mav))

    def send_heartbeat(self) -> None:
        m = self._mav_mod
        self._send(
            m.MAVLink_heartbeat_message(
                type=m.MAV_TYPE_GENERIC,
                autopilot=m.MAV_AUTOPILOT_INVALID,
                base_mode=0,
                custom_mode=0,
                system_status=m.MAV_STATE_ACTIVE,
                mavlink_version=3,
            )
        )

    def send_system_time(self, time_usec: int) -> None:
        m = self._mav_mod
        self._send(m.MAVLink_system_time_message(time_unix_usec=time_usec, time_boot_ms=int(time_usec // 1000)))

    def send_hil_sensor(
        self,
        time_usec: int,
        acc,  # (3,) FRD specific force [m/s^2]
        gyro,  # (3,) FRD angular rate [rad/s]
        mag,  # (3,) FRD magnetic field [gauss]
        abs_pressure: float,  # [hPa]
        pressure_alt: float,  # [m]
        temperature: float,  # [degC]
        fields_updated: int,
    ) -> None:
        m = self._mav_mod
        self._send(
            m.MAVLink_hil_sensor_message(
                time_usec=int(time_usec),
                xacc=float(acc[0]),
                yacc=float(acc[1]),
                zacc=float(acc[2]),
                xgyro=float(gyro[0]),
                ygyro=float(gyro[1]),
                zgyro=float(gyro[2]),
                xmag=float(mag[0]),
                ymag=float(mag[1]),
                zmag=float(mag[2]),
                abs_pressure=float(abs_pressure),
                diff_pressure=0.0,
                pressure_alt=float(pressure_alt),
                temperature=float(temperature),
                fields_updated=int(fields_updated),
                id=0,
            )
        )

    def send_hil_gps(
        self,
        time_usec: int,
        lat_deg: float,
        lon_deg: float,
        alt_m: float,
        vn_ms: float,
        ve_ms: float,
        vd_ms: float,
        eph_m: float = 0.3,
        epv_m: float = 0.4,
        satellites: int = 12,
    ) -> None:
        import math

        m = self._mav_mod
        vel_ms = math.sqrt(vn_ms * vn_ms + ve_ms * ve_ms + vd_ms * vd_ms)
        cog = math.degrees(math.atan2(ve_ms, vn_ms)) % 360.0
        self._send(
            m.MAVLink_hil_gps_message(
                time_usec=int(time_usec),
                fix_type=3,
                lat=int(lat_deg * 1e7),
                lon=int(lon_deg * 1e7),
                alt=int(alt_m * 1e3),
                eph=int(eph_m * 100),
                epv=int(epv_m * 100),
                vel=int(vel_ms * 100),
                vn=int(vn_ms * 100),
                ve=int(ve_ms * 100),
                vd=int(vd_ms * 100),
                cog=int(cog * 100),
                satellites_visible=int(satellites),
                id=0,
                yaw=0,
            )
        )

    # -------------------------------------------------------------------- recv
    def poll_messages(self) -> list:
        """Read whatever bytes are currently available and return parsed messages.

        Returns an empty list if no complete message arrived. Raises ``ConnectionError``
        if the peer closed the socket.
        """
        try:
            chunk = self.sock.recv(4096)
        except BlockingIOError:
            return []
        if not chunk:
            raise ConnectionError("PX4 closed the simulator connection.")
        msgs = self.mav.parse_buffer(chunk)
        return msgs if msgs else []

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass


def parse_actuator_controls(msg):
    """Extract ``(controls, armed)`` from a HIL_ACTUATOR_CONTROLS message.

    ``controls`` is the raw normalized list of 16 channel outputs; ``armed`` reflects the
    PX4 safety-armed flag in the ``mode`` field.
    """
    controls = list(msg.controls)
    armed = bool(int(msg.mode) & MAV_MODE_FLAG_SAFETY_ARMED)
    return controls, armed
