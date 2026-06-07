"""Typed configuration for the PX4 SITL bridge."""

import os
from typing import Optional

from pydantic import Field

import genesis as gs
from genesis.options.options import Options


class PX4Options(Options):
    """Configuration for :class:`~genesis.ext.px4.bridge.PX4Bridge`.

    A single options object configures every parallel PX4 instance; per-instance values
    (TCP ports, instance ids, log paths) are derived from these by adding the env index.
    """

    # ----- PX4 process / airframe ---------------------------------------------------
    px4_dir: str = ""
    """Path to a built PX4-Autopilot checkout (the directory containing ``build/``).
    Required when ``auto_spawn`` is True."""

    px4_binary: str = ""
    """Explicit path to the ``px4`` binary. If empty, derived from ``px4_dir`` and
    ``build_dir`` as ``<px4_dir>/<build_dir>/bin/px4``."""

    build_dir: str = "build/px4_sitl_default"
    """Build subdirectory of ``px4_dir`` holding ``bin/px4`` and ``etc/`` (rootfs)."""

    airframe: str = "none_iris"
    """PX4 simulator model, exported as ``PX4_SIM_MODEL``. Use a non-``gz_`` airframe (e.g.
    ``none_iris``) so PX4 starts ``simulator_mavlink`` and connects out to the Genesis bridge
    over TCP. A ``gz_*`` model makes PX4 launch Gazebo instead and never connects."""

    sys_autostart: Optional[int] = None
    """PX4 ``PX4_SYS_AUTOSTART`` airframe id. If None, PX4 picks it from ``airframe``."""

    auto_spawn: bool = True
    """If True the bridge launches the PX4 processes; if False it only listens and the
    user starts PX4 (or a fake server, in tests) externally."""

    extra_px4_env: dict = Field(default_factory=dict)
    """Extra environment variables merged into each spawned PX4 process."""

    headless: bool = True
    """Run PX4 without its interactive pxh shell (sets ``HEADLESS=1``)."""

    # ----- Networking ---------------------------------------------------------------
    host: str = "127.0.0.1"
    """Interface the bridge's simulator TCP servers bind to."""

    base_tcp_port: int = 4560
    """Simulator MAVLink TCP port for env 0; env ``i`` uses ``base_tcp_port + i``."""

    connect_timeout: float = 60.0
    """Seconds to wait for each PX4 instance to connect after spawning."""

    # ----- Geodety / world ----------------------------------------------------------
    home: tuple[float, float, float] = (47.397742, 8.545594, 488.0)
    """Reference (latitude_deg, longitude_deg, altitude_m_AMSL) mapping the local origin
    onto the globe for simulated GPS."""

    world_frame: str = "ENU"
    """Convention of the Genesis world frame. Only ``ENU`` is currently supported."""

    magnetic_field: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Unused placeholder; the magnetometer reading is taken from the attached IMU sensor
    whose own ``magnetic_field`` option defines the reference field."""

    # ----- Actuator mapping ---------------------------------------------------------
    max_rpm: float = 25000.0
    """Propeller RPM corresponding to a normalized PX4 motor output of 1.0."""

    motor_mapping: Optional[tuple[int, ...]] = None
    """Maps PX4 motor output channel -> Genesis propeller index. ``motor_mapping[k]`` is
    the propeller fed by PX4 control channel ``k``. If None, identity (channel k ->
    propeller k) is used for the first ``n_propellers`` channels."""

    arm_gate: bool = True
    """If True, propeller RPMs are forced to zero whenever PX4 reports a disarmed state."""

    # ----- Sensor rates -------------------------------------------------------------
    gps_hz: float = 0.0
    """GPS update rate [Hz]. ``0`` means once per simulation step."""

    mag_hz: float = 0.0
    """Magnetometer update rate [Hz]. ``0`` means once per simulation step."""

    baro_hz: float = 0.0
    """Barometer update rate [Hz]. ``0`` means once per simulation step."""

    def model_post_init(self, context) -> None:
        if self.world_frame != "ENU":
            gs.raise_exception(f"PX4Options.world_frame only supports 'ENU', got '{self.world_frame}'.")
        if self.auto_spawn:
            if not self.px4_dir:
                gs.raise_exception("PX4Options.px4_dir is required when auto_spawn=True.")
            if not os.path.isdir(os.path.expanduser(self.px4_dir)):
                gs.raise_exception(f"PX4Options.px4_dir does not exist: {self.px4_dir}")

    # ----- Derived helpers ----------------------------------------------------------
    def resolved_binary(self) -> str:
        """Absolute path to the px4 binary, derived from the options."""
        if self.px4_binary:
            return os.path.expanduser(self.px4_binary)
        return os.path.join(os.path.expanduser(self.px4_dir), self.build_dir, "bin", "px4")

    def rootfs_dir(self) -> str:
        """Working directory (rootfs) PX4 is launched from."""
        return os.path.join(os.path.expanduser(self.px4_dir), self.build_dir)
