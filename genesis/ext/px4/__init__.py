"""PX4 SITL bridge plugin for Genesis.

Lets the real PX4 flight stack (Software-In-The-Loop) fly Genesis drones over PX4's
MAVLink simulator (HIL) lockstep protocol -- the same role the PX4 plugin plays in Gazebo
-- with one PX4 instance per parallel environment.

Exposed as ``gs.px4``::

    px4 = gs.px4.PX4Bridge(scene, drone, imu, gs.px4.PX4Options(px4_dir="~/PX4-Autopilot"))
    px4.start()
    ...
    px4.stop()
"""

from .bridge import PX4Bridge
from .options import PX4Options

__all__ = ["PX4Bridge", "PX4Options"]
