"""PX4 SITL bridge plugin for Genesis.

Lets the real PX4 flight stack (Software-In-The-Loop) fly Genesis drones over PX4's
MAVLink simulator (HIL) lockstep protocol -- the same role the PX4 plugin plays in Gazebo
-- with one PX4 instance per parallel environment.

Exposed as ``gs.px4``::

    px4 = gs.px4.PX4Bridge(scene, drone, imu, gs.px4.PX4Options(px4_dir="~/PX4-Autopilot"))
    px4.start()
    ...
    px4.stop()

To *command* flight from the script (instead of an external ground station), drive PX4 over
its offboard endpoint with :class:`~genesis.ext.px4.OffboardPilot`::

    pilot = gs.px4.OffboardPilot(scene, px4.mavlink_endpoints)
    pilot.wait_until_connected()
    pilot.engage()                       # OFFBOARD + arm
    pilot.broadcast_position(0, 0, -3)   # climb to 3 m (NED down is negative)
"""

from .airframe import QuadXLayout, quad_x_layout, quad_x_motor_mapping, quad_x_propeller_spin, read_propeller_positions
from .bridge import PX4Bridge
from .offboard import OffboardClient, OffboardPilot
from .options import PX4Options

__all__ = [
    "PX4Bridge",
    "OffboardClient",
    "OffboardPilot",
    "PX4Options",
    "QuadXLayout",
    "quad_x_layout",
    "quad_x_motor_mapping",
    "quad_x_propeller_spin",
    "read_propeller_positions",
]
