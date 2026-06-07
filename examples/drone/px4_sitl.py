"""Fly Genesis drones with the real PX4 flight stack (SITL) over the MAVLink HIL link.

This is the Genesis equivalent of the Gazebo PX4 setup: each parallel environment is flown
by its own PX4 SITL instance. The :class:`~genesis.ext.px4.PX4Bridge` streams simulated
IMU/GPS/baro to PX4 and applies the motor commands PX4 returns, in lockstep.

On top of that HIL link this example *commands* flight with
:class:`~genesis.ext.px4.OffboardPilot`, which talks to PX4's offboard MAVLink endpoint
(``udp://:14540`` for instance 0): it arms PX4, switches it to OFFBOARD mode and streams
setpoints, in one of two modes:

- ``--mode waypoint`` (default): fly a fixed list of waypoints (a takeoff + square route by
  default; override with ``--waypoints``).
- ``--mode interactive``: fly by keyboard in the viewer, sending velocity setpoints.

Prerequisites
-------------
- A built PX4-Autopilot checkout (``make px4_sitl``) -- pass its path with ``--px4-dir``.
- ``pip install pymavlink`` (or ``pip install genesis-world[px4]``).

Examples
--------
    # Autonomous waypoint flight, with the viewer.
    python examples/drone/px4_sitl.py --px4-dir ~/PX4-Autopilot -v

    # Keyboard flight.
    python examples/drone/px4_sitl.py --px4-dir ~/PX4-Autopilot --mode interactive

    # Two drones flying the same route.
    python examples/drone/px4_sitl.py --px4-dir ~/PX4-Autopilot --n-envs 2
"""

import argparse

import numpy as np

import genesis as gs
from genesis.ext.px4 import geo
from genesis.vis.keybindings import Key, KeyAction, Keybind

# Default waypoint route, local ENU metres relative to the start: take off, fly a square at
# altitude, then descend. Overridable with --waypoints.
DEFAULT_WAYPOINTS_ENU = [
    (0.0, 0.0, 3.0),
    (3.0, 0.0, 3.0),
    (3.0, 3.0, 3.0),
    (0.0, 3.0, 3.0),
    (0.0, 0.0, 3.0),
    (0.0, 0.0, 1.0),
]


def parse_waypoints(spec: str) -> list[tuple[float, float, float]]:
    """Parse a ``"x,y,z;x,y,z;..."`` ENU waypoint string into a list of (x, y, z) tuples."""
    waypoints = []
    for chunk in spec.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = chunk.split(",")
        if len(parts) != 3:
            raise ValueError(f"Waypoint '{chunk}' must have exactly 3 comma-separated values (x,y,z).")
        waypoints.append(tuple(float(p) for p in parts))
    if not waypoints:
        raise ValueError("--waypoints did not contain any waypoints.")
    return waypoints


class VelocityCommand:
    """Mutable keyboard-driven velocity setpoint (world NED) for interactive mode."""

    def __init__(self, speed: float = 1.5, climb: float = 1.0, yaw_rate: float = 0.6):
        self._speed = speed
        self._climb = climb
        self._yaw_rate_mag = yaw_rate
        # Accumulated key state as unit contributions; scaled to m/s in value().
        self._n = 0.0  # +north
        self._e = 0.0  # +east
        self._u = 0.0  # +up (ENU); converted to -down for NED
        self._yaw = 0.0  # +clockwise yaw rate

    def add(self, dn: float, de: float, du: float, dyaw: float) -> None:
        self._n += dn
        self._e += de
        self._u += du
        self._yaw += dyaw

    def value(self) -> tuple[float, float, float, float]:
        """Return (vn, ve, vd, yaw_rate), clipped to one unit of input per axis."""
        vn = np.clip(self._n, -1.0, 1.0) * self._speed
        ve = np.clip(self._e, -1.0, 1.0) * self._speed
        vd = -np.clip(self._u, -1.0, 1.0) * self._climb  # ENU up -> NED down
        yaw_rate = np.clip(self._yaw, -1.0, 1.0) * self._yaw_rate_mag
        return vn, ve, vd, yaw_rate


def add_waypoint_markers(scene, waypoints_enu):
    """Place a fixed, non-colliding sphere "balloon" at each ENU waypoint to show the route.

    Colored green (first) -> red (last) so the order is visible. Must be called before
    scene.build(). Mirrors the target-sphere pattern in examples/drone/hover_env.py.
    """
    n = len(waypoints_enu)
    markers = []
    for i, (x, y, z) in enumerate(waypoints_enu):
        frac = i / max(1, n - 1)
        markers.append(
            scene.add_entity(
                morph=gs.morphs.Mesh(
                    file="meshes/sphere.obj",
                    pos=(x, y, z),
                    scale=0.12,
                    fixed=True,  # stay put (no gravity) -- pure visual marker
                    collision=False,  # never interact with the drone
                ),
                surface=gs.surfaces.Rough(
                    diffuse_texture=gs.textures.ColorTexture(color=(frac, 1.0 - frac, 0.2)),
                ),
            )
        )
    return markers


def run_waypoint(pilot, scene, drone, waypoints_enu, args) -> None:
    """Fly the shared ENU waypoint route; advance when all drones reach the current target."""
    targets_ned = [geo.enu_to_ned(np.asarray(wp, dtype=np.float64)) for wp in waypoints_enu]
    last = len(waypoints_enu) - 1
    idx = 0
    route_done = False
    gs.logger.info(f"Flying waypoint 0/{last}: {waypoints_enu[0]} (ENU).")
    for _ in range(args.steps):
        ned = targets_ned[idx]
        pilot.broadcast_position(ned[0], ned[1], ned[2])
        scene.step()

        # Arrival check in Genesis ENU (drone.get_pos() is (3,) for n_envs=0, else (B, 3)).
        pos = drone.get_pos().detach().cpu().numpy().reshape(-1, 3)
        if np.all(np.linalg.norm(pos - np.asarray(waypoints_enu[idx]), axis=1) < args.arrival_radius):
            if idx < last:
                idx += 1
                gs.logger.info(
                    f"Reached waypoint {idx - 1}/{last}; flying to {idx}/{last}: {waypoints_enu[idx]} (ENU)."
                )
            elif not route_done:
                route_done = True
                gs.logger.info(f"Reached final waypoint {last}/{last}; holding position.")
    gs.logger.info("Step budget exhausted; exiting.")


def register_interactive_keybinds(scene, command: VelocityCommand, stop) -> None:
    """Register HOLD/RELEASE keybinds mapping keys to velocity contributions."""

    def axis(name: str, key: Key, delta: tuple[float, float, float, float]):
        d = np.array(delta, dtype=np.float64)
        return [
            Keybind(f"{name}_hold", key, KeyAction.HOLD, callback=command.add, args=tuple(d)),
            Keybind(f"{name}_release", key, KeyAction.RELEASE, callback=command.add, args=tuple(-d)),
        ]

    # Q/E (yaw) are chosen to avoid the viewer's default A=camera_rotation, D=wireframe binds.
    scene.viewer.register_keybinds(
        *axis("forward", Key.UP, (1.0, 0.0, 0.0, 0.0)),
        *axis("backward", Key.DOWN, (-1.0, 0.0, 0.0, 0.0)),
        *axis("right", Key.RIGHT, (0.0, 1.0, 0.0, 0.0)),
        *axis("left", Key.LEFT, (0.0, -1.0, 0.0, 0.0)),
        *axis("up", Key.SPACE, (0.0, 0.0, 1.0, 0.0)),
        *axis("down", Key.LSHIFT, (0.0, 0.0, -1.0, 0.0)),
        *axis("yaw_right", Key.E, (0.0, 0.0, 0.0, 1.0)),
        *axis("yaw_left", Key.Q, (0.0, 0.0, 0.0, -1.0)),
        Keybind("quit", Key.ESCAPE, KeyAction.RELEASE, callback=stop),
    )


def run_interactive(pilot, scene, drone, args) -> None:
    """Fly the single drone by keyboard, streaming velocity setpoints to PX4."""
    command = VelocityCommand(speed=args.cruise_speed)
    is_running = True

    def stop():
        nonlocal is_running
        is_running = False

    scene.viewer.follow_entity(drone)
    register_interactive_keybinds(scene, command, stop)

    print("\nInteractive PX4 flight (world-NED velocity setpoints):")
    print("  ↑ / ↓     forward / backward (north / south)")
    print("  → / ←     right / left (east / west)")
    print("  space     ascend")
    print("  shift     descend")
    print("  E / Q     yaw right / left")
    print("  escape    quit")

    for _ in range(args.steps):
        if not is_running:
            break
        pilot.broadcast_velocity(*command.value())
        scene.step()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--px4-dir", type=str, required=True, help="Path to a built PX4-Autopilot checkout.")
    parser.add_argument(
        "--airframe",
        type=str,
        default="none_iris",
        help="PX4 SIM model (PX4_SIM_MODEL). Use a non-'gz_' airframe (e.g. none_iris) so PX4 "
        "connects to Genesis via simulator_mavlink instead of launching Gazebo.",
    )
    parser.add_argument("--mode", choices=("waypoint", "interactive"), default="waypoint", help="Flight driver.")
    parser.add_argument("--n-envs", type=int, default=1, help="Number of parallel drones / PX4 instances.")
    parser.add_argument("--steps", type=int, default=200000)
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument(
        "--waypoints",
        type=str,
        default=None,
        help='Waypoint route as ENU "x,y,z;x,y,z;...". Defaults to a takeoff + square route.',
    )
    parser.add_argument("--warmup-steps", type=int, default=200, help="Setpoint-stream warm-up steps before arming.")
    parser.add_argument("--arrival-radius", type=float, default=0.3, help="Waypoint arrival threshold [m].")
    parser.add_argument("--cruise-speed", type=float, default=1.5, help="Interactive horizontal speed [m/s].")
    args = parser.parse_args()

    waypoints_enu = parse_waypoints(args.waypoints) if args.waypoints else DEFAULT_WAYPOINTS_ENU

    n_envs = args.n_envs
    show_viewer = args.vis
    if args.mode == "interactive":
        show_viewer = True
        if n_envs != 1:
            gs.logger.warning("Interactive mode controls a single drone; forcing --n-envs 1.")
            n_envs = 1

    gs.init(backend=gs.gpu)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.004),  # 250 Hz, PX4-friendly sensor rate
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.0, 0.0, 2.0),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
        ),
        show_viewer=show_viewer,
    )

    scene.add_entity(gs.morphs.Plane())
    # cf2x's default propeller spins (-1, 1, -1, 1) already match PX4's iris yaw convention once
    # motor_mapping (below) aligns the channels to the right arms, so no spin override is needed.
    drone = scene.add_entity(gs.morphs.Drone(file="urdf/drones/cf2x.urdf", pos=(0.0, 0.0, 0.2)))

    # Visualize the route with balloon-like spheres at each waypoint (waypoint mode only).
    if args.mode == "waypoint":
        add_waypoint_markers(scene, waypoints_enu)

    # IMU supplies accel/gyro/mag for HIL_SENSOR. magnetic_field is in gauss (PX4 units).
    imu = scene.add_sensor(
        gs.sensors.IMU(
            entity_idx=drone.idx,
            link_idx_local=0,
            magnetic_field=(0.21, 0.0, 0.43),  # ~Zurich field, gauss
        )
    )

    scene.build(n_envs=n_envs)

    px4 = gs.px4.PX4Bridge(
        scene,
        drone,
        imu,
        gs.px4.PX4Options(
            px4_dir=args.px4_dir,
            airframe=args.airframe,
            home=(47.397742, 8.545594, 488.0),
            max_rpm=25000.0,
            # PX4 iris motor channel -> cf2x propeller index, matched by arm position:
            # ch0 front-right=prop0, ch1 rear-left=prop2, ch2 front-left=prop3, ch3 rear-right=prop1.
            # Identity mapping puts 3 of 4 motors on the wrong arm and the drone tumbles on takeoff.
            motor_mapping=(0, 2, 3, 1),
        ),
    )

    with px4:
        print("PX4 offboard endpoints (also reachable from QGC / MAVSDK):")
        for i, (host, port) in enumerate(px4.mavlink_endpoints):
            print(f"  instance {i}: udp://{host}:{port}")

        with gs.px4.OffboardPilot(scene, px4.mavlink_endpoints) as pilot:
            try:
                pilot.wait_until_connected()
                pilot.engage(warmup_steps=args.warmup_steps)
                if args.mode == "interactive":
                    run_interactive(pilot, scene, drone, args)
                else:
                    run_waypoint(pilot, scene, drone, waypoints_enu, args)
            except KeyboardInterrupt:
                gs.logger.info("Interrupted; releasing offboard control.")


if __name__ == "__main__":
    main()
