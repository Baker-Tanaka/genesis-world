"""Fly Genesis drones with the real PX4 flight stack (SITL) over the MAVLink HIL link.

This is the Genesis equivalent of the Gazebo PX4 setup: each parallel environment is flown
by its own PX4 SITL instance. :class:`~genesis.ext.px4.PX4Bridge` streams simulated
IMU/GPS/baro to PX4 and applies the motor commands PX4 returns, in lockstep, while
:class:`~genesis.ext.px4.OffboardPilot` arms PX4, switches it to OFFBOARD mode and flies a
list of waypoints over PX4's offboard MAVLink endpoint (``udp://:14540`` for instance 0).

Any X-quad URDF works: :func:`gs.px4.quad_x_layout` derives the PX4 motor mapping, propeller
spin directions and thrust scale from the URDF geometry, so ``--drone`` can point at a
different airframe with no hand-tuning.

Prerequisites
-------------
- A built PX4-Autopilot checkout (``make px4_sitl``) -- pass its path with ``--px4-dir``.
- ``pip install pymavlink`` (or ``pip install genesis-world[px4]``).

Examples
--------
    # Default Crazyflie route, with the viewer and balloon waypoint markers.
    python examples/drone/px4_sitl.py --px4-dir ~/PX4-Autopilot -v

    # A different (heavier) X-quad URDF.
    python examples/drone/px4_sitl.py --px4-dir ~/PX4-Autopilot --drone urdf/drones/racer.urdf -v

    # Two drones flying the same route.
    python examples/drone/px4_sitl.py --px4-dir ~/PX4-Autopilot --n-envs 2
"""

import argparse

import genesis as gs

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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--px4-dir", type=str, required=True, help="Path to a built PX4-Autopilot checkout.")
    parser.add_argument(
        "--drone",
        type=str,
        default="urdf/drones/cf2x.urdf",
        help="X-quad drone URDF (assets-relative or absolute). PX4 mapping/spin/thrust scale "
        "are auto-derived from its geometry. e.g. urdf/drones/cf2x.urdf, urdf/drones/racer.urdf.",
    )
    parser.add_argument(
        "--airframe",
        type=str,
        default="none_iris",
        help="PX4 SIM model (PX4_SIM_MODEL). Use a non-'gz_' airframe (e.g. none_iris) so PX4 "
        "connects to Genesis via simulator_mavlink instead of launching Gazebo.",
    )
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
    args = parser.parse_args()

    waypoints_enu = parse_waypoints(args.waypoints) if args.waypoints else DEFAULT_WAYPOINTS_ENU

    gs.init(backend=gs.gpu)

    # Derive the PX4 quad-X configuration (channel mapping, spins, thrust scale) from the URDF.
    layout = gs.px4.quad_x_layout(args.drone)
    max_rpm = layout.max_rpm if layout.max_rpm is not None else 25000.0
    gs.logger.info(
        f"Drone '{args.drone}': motor_mapping={layout.motor_mapping}, "
        f"propellers_spin={layout.propellers_spin}, max_rpm={max_rpm:.0f}."
    )

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.004),  # 250 Hz, PX4-friendly sensor rate
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.0, 0.0, 2.0),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
        ),
        show_viewer=args.vis,
    )

    scene.add_entity(gs.morphs.Plane())
    drone = scene.add_entity(
        gs.morphs.Drone(file=args.drone, pos=(0.0, 0.0, 0.2), propellers_spin=layout.propellers_spin)
    )
    add_waypoint_markers(scene, waypoints_enu)

    # IMU supplies accel/gyro/mag for HIL_SENSOR. magnetic_field is in gauss (PX4 units).
    imu = scene.add_sensor(
        gs.sensors.IMU(
            entity_idx=drone.idx,
            link_idx_local=0,
            magnetic_field=(0.21, 0.0, 0.43),  # ~Zurich field, gauss
        )
    )

    scene.build(n_envs=args.n_envs)

    px4 = gs.px4.PX4Bridge(
        scene,
        drone,
        imu,
        gs.px4.PX4Options(
            px4_dir=args.px4_dir,
            airframe=args.airframe,
            home=(47.397742, 8.545594, 488.0),
            max_rpm=max_rpm,
            motor_mapping=layout.motor_mapping,
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
                pilot.fly_waypoints(drone, waypoints_enu, arrival_radius=args.arrival_radius, max_steps=args.steps)
            except KeyboardInterrupt:
                gs.logger.info("Interrupted; releasing offboard control.")


if __name__ == "__main__":
    main()
