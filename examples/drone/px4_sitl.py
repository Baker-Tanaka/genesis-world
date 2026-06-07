"""Fly Genesis drones with the real PX4 flight stack (SITL) over the MAVLink HIL link.

This is the Genesis equivalent of the Gazebo PX4 setup: each parallel environment is flown
by its own PX4 SITL instance. The bridge streams simulated IMU/GPS/baro to PX4 and applies
the motor commands PX4 returns, in lockstep.

Prerequisites
-------------
- A built PX4-Autopilot checkout (``make px4_sitl``) -- pass its path with ``--px4-dir``.
- ``pip install pymavlink`` (or ``pip install genesis-world[px4]``).
- To actually command flight, connect QGroundControl or MAVSDK to the offboard endpoints
  printed at startup (PX4 default ``udp://:14540`` for instance 0), then arm / takeoff.

Example
-------
    python examples/drone/px4_sitl.py --px4-dir ~/PX4-Autopilot --n-envs 2 -v
"""

import argparse

import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--px4-dir", type=str, required=True, help="Path to a built PX4-Autopilot checkout.")
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
    args = parser.parse_args()

    gs.init(backend=gs.gpu)

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
    drone = scene.add_entity(gs.morphs.Drone(file="urdf/drones/cf2x.urdf", pos=(0.0, 0.0, 0.2)))

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
            max_rpm=25000.0,
        ),
    )

    with px4:
        print("PX4 offboard endpoints (connect QGC / MAVSDK here):")
        for i, (host, port) in enumerate(px4.mavlink_endpoints):
            print(f"  instance {i}: udp://{host}:{port}")
        for _ in range(args.steps):
            scene.step()


if __name__ == "__main__":
    main()
