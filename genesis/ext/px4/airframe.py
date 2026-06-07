"""Auto-configure a Genesis quadrotor for the PX4 'iris' / quad-X airframe.

PX4's default multirotor airframe (e.g. ``none_iris``) is an X-quad whose four motor output
channels map to specific arms and spin directions, fixed by its ``CA_ROTOR*`` geometry. A
Genesis drone URDF, however, numbers its propellers in an arbitrary order and ships its own
spin directions, so to fly an arbitrary quad with PX4 we must

1. map each PX4 motor channel to the propeller sitting on the matching arm, and
2. make each propeller spin the way PX4 expects for that arm (otherwise yaw runs away).

Both follow purely from the propeller positions, so this module derives them from the URDF::

    layout = gs.px4.quad_x_layout("urdf/drones/cf2x.urdf")
    drone  = scene.add_entity(gs.morphs.Drone(file=..., propellers_spin=layout.propellers_spin))
    px4    = gs.px4.PX4Bridge(scene, drone, imu, gs.px4.PX4Options(
                 ..., motor_mapping=layout.motor_mapping, max_rpm=layout.max_rpm))

Frames: Genesis body is FLU (x forward, y left, z up); PX4 body is FRD (x forward, y right).
"""

import math
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np

import genesis as gs

DEFAULT_PROPELLER_LINKS = ("prop0_link", "prop1_link", "prop2_link", "prop3_link")
GRAVITY = 9.80665

# PX4 X-quad motor channel -> arm corner, as FRD signs (sign_x_forward, sign_y_right). Taken
# from the standard iris / quad_x ``CA_ROTOR*`` geometry: ch0 front-right, ch1 rear-left,
# ch2 front-left, ch3 rear-right.
_PX4_QUAD_X_CORNERS = ((+1, +1), (-1, -1), (+1, -1), (-1, +1))


@dataclass(frozen=True)
class QuadXLayout:
    """PX4-compatible configuration derived from a quad-X URDF.

    Attributes
    ----------
    motor_mapping : tuple[int, ...]
        PX4 motor channel -> Genesis propeller index. Pass to ``PX4Options.motor_mapping``.
    propellers_spin : tuple[int, ...]
        Spin (1 = CCW, -1 = CW) each propeller must have for PX4's yaw convention. Pass to
        ``gs.morphs.Drone.propellers_spin``.
    positions : np.ndarray
        (n, 3) propeller positions in the body FLU frame, in URDF order.
    hover_rpm : float | None
        Per-propeller RPM that balances gravity, from the URDF ``kf`` and total mass. ``None``
        if the URDF lacks the data.
    max_rpm : float | None
        Suggested ``PX4Options.max_rpm`` so PX4's hover throttle maps onto ``hover_rpm``.
        ``None`` if it could not be computed.
    """

    motor_mapping: tuple[int, ...]
    propellers_spin: tuple[int, ...]
    positions: np.ndarray
    hover_rpm: float | None
    max_rpm: float | None


def _resolve_urdf(path: str) -> str:
    """Resolve a URDF path the same way morphs do: cwd first, then the Genesis assets dir."""
    if os.path.isfile(path):
        return path
    candidate = os.path.join(gs.utils.get_assets_dir(), path)
    if os.path.isfile(candidate):
        return candidate
    gs.raise_exception(f"URDF not found in current directory or assets directory: '{path}'.")


def _xyz(node) -> np.ndarray:
    if node is not None and node.get("xyz"):
        return np.array([float(v) for v in node.get("xyz").split()], dtype=np.float64)
    return np.zeros(3, dtype=np.float64)


def _propeller_position(root, link_name: str) -> np.ndarray:
    """Body-frame position of a propeller: its parent fixed-joint origin plus the link origin.

    For the Genesis drone URDFs the joints are identity and the offset lives in the link's
    inertial (else visual) origin, but composing both keeps this correct for other layouts.
    """
    link = next((ln for ln in root.findall("./link") if ln.get("name") == link_name), None)
    if link is None:
        gs.raise_exception(f"Propeller link '{link_name}' not found in URDF.")
    joint = next(
        (
            j
            for j in root.findall("./joint")
            if (j.find("child") is not None and j.find("child").get("link") == link_name)
        ),
        None,
    )
    joint_origin = _xyz(joint.find("origin")) if joint is not None else np.zeros(3)
    link_origin = link.find("./inertial/origin")
    if link_origin is None:
        link_origin = link.find("./visual/origin")
    return joint_origin + _xyz(link_origin)


def read_propeller_positions(urdf_path: str, link_names=DEFAULT_PROPELLER_LINKS) -> np.ndarray:
    """Return the (n, 3) body-FLU propeller positions for ``link_names``, in order."""
    root = ET.parse(_resolve_urdf(urdf_path)).getroot()
    return np.array([_propeller_position(root, name) for name in link_names], dtype=np.float64)


def quad_x_motor_mapping(positions) -> tuple[int, ...]:
    """PX4 motor channel -> propeller index, matched by arm position (X-quad only)."""
    positions = np.asarray(positions, dtype=np.float64)
    if positions.shape[0] != 4:
        gs.raise_exception(f"quad_x_motor_mapping expects 4 propellers, got {positions.shape[0]}.")
    mapping: list[int] = []
    for sign_x, sign_y_right in _PX4_QUAD_X_CORNERS:
        match = next(
            (
                i
                for i, (x, y, _) in enumerate(positions)
                if i not in mapping and np.sign(x) == sign_x and np.sign(-y) == sign_y_right
            ),
            None,
        )
        if match is None:
            gs.raise_exception(
                "Could not match propellers to PX4's X-quad arms. Is this an X configuration "
                "(propellers off both body axes)? Plus-shaped quads need a PX4 quad_+ airframe."
            )
        mapping.append(match)
    return tuple(mapping)


def quad_x_propeller_spin(positions) -> tuple[int, ...]:
    """Required spin per propeller for PX4's yaw convention: ``sign(x * y)`` in body FLU."""
    positions = np.asarray(positions, dtype=np.float64)
    spins = [int(np.sign(x * y)) for x, y, _ in positions]
    if 0 in spins:
        gs.raise_exception("A propeller lies on a body axis; this is not an X-quad layout.")
    return tuple(spins)


def _thrust_scale(root, n_rotors: int, hover_throttle: float, gravity: float):
    """Compute (hover_rpm, suggested max_rpm) from the URDF ``kf`` and total mass, if present."""
    props = root.find("./properties")
    kf = float(props.get("kf")) if (props is not None and props.get("kf")) else None
    mass = sum(float(m.get("value", 0.0)) for m in root.findall(".//inertial/mass"))
    if not kf or mass <= 0.0:
        return None, None
    hover_rpm = math.sqrt(mass * gravity / (n_rotors * kf))
    return hover_rpm, hover_rpm / hover_throttle


def quad_x_layout(
    urdf_path: str,
    link_names=DEFAULT_PROPELLER_LINKS,
    hover_throttle: float = 0.5,
    gravity: float = GRAVITY,
) -> QuadXLayout:
    """Derive a full PX4 quad-X configuration (mapping, spins, thrust scale) from a URDF.

    ``hover_throttle`` is PX4's expected hover throttle (``MPC_THR_HOVER``, default 0.5); the
    suggested ``max_rpm`` is sized so a normalized ``hover_throttle`` command produces hover.
    """
    root = ET.parse(_resolve_urdf(urdf_path)).getroot()
    positions = np.array([_propeller_position(root, name) for name in link_names], dtype=np.float64)
    hover_rpm, max_rpm = _thrust_scale(root, len(link_names), hover_throttle, gravity)
    return QuadXLayout(
        motor_mapping=quad_x_motor_mapping(positions),
        propellers_spin=quad_x_propeller_spin(positions),
        positions=positions,
        hover_rpm=hover_rpm,
        max_rpm=max_rpm,
    )
