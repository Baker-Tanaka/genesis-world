"""Pure-numpy coordinate, frame and atmosphere conversions for the PX4 bridge.

None of these functions depend on torch, a GPU or a running PX4 instance, so they can be
unit-tested in isolation. Everything is vectorized over a leading environment batch axis
``N`` so the bridge can convert all parallel envs in one shot.

Frame conventions
-----------------
- Genesis world  : ENU  (x = East, y = North, z = Up)            [configurable]
- Genesis body   : FLU  (x = Forward, y = Left, z = Up)
- PX4 world      : NED  (x = North, y = East, z = Down)
- PX4 body       : FRD  (x = Forward, y = Right, z = Down)
"""

import numpy as np

# ----------------------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------------------

EARTH_RADIUS = 6378137.0  # WGS-84 equatorial radius [m]

# International Standard Atmosphere (troposphere, < 11 km)
_ISA_P0 = 101325.0  # sea-level static pressure [Pa]
_ISA_T0 = 288.15  # sea-level standard temperature [K]
_ISA_L = 0.0065  # temperature lapse rate [K/m]
_ISA_G = 9.80665  # gravitational acceleration [m/s^2]
_ISA_M = 0.0289644  # molar mass of dry air [kg/mol]
_ISA_R = 8.31447  # universal gas constant [J/(mol*K)]
_ISA_EXP = (_ISA_G * _ISA_M) / (_ISA_R * _ISA_L)


# ----------------------------------------------------------------------------------------
# Vector frame conversions  (input/output shape: (..., 3))
# ----------------------------------------------------------------------------------------


def enu_to_ned(v):
    """ENU vector -> NED vector:  N = y_E, E = x_E, D = -z_U."""
    v = np.asarray(v, dtype=np.float64)
    out = np.empty_like(v)
    out[..., 0] = v[..., 1]
    out[..., 1] = v[..., 0]
    out[..., 2] = -v[..., 2]
    return out


# ENU<->NED is an involution, so the inverse is the same swap.
ned_to_enu = enu_to_ned


def flu_to_frd(v):
    """FLU body vector -> FRD body vector:  x stays, y and z are negated."""
    v = np.asarray(v, dtype=np.float64)
    out = np.empty_like(v)
    out[..., 0] = v[..., 0]
    out[..., 1] = -v[..., 1]
    out[..., 2] = -v[..., 2]
    return out


frd_to_flu = flu_to_frd


# ----------------------------------------------------------------------------------------
# Quaternion conversions  (w-x-y-z layout, matching genesis.utils.geom; shape: (..., 4))
# ----------------------------------------------------------------------------------------


def _quat_mul(a, b):
    """Hamilton product of two w-x-y-z quaternion arrays, broadcast over the batch."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    out = np.empty(np.broadcast_shapes(a.shape, b.shape), dtype=np.float64)
    out[..., 0] = aw * bw - ax * bx - ay * by - az * bz
    out[..., 1] = aw * bx + ax * bw + ay * bz - az * by
    out[..., 2] = aw * by - ax * bz + ay * bw + az * bx
    out[..., 3] = aw * bz + ax * by - ay * bx + az * bw
    return out


# Rotation that maps ENU -> NED (equivalently FLU -> FRD); it is its own inverse.
# It corresponds to a 180 deg rotation about the (1, 1, 0)/sqrt(2) axis.
_Q_ENU_NED = np.array([0.0, np.sqrt(0.5), np.sqrt(0.5), 0.0])


def enu_flu_quat_to_ned_frd(quat):
    """Convert an attitude quaternion expressed as ENU->FLU into NED->FRD.

    q_ned_frd = q_enu_ned * q_enu_flu * q_enu_ned^-1   (q_enu_ned is an involution).
    """
    quat = np.asarray(quat, dtype=np.float64)
    q = np.broadcast_to(_Q_ENU_NED, quat.shape)
    return _quat_mul(_quat_mul(q, quat), q)


# ----------------------------------------------------------------------------------------
# Geodety
# ----------------------------------------------------------------------------------------


def local_ned_to_latlon(home, north, east, down):
    """Flat-earth projection of a local NED offset (metres) onto WGS-84 lat/lon/alt.

    Parameters
    ----------
    home : tuple(float, float, float)
        Reference (lat_deg, lon_deg, alt_m_AMSL).
    north, east, down : array_like
        Local NED offsets from ``home`` in metres (any common broadcastable shape).

    Returns
    -------
    lat_deg, lon_deg, alt_m : np.ndarray
        Geodetic coordinates; ``alt_m`` is AMSL (``home_alt - down``).
    """
    lat0, lon0, alt0 = home
    north = np.asarray(north, dtype=np.float64)
    east = np.asarray(east, dtype=np.float64)
    down = np.asarray(down, dtype=np.float64)

    lat0_rad = np.radians(lat0)
    dlat = north / EARTH_RADIUS
    dlon = east / (EARTH_RADIUS * np.cos(lat0_rad))

    lat = lat0 + np.degrees(dlat)
    lon = lon0 + np.degrees(dlon)
    alt = alt0 - down
    return lat, lon, alt


# ----------------------------------------------------------------------------------------
# Atmosphere (barometer)
# ----------------------------------------------------------------------------------------


def altitude_to_pressure(alt_m):
    """ISA static pressure [hPa] at geopotential altitude ``alt_m`` (AMSL)."""
    alt_m = np.asarray(alt_m, dtype=np.float64)
    pressure_pa = _ISA_P0 * (1.0 - _ISA_L * alt_m / _ISA_T0) ** _ISA_EXP
    return pressure_pa / 100.0  # Pa -> hPa


def pressure_to_altitude(pressure_hpa):
    """Inverse of :func:`altitude_to_pressure`: pressure [hPa] -> altitude [m]."""
    pressure_pa = np.asarray(pressure_hpa, dtype=np.float64) * 100.0
    return (_ISA_T0 / _ISA_L) * (1.0 - (pressure_pa / _ISA_P0) ** (1.0 / _ISA_EXP))


def temperature_at(alt_m):
    """ISA temperature [degC] at altitude ``alt_m`` (AMSL)."""
    alt_m = np.asarray(alt_m, dtype=np.float64)
    return (_ISA_T0 - _ISA_L * alt_m) - 273.15
