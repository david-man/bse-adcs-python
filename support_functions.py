import pyorb.kepler as kepler
import pyorb.orbit as orbit
import ppigrf
from GravityModels.CelestialBodies.Planets import Earth
from GravityModels.Models import SphericalHarmonics
import pymap3d
from datetime import datetime
from astropy.time import Time
import math
import numpy as np
from numpy.typing import NDArray
from pyquaternion import Quaternion

#constants
EARTH_SIM = SphericalHarmonics(Earth().sh_file, 4)
AIR_DENSITY = 10e-15
MU_EARTH = 3.986e14
MASS_EARTH = 5.972e24
BODY_LENGTH = 3
BODY_MASS = 1
INV_BALLISTIC_COEFF = 70

def get_drag_accel(velocity_vector : NDArray[np.float64]):
    '''Gets the drag acceleration vector'''
    assert(len(velocity_vector) == 3)
    velocity = np.array([velocity_vector[0], velocity_vector[1], velocity_vector[2]])
    drag_direction = -velocity/np.linalg.norm(velocity)
    drag_coefficient = AIR_DENSITY * BODY_LENGTH * INV_BALLISTIC_COEFF
    drag_force = 1/2 * AIR_DENSITY * (np.linalg.norm(velocity)**2) * drag_coefficient
    drag_acceleration = drag_force/BODY_MASS
    return drag_direction * drag_acceleration

def get_gravity_accel(position_vector : NDArray[np.float64]):
    assert(len(position_vector) == 3)
    '''Gets the gravity acceleration vector'''
    return EARTH_SIM.compute_acceleration(np.array([position_vector]))[0]

def propagate_orbit(kepler_start : NDArray[np.float64], duration : int):
    '''Propagates an orbit around Earth for [duration] seconds'''
    assert(len(kepler_start) == 6)
    orb = orbit.Orbit(M0 = MASS_EARTH, kepler = kepler_start.reshape((6, 1)))
    orb.propagate(duration)
    return orb.kepler[:, 0]

def kep_to_cart(kepler_arr : NDArray[np.float64]):
    '''Kepler 6-element coordinates to Cartesian 6-element coordinates'''
    assert(len(kepler_arr) == 6)
    return kepler.kep_to_cart(kep = kepler_arr, mu = MU_EARTH)

def cart_to_kep(cart_arr : NDArray[np.float64]):
    
    '''Cartesian 6-element coordinates to Keplerian 6-element coordinates'''
    assert(len(cart_arr) == 6)
    return kepler.cart_to_kep(cart = cart_arr, mu = MU_EARTH)

def igdf_eci_vector(x : float, y : float, z : float, time : datetime.time):
    '''Gets the IGDF Magnetic Field vector for an x, y, z in the ECI frame'''
    lat, lon, alt = pymap3d.eci2geodetic(x, y, z, time)
    b_east, b_north, b_up = ppigrf.igrf(lon, lat, alt/1000, time)
    ecefb_x, ecefb_y, ecefb_z = pymap3d.enu2ecef(b_east, b_north, b_up, lat, lon, alt)
    b_x, b_y, b_z = pymap3d.ecef2eci(ecefb_x, ecefb_y, ecefb_z, time)
    b_vec = np.array([b_x - x, b_y - y, b_z - z])
    return b_vec.reshape((3,))

def _right_ascension_to_longitude(right_ascension : np.float64, time : datetime.time):
    #takes in degrees, returns values in degrees
    current_time = Time(time)
    current_sidereal = current_time.sidereal_time('apparent', 'greenwich')
    return (math.degrees(right_ascension) - current_sidereal.degree) % 360

def _raw_sun_vector(time : datetime.time):
    #returns values in DEGREES
    date_relative_to_J2000 = (time - datetime(2000, 1, 1, 12, 0, 0)).days
    mean_longitude = (280.461 + 0.9856474 * date_relative_to_J2000) % 360
    mean_anomaly = (357.528 + 0.9856003 * date_relative_to_J2000) % 360

    ecliptic_longitude = mean_longitude + 1.915 * math.sin(math.radians(mean_anomaly)) + 0.020 * math.sin(2* math.radians(mean_anomaly))

    obliquity = 23.439 - 0.0000004 * date_relative_to_J2000
    Y = math.cos(math.radians(obliquity)) * math.sin(math.radians(ecliptic_longitude))
    X = math.cos(math.radians(ecliptic_longitude))

    a = math.atan(Y/X)

    right_ascension = 0
    if(X<0): 
        right_ascension = (a + math.pi)
    elif(X > 0 and Y < 0):
        right_ascension = (a + 2*math.pi)
    else:
        right_ascension = a

    declination = math.asin(math.sin(math.radians(obliquity))*math.sin(math.radians(ecliptic_longitude)))
    return np.array([math.degrees(right_ascension), math.degrees(declination)])

def eci_sun_vector(time : datetime.time):
    '''Gets the sun unit vector in the ECI frame for any given time'''
    raw_sun_ra, sun_lat = _raw_sun_vector(time)
    sun_lon = _right_ascension_to_longitude(raw_sun_ra, time)

    ecef_sunx, ecef_suny, ecef_sunz = pymap3d.geodetic2ecef(sun_lat, sun_lon, 0)#vector that touches the surface
    sun_x, sun_y, sun_z = pymap3d.ecef2eci(ecef_sunx, ecef_suny, ecef_sunz, time)

    sun_vec = np.array([sun_x, sun_y, sun_z])
    sun_unit_vec = sun_vec/ (np.linalg.norm(sun_vec))
    return sun_unit_vec

def quat_diff(q1, q2):
    '''Calculates the Euler angle differences between two quaternions'''
    assert(isinstance(q1, Quaternion))
    assert(isinstance(q2, Quaternion))
    qd = q1.conjugate * q2

    # Calculate Euler angles from this difference quaternion
    phi   = math.atan2( 2 * (qd.w * qd.x + qd.y * qd.z), 1 - 2 * (qd.x**2 + qd.y**2) )
    theta = math.asin ( 2 * (qd.w * qd.y - qd.z * qd.x) )
    psi   = math.atan2( 2 * (qd.w * qd.z + qd.x * qd.y), 1 - 2 * (qd.y**2 + qd.z**2) )

    return math.degrees(phi), math.degrees(theta), math.degrees(psi)

def skew_symmetric(w):
    '''Calculates the skew symmetric matrix for w'''
    return -np.array([
        [0, w[2], -w[1]],
        [-w[2], 0, w[0]],
        [w[1], -w[0], 0]
    ])