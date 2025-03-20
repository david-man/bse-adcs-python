from main_structure import Framework
import support_functions
import math
import numpy as np
from datetime import datetime
import constants
from pyquaternion import Quaternion
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints

def f(state, dt):
    g = support_functions.get_gravity_accel(state[:3])
    return state + dt * np.concatenate([state[3:] + 1/2 * g * dt, g])
def h(state):
    n = np.linalg.norm(support_functions.igdf_eci_vector(state[0], state[1], state[2], datetime.now()))
    return np.array([n])


dt = 10
points = MerweScaledSigmaPoints(6, alpha=10, beta=2., kappa=-3)
kf = UnscentedKalmanFilter(dim_x=6, dim_z=1, dt=dt, fx=f, hx=h, points=points)

iss_posn = np.array([7e6, math.radians(0.05), math.radians(50), math.radians(311.6218), math.radians(199.2431), math.radians(48.4420)])
xyz = support_functions.kep_to_cart(iss_posn)[:3]
kf.x = support_functions.kep_to_cart(iss_posn)
kf.P = np.eye(6) * 100000
kf.Q = np.eye(6) * 1
kf.R = 1000000

true_posn = iss_posn
for i in range(int(90 * 60 / dt)):
    xyz = support_functions.kep_to_cart(true_posn)[:3]
    if(i % (int(90 * 60 / dt)/6) == 0):
        kf.x = support_functions.kep_to_cart(true_posn)
    print(np.linalg.norm(xyz - kf.x[:3])/1000)
    mag = support_functions.igdf_eci_vector(xyz[0], xyz[1], xyz[2], datetime.now()) + np.random.normal(0, 500, 3)
    kf.predict()
    kf.update(np.linalg.norm(mag))
    print("\n")
    
    true_posn = support_functions.propagate_orbit(true_posn, dt)


