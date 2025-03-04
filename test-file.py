from main_structure import Framework
import support_functions
import math
import numpy as np
from datetime import datetime
import constants
from pyquaternion import Quaternion


def euler_from_quaternion(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)
    
    return roll_x, pitch_y, yaw_z # in radians
iss_posn = np.array([7e6, math.radians(0.05), math.radians(50), math.radians(311.6218), math.radians(199.2431), math.radians(48.4420)])
xyz = support_functions.kep_to_cart(iss_posn)[:3]
magnetometer = support_functions.igdf_eci_vector(xyz[0], xyz[1], xyz[2], datetime.now())
framework = Framework(iss_posn, magnetometer, True)
random_4 = []
random_3 = [0.2, 0.3, 0.6]
if(np.linalg.norm(random_3) < 1):
    random_4 = np.array([random_3[0], random_3[1], random_3[2], math.sqrt(1 - np.linalg.norm(random_3)**2)])

true_rot = Quaternion(random_4)
true_posn = iss_posn

for i in range(int(90*60/constants.DT)):#1 revolution
    true_posn = support_functions.propagate_orbit(true_posn, constants.DT)
    time = datetime.now()
    xyz = support_functions.kep_to_cart(true_posn)[:3]
    mag = support_functions.igdf_eci_vector(xyz[0], xyz[1], xyz[2], time)
    magnetometer = true_rot.rotate(mag)
    #sun = true_rot.rotate(support_functions.eci_sun_vector(time))
    framework.propagate(np.concatenate([magnetometer], axis = 0), time)
    distance = np.linalg.norm(support_functions.kep_to_cart(true_posn)[:3] - support_functions.kep_to_cart(framework.get_position_eci())[:3])
    w = Quaternion(axis = [1, 0, 0], degrees = 0.05)
    print("KM OFFSET: %f", distance/1000)
    print("REAL: ", true_rot)
    print("PRED: ", framework.quaternion_estimation.estimate)
    print("W: ", framework.quaternion_estimation.gyro_bias)
    print("BDOT: ", framework.bdot_estimation.state_estimate[3:6])
    
    print("DIFF: ", support_functions.quat_diff(framework.quaternion_estimation.estimate, true_rot))
    true_rot *= w
    print("\n")

print("KM OFFSET: %f", distance/1000)
print("\n")


