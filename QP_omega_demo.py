from casadi import *
from pyquaternion import Quaternion
import spatial_casadi as sc
from scipy.spatial.transform import Rotation

'''
WARNING: THIS CODE IS NOT THAT STABLE, ESPECIALLY WITH OMEGA. 

This file demonstrates the ability for Quadratic Programming to decently estimate the omega and the true rotation given a measurement at t = now and t = now - 1.
Due to the problem of multiple solutions, it assumes a decently close initial guess, which makes the 1-vector QP option more appealing
'''

def get_inverse_quat(q):
    return sc.Rotation.from_quat(vertcat(-q[:3], q[3]))

def quat_rotate(q, rotation_vector):
    k = vertcat(rotation_vector, 0)
    rot_quat = sc.Rotation.from_quat(k)
    quat = sc.Rotation.from_quat(q)
    inverted_quaternion = get_inverse_quat(quat.as_quat())
    return (quat * rot_quat * inverted_quaternion).as_quat()[:3]

last_true_rot = Quaternion([1, 0, 0, 0.1]).normalised
true_rot = Quaternion([1, 0.05, 0.05, 0.05]).normalised
last_vec_1 = np.array([1.01, -0.01, 0.01])
vec_1 = np.array([1, 0, 0])
vec_derivative = vec_1 - last_vec_1

rotated_vec_1 = true_rot.rotate(vec_1)
last_rotated_vec_1 = last_true_rot.rotate(last_vec_1)
measured_derivative = rotated_vec_1 - last_rotated_vec_1

opti = Opti()
quat = opti.variable(4, 1)
w = opti.variable(3, 1)
opti.subject_to(norm_2(quat) == 1)
opti.subject_to(quat[3] > 0.1)
opti.subject_to(norm_2(w) < 0.5)
opti.set_initial(quat, [0,0,0,1])
opti.set_initial(w, [0.00001, 0, 0])

opti.minimize(norm_2(quat_rotate(quat, vec_1) - rotated_vec_1)**2 + \
              norm_2(vec_derivative - (measured_derivative + cross(w, vec_1)))**2)#comes from the derivative of a rotating vector, found at https://orbital-mechanics.space/review/time-derivatives-of-moving-vectors.html
opti.solver('ipopt', {"print_time": False, "verbose": False}, {"print_level":1, "max_iter": 25000})
derivative = Rotation.from_quat((last_true_rot * true_rot.inverse).elements, scalar_first = True)

try:
    result = opti.solve()
    q = result.value(quat)
    print("FEASIBLE")
    print(Quaternion(scalar = q[3], vector = q[:3]).normalised, result.value(w))
    print(true_rot, derivative.as_rotvec())
    print(derivative.as_rotvec() - result.value(w))
except RuntimeError as e:
    print(e)
    q = opti.debug.value(quat)
    print("INFEASIBLE")
    print(Quaternion(scalar = q[3], vector = q[:3]).normalised, opti.debug.value(w))
    print(true_rot, derivative.as_rotvec())
    print(norm_2(quat_rotate(q, vec_1) - rotated_vec_1))
    print(vec_derivative - (measured_derivative + cross(opti.debug.value(w), vec_1)))
