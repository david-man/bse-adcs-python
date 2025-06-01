from casadi import *
from pyquaternion import Quaternion
import spatial_casadi as sc
from scipy.spatial.transform import Rotation

'''
This file demonstrates the ability for Quadratic Programming to, given a decently close initial guess as to the true rotation, accurately
estimate the true rotation with JUST ONE VECTOR. 
'''

def get_inverse_quat(q):
    return sc.Rotation.from_quat(vertcat(-q[:3], q[3]))

def quat_rotate(q, rotation_vector):
    k = vertcat(rotation_vector, 0)
    rot_quat = sc.Rotation.from_quat(k)
    quat = sc.Rotation.from_quat(q)
    inverted_quaternion = get_inverse_quat(quat.as_quat())
    return (quat * rot_quat * inverted_quaternion).as_quat()[:3]



true_rot = Quaternion([1, 2, 3, 4] + np.random.normal(loc = 0, scale = 0.05, size = (4,))).normalised
vec_1 = np.array([1,2,3])
rotated_vec_1 = true_rot.rotate(vec_1)

opti = Opti()
quat = opti.variable(4, 1)
initial_quat = np.array([2,3,4,1])/np.linalg.norm([1,2,3,4])
opti.set_initial(quat, initial_quat)
opti.subject_to(norm_2(quat) == 1)
opti.subject_to(norm_2(quat_rotate(quat, vec_1) - rotated_vec_1)**2 == 0)
opti.minimize(norm_2(quat - initial_quat + 0.1))

opti.solver('ipopt', {"print_time": False, "verbose": False}, {"print_level":1, "max_iter": 25000})
try:
    result = opti.solve()
    q = result.value(quat)
    print(Quaternion(scalar = q[3], vector = q[:3]).normalised, true_rot)
    print(quat_rotate(q, vec_1) - rotated_vec_1)
except RuntimeError as e:
    q = opti.debug.value(quat)
    print(e)
    print(Quaternion(scalar = q[3], vector = q[:3]).normalised, true_rot)