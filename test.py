from pyquaternion import Quaternion
from typing import List
import numpy as np
import math

def quaternion_to_rotation(x : Quaternion):
    quat = x.elements
    if quat[0] < 0:
        quat = -quat
    theta = 2 * math.acos(quat[0])
    if theta == 0:
        return np.array([0,0,0])
    vec = theta * quat[1:] / np.sqrt(1-quat[0]**2)
    return vec
def rotation_to_quat(vec):
    if(np.linalg.norm(vec) == 0):
        return Quaternion()
    axis = vec / np.linalg.norm(vec)
    angle = np.linalg.norm(vec)
    return Quaternion(angle = angle, axis = axis)
def get_error_vectors(Y: List[Quaternion], x : Quaternion):
    error_vectors = []
    for quat in Y:
        error_quaternion = quat * x.inverse
        error_vectors.append(quaternion_to_rotation(error_quaternion))
    return error_vectors
def gradient_descent(Y : List[Quaternion], x : Quaternion):
    for i in range(500):
        error_vectors = get_error_vectors(Y, x)
        ave = np.mean(error_vectors, axis=0)   # formula 54
        average_error_quat = rotation_to_quat(ave)
        x = average_error_quat * x       # formula 55
    return x

if __name__ == '__main__':
    Y = [Quaternion([1, 0.1, 0.01, 0.01]).normalised, Quaternion([1, -0.1, -0.01, 0]).normalised, Quaternion()]
    x = Quaternion([1,0,0,0]).normalised
    print(gradient_descent(Y, x))