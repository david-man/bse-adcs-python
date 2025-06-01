import numpy as np
import pyquaternion
from pyquaternion import Quaternion
import support_functions
from numpy.linalg import cholesky
from scipy.linalg import sqrtm
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
import math
from numpy.typing import NDArray
from typing import List, Tuple
from copy import deepcopy

#ALL CREDITS TO https://kodlab.seas.upenn.edu/uploads/Arun/UKFpaper.pdf FOR THE EQUATIONS
'''
Overall Idea of a UKF: 
- Like the MEKF, rather than using a 7-element state [quaternion, gyro bias], we use a 6-element state [quaternion error vector, gyro bias] and assume they are both small
- Given a timestep, a measurement, and our last state, we take some informed samples around our estimated state to guess where we are(what our current state is)
'''

global_Q = np.eye(6)*0.1
global_R = np.eye(4)*0.001
dt = 1
def quaternion_to_rotation(x : Quaternion):
    '''
    Function that converts a quaternion into a rotation vector
    '''
    quat = x.elements
    if quat[0] < 0:
        quat = -quat
    if(quat[0] >= 1):
        return np.zeros(3)
    theta = 2 * math.acos(quat[0])
    if(theta == 0):
        return np.zeros(3)
    vec = theta * quat[1:] / np.sqrt(1-quat[0]**2)
    return vec
def rotation_to_quat(vec):
    '''
    Function that converts a rotation vector into a quaternion
    '''
    if(np.linalg.norm(vec) == 0):
        return Quaternion()
    axis = vec / np.linalg.norm(vec)
    angle = np.linalg.norm(vec)
    return Quaternion(angle = angle, axis = axis)
def get_error_vectors(Y: List[Quaternion], x : Quaternion):
    '''
    Function that gets the error vectors between all quaternions in Y and the quaternion X
    '''
    error_vectors = []
    for quat in Y:
        error_quaternion = quat * x.inverse
        error_vectors.append(quaternion_to_rotation(error_quaternion))
    return error_vectors
def gradient_descent(Y : List[Quaternion], x : Quaternion):
    '''
    Function that performs gradient descent to find the average quaternion, as quaternions are not additive
    '''
    for i in range(100):
        error_vectors = get_error_vectors(Y, x)
        ave = np.mean(error_vectors, axis=0)   # formula 54
        if(np.linalg.norm(ave) < 0.001):
            break
        average_error_quat = rotation_to_quat(ave)
        x = x * average_error_quat   # formula 55
        
    return x, error_vectors


def get_weights(lam, x, alpha, beta):
    '''
    Derived from Van Der Merwe Scaled(Weighted) Sigma Points
    '''
    n = x.shape[0]
    c = .5 / (n + lam)
    covariance_weights = np.full(2*n + 1, c)
    mean_weights = np.full(2*n + 1, c)
    covariance_weights[0] = lam / (n + lam) + (1 - alpha**2 + beta)
    mean_weights[0] = lam/ (n + lam)
    return covariance_weights, mean_weights
def get_sigma_points(lam, x, P):
    '''
    Derived from Van Der Merwe Scaled(Weighted) Sigma Points
    '''
    n = x.shape[0]
    U =cholesky((lam + n)*P)
    sigmas = [x]
    for k in range(1, n + 1, 1):#from 1 -> n
        idx = k - 1
        sigmas.append(np.add(x, U[idx]))
    for k in range(n + 1, 2*n + 1, 1):#from n + 1 -> 2n + 1
        idx = k - (n + 1)
        sigmas.append(np.subtract(x, U[idx]))
    return np.array(sigmas)
def calculate_lambda(alpha, x):
    '''
    Derived from Van Der Merwe Scaled(Weighted) Sigma Points
    '''
    n = x.shape[0]
    kappa = 3-n
    return alpha**2 * (n + kappa) - n

def error_sigmas_to_quat_sigmas(error_sigmas, rotation : Quaternion):
    '''
    Converts the 6-element sigma point to a 7-element sigma point through use of the current rotation and attitude error vector
    '''
    quat_sigmas = []
    for sigma in error_sigmas:
        quat_rot = rotation_to_quat(sigma[:3])
        new_quat_rot = rotation * quat_rot
        quat_sigmas.append(np.concatenate([new_quat_rot.elements, sigma[3:]]))
    return np.array(quat_sigmas)

def propagate_quat_sigmas(quat_sigmas):
    '''
    Propagates the 7-element sigma points forward a timestep so that we can guess
    '''
    new_sigmas = []
    for sigma in quat_sigmas:
        quaternion = sigma[:4]
        w = sigma[4:7]
        new_quaternion = Quaternion(quaternion)*rotation_to_quat(w * dt)
        new_sigmas.append(np.concatenate([new_quaternion.elements, w]))
    return np.array(new_sigmas)

def get_measurements(quat_sigmas):
    '''
    Gets the measurements expected of the 7-element sigma points. Currently just uses the first 4 elements, as we feed a raw quaternion as the measurement
    '''
    measurements = []
    for sigma in quat_sigmas:
        quat = Quaternion(sigma[:4])
        measurements.append(quat.elements)
    return np.array(measurements)
def iterate(error_state, rotation : Quaternion, P, obs):
    '''
    Iterates forward given our last 6-element state & covariance, our last rotation, and a measurement
    '''
    global global_Q, global_R
    n = error_state.shape[0]
    alpha = 1e-3
    beta = 2
    lam = calculate_lambda(alpha, error_state)
    sigmas = get_sigma_points(lam, error_state, P + global_Q)
    quat_sigmas = error_sigmas_to_quat_sigmas(sigmas, rotation)
    propagated_quat_sigmas = propagate_quat_sigmas(quat_sigmas)
    measurements = get_measurements(propagated_quat_sigmas)
    covariance_weights, mean_weights = get_weights(lam, error_state, alpha, beta)

    quaternions_of_propagated_sigmas = []
    for y in propagated_quat_sigmas:
        quaternions_of_propagated_sigmas.append(Quaternion(y[:4]))
    average_quaternion, propagated_error_vectors = gradient_descent(quaternions_of_propagated_sigmas, rotation)
    propagated_errors = np.hstack([propagated_error_vectors, propagated_quat_sigmas[:, 4:]])

    
    mean_error = np.zeros(n)
    for column in range(n):#for each state element
        var_mean = 0
        for row in range(2*n + 1):#each row is the corresponding element for 1 sigma point
            var_mean += propagated_errors[row][column] * mean_weights[row]
        mean_error[column] = var_mean
    mean_measurement = np.zeros(measurements.shape[1])
    for column in range(measurements.shape[1]):#for each measurement
        var_mean = 0
        for row in range(2*n + 1):#each row is the corresponding measurement for 1 sigma point
            var_mean += measurements[row][column] * mean_weights[row]
        mean_measurement[column] = var_mean

    P_hat = np.zeros((n, n))
    for i, error in enumerate(propagated_errors):
        err = error - mean_error
        P_hat += covariance_weights[i] * np.outer(err, err)
    
    P_xz = np.zeros((n, measurements.shape[1]))
    for i, error in enumerate(propagated_errors):
        err = error - mean_error
        msmt_err = measurements[i] - mean_measurement
        P_xz += covariance_weights[i] * np.outer(err,msmt_err)
    
    P_zz = np.zeros((measurements.shape[1], measurements.shape[1]))
    for i, msmt in enumerate(measurements):
        msmt_err = msmt - mean_measurement
        P_zz += covariance_weights[i] * np.outer(msmt_err, msmt_err)

    k = P_xz @ np.linalg.pinv(P_zz)
    x_hat = mean_error + k@(obs - mean_measurement)
    P = P_hat - k@global_R@k.T

    x_hat_rot = rotation_to_quat(x_hat[:3]) * rotation_to_quat(x_hat[3:])
    return x_hat, rotation * x_hat_rot, P


if __name__ == '__main__':
    true_rot = Quaternion([1,0.2,0.2,0.2]).normalised
    rot = Quaternion()
    P = np.eye(6)*0.1
    state = np.zeros(6)
    for i in range(20):
        measurement = Quaternion(true_rot.elements + np.random.normal(0, scale = 0.0005, size = 4)).normalised.elements
        state, rot, P = iterate(state, rot, P, measurement)
        state[:3] = np.zeros(3)#we need to reset our error vector here, as we already tacked on the error vector to the rotation at the end of the last state! 
    print(rot)
    print(true_rot)
    print(state)