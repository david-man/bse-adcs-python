import numpy as np
from numpy.linalg import cholesky

'''
This file demonstrates a classic UKF. It is here more for teaching purposes than anything else. Refer to the MUKF demo for explanations of things
'''
dt = 1
global_state = np.array([1,1])#x, y
global_P = np.eye(2)
global_Q = np.eye(2)
global_R = np.eye(2)
global_sigmas = []
def f(state, dt):
    return state
def h(state):
    return state

def get_weights(lam, x, alpha, beta):
    n = x.shape[0]
    c = .5 / (n + lam)
    covariance_weights = np.full(2*n + 1, c)
    mean_weights = np.full(2*n + 1, c)
    covariance_weights[0] = lam / (n + lam) + (1 - alpha**2 + beta)
    mean_weights[0] = lam/ (n + lam)
    return covariance_weights, mean_weights
def get_sigma_points(lam, x, P):
    n = x.shape[0]
    U = cholesky((lam + n)*P)
    sigmas = [x]
    for k in range(1, n + 1, 1):#from 1 -> n
        idx = k - 1
        sigmas.append(np.add(x, U[idx]))
    for k in range(n + 1, 2*n + 1, 1):#from n + 1 -> 2n + 1
        idx = k - (n + 1)
        sigmas.append(np.subtract(x, U[idx]))
    return np.array(sigmas)
def calculate_lambda(alpha, x):
    n = x.shape[0]
    kappa = 3 - n
    return alpha**2 * (n + kappa) - n
def process_sigmas(sigma_points):
    resultant_sigmas = []
    for point in sigma_points:
        resultant_sigmas.append(f(point, dt))
    return np.array(resultant_sigmas)
def predict():
    global global_state, global_P, global_Q, global_R, global_sigmas
    n = global_state.shape[0]
    alpha = 1e-3
    beta = 2
    lam = calculate_lambda(alpha, global_state)
    sigmas = get_sigma_points(lam, global_state, global_P)
    covariance_weights, mean_weights = get_weights(lam, global_state, alpha, beta)
    resultant_sigmas = process_sigmas(sigmas)

    next_x_mean = []#mean of sigma points
    for column in range(resultant_sigmas.shape[1]):#each column is 1 variable
        var_mean = 0
        for row in range(resultant_sigmas.shape[0]):#each row is 1 sigma point
            var_mean += resultant_sigmas[row][column] * mean_weights[row]
        next_x_mean.append(var_mean)
    next_x_mean = np.array(next_x_mean)

    next_P = np.zeros((n, n))
    for i, resultant_point in enumerate(resultant_sigmas):
        diff = resultant_point - next_x_mean
        diff = np.reshape(diff, (n, 1))
        next_P += diff@diff.T * covariance_weights[i]
    next_P += dt * global_Q
    
    global_state = next_x_mean
    global_P = next_P
    global_sigmas = resultant_sigmas

def update(obs):
    global global_state, global_P, global_Q, global_R, global_sigmas
    n = global_state.shape[0]
    alpha = 1e-3
    beta = 2
    lam = calculate_lambda(alpha, global_state)
    covariance_weights, mean_weights = get_weights(lam, global_state, alpha, beta)

    sigma_measurements = []
    for sigma_point in global_sigmas:
        sigma_measurements.append(h(sigma_point))
    sigma_measurements = np.array(sigma_measurements)

    measurement_means  =[]
    for column in range(sigma_measurements.shape[1]):#for each measurement
        var_mean = 0
        for row in range(sigma_measurements.shape[0]):#each row is the corresponding measurement for 1 sigma point
            var_mean += sigma_measurements[row][column] * mean_weights[row]
        measurement_means.append(var_mean)
    measurement_means = np.array(measurement_means)

    P_yy = np.zeros((sigma_measurements.shape[1], sigma_measurements.shape[1]))
    for i, resultant_point in enumerate(sigma_measurements):
        diff = resultant_point - measurement_means
        diff = np.reshape(diff, (sigma_measurements.shape[1], 1))
        P_yy += diff@diff.T * covariance_weights[i]
    P_yy += global_R

    P_xy = np.zeros((n, sigma_measurements.shape[1]))
    for i in range(len(sigma_measurements)):
        sigma_point = global_sigmas[i]
        sigma_measurement = sigma_measurements[i]
        sigma_diff = (sigma_point - global_state).reshape((n, 1))
        measurement_diff = (sigma_measurement - measurement_means).reshape((1, sigma_measurements.shape[1]))
        P_xy += sigma_diff@measurement_diff * covariance_weights[i]
    
    k = P_xy@np.linalg.inv(P_yy)

    global_state += k@(obs - measurement_means)
    global_P -= k@P_yy@k.T
if __name__ == '__main__':
    
    P = np.eye(3)
    start = np.array([2.0, 0])
    for i in range(5):
        predict()
        update(start)
    print(global_state)