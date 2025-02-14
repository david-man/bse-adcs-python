from datetime import datetime
import numpy as np
from typing import Callable
from numpy.typing import NDArray
from pyquaternion import Quaternion

class KalmanFilter():
    '''Class detailing an abstract Kalman Filter'''
    def __init__(self, observation_noise_matrix : NDArray[np.float64], 
                 process_noise_matrix : NDArray[np.float64], 
                 starting_state : NDArray[np.float64], 
                 starting_covariance : NDArray[np.float64], 
                 simulation_dt : int):
        self.state_estimate = starting_state
        self.covariance = starting_covariance
        self.Q = process_noise_matrix
        self.R = observation_noise_matrix
        self.dt = simulation_dt
        self.state_size = len(starting_state)
        self.measurement_size = len(process_noise_matrix)
        self.time = datetime.now()
        
        assert(self.Q.shape[0] == len(starting_state))

        assert(self.Q.shape[0] == self.Q.shape[1])#assert square matrices
        assert(self.covariance.shape[0] == self.covariance.shape[1])
        assert(self.R.shape[0] == self.R.shape[1]) 

    def f(self, state) -> NDArray[np.float64]:
        '''Proceeds to the next state by dt time'''
        return state
    def h(self, state) -> NDArray[np.float64]:
        '''Gets the expected measurements at a state'''
        return state
    def predict(self) -> None:
        '''Make initial predictions for state & covariance matrices'''
        pass
    def update(self, measurements) -> None:
        '''Updates the predictions in the KF based on kalman gain terms'''
        pass
    def iterate(self, measurements) -> None:
        '''Performs a predict+update iteration'''
        self.predict()
        self.update(measurements)
    def set_time(self, time : datetime.time):
        '''Sets a standard time for calculations'''
        self.time = time

    def estimate_jacobian(self, f : Callable[NDArray[np.float64], NDArray[np.float64]],
                     state : NDArray[np.float64], #estimate jacobian at this state
                      epsilon = 0.000001 #limit definition of the derivative applied to vectorvalued partials
                     ):
        '''Estimates the Jacobian of a given function'''
        y = f(state)
        n = state.shape[0]#function input vector shape
        m = y.shape[0]#function output vector shape
        J = np.zeros((m, n))
        for i in range(n):
            state_1 = np.copy(state)
            state_1[i] = state[i] + epsilon
            f_1 = f(state_1)
            for j in range(m):
                J[j][i] = (f_1[j] - y[j])/epsilon

        return J

class EKF(KalmanFilter):
    '''Class detailing a standard EKF'''
    def predict(self):
        F = self.estimate_jacobian(self.f, self.state_estimate)
        self.state_estimate = self.f(self.state_estimate)
        self.covariance = F@self.covariance@F.T
    def update(self, measurements):
        assert(len(measurements) == self.measurement_size)

        residual = measurements - self.h(self.state_estimate)
        H = self.estimate_jacobian(self.h, self.state_estimate)
        S = H@self.covariance@H.T + self.R
        kalman_gain = self.covariance@H.T@np.linalg.inv(S)

        self.state_estimate = self.state_estimate+ kalman_gain@residual
        self.covariance = (np.eye(self.state_estimate) - kalman_gain@H)@self.covariance

class QuaternionMEKF(KalmanFilter):
    '''Class detailing a Quaternion-Euler Velocity MEKF'''
    def xi(self, quaternion, angular_velocities):
        angular_velocity_quat = Quaternion(scalar = 0, vector = angular_velocities*self.dt)
        return 1/2*quaternion*angular_velocity_quat
    
    def cross_product_matrix(self, mat):
        '''Gets the skew symmetric matrix for mat'''
        return np.cross(np.eye(3), mat)
    
    def predict(self):
        previous_quaternion = Quaternion(self.state_estimate[:4])
        angular_velocities = self.state_estimate[4:]
        estimated_quaternion = previous_quaternion + self.xi(previous_quaternion, angular_velocities)
        predicted_quaternion_normal = estimated_quaternion/estimated_quaternion.norm
        self.state_estimate = np.concatenate([predicted_quaternion_normal.elements, angular_velocities], axis = 0)
    def update(self, measurements):
        raise NotImplementedError("This function makes no sense in MEKF")
    def iterate(self, measurements):
        measured_quaternion = Quaternion(measurements[:4])
        #smart people found this based on the update function f(quaternion, w) = quaternion + xi(quaternion, w)
        #i am not smart people. do not go to me for advice on why F and H look like this.
        F = np.concatenate([
            np.concatenate([-self.cross_product_matrix(self.state_estimate[4:]), -np.eye(3)], axis = 1),
            np.zeros((3,6))
        ])
        H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ])
        previous_quaternion = Quaternion(self.state_estimate[:4])
        angular_velocities = self.state_estimate[4:]
        estimated_quaternion = previous_quaternion + self.xi(previous_quaternion, angular_velocities)

        #standard kalman filter stuff
        estimated_cov_diff = F@self.covariance@F.T + self.Q
        S = H@estimated_cov_diff@H.T + self.R
        kalman_gain = estimated_cov_diff@H.T@np.linalg.inv(S)
        predicted_covariance = (np.eye(6) - kalman_gain@H)@estimated_cov_diff

        #multiplicative EKF specific calculations
        error_quaternion = measured_quaternion*(previous_quaternion.inverse)
        a = error_quaternion.elements[1:]/error_quaternion[0]
        d_state = kalman_gain@a
        new_angular_velocities = angular_velocities + d_state[3:]
        d_attitude = Quaternion(vector = d_state[:3], scalar = 2)#gives the attitude error

        #updating d_quat and applying it
        d_quat = d_attitude* previous_quaternion + self.xi(d_attitude* previous_quaternion, angular_velocities)
        predicted_quaternion_unnormal = estimated_quaternion + d_quat
        predicted_quaternion_normal = predicted_quaternion_unnormal/predicted_quaternion_unnormal.norm

        self.state_estimate = np.concatenate([predicted_quaternion_normal.elements, new_angular_velocities], axis = 0)
        self.covariance = predicted_covariance

    def get_quaternion(self):
        return self.state_estimate[:4]
    def get_w(self):
        return self.state_estimate[4:]