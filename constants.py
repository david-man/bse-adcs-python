#A file of general constants

import numpy as np
MAG_BIAS = 0
PHOTO_BIAS = 0

ORBIT_Q = np.eye(6)*0.001#change
ORBIT_R = np.array([MAG_BIAS**2])

BDOT_Q = np.eye(9)#change
BDOT_R = np.eye(3)*(MAG_BIAS**2)

ROT_Q = np.eye(6)#use the Q equation from Table II of the "backup satellite" paper
ROT_R = np.eye(3)#use Equation 26 of the same paper when actually finding R
