# Codes are heavily borrowed from Chen Kong
import numpy as np
from utils import estimate_params

# Generate Camera Matrix Randomly.

K = np.array([[1, 0, 1e2], [0, 1, 1e2], [0, 0, 1]])
R, _, _ = np.linalg.svd(np.random.rand(3, 3))
if np.linalg.det(R) < 0:
    R = -R
t = np.random.rand(3,1)
Rt = np.concatenate((R, t), axis=1)
P = np.matmul(K, Rt)

# Generate 2D and 3D Points Randomly.

N = 10
X = np.random.rand(3, N)
x = np.matmul(P, np.concatenate((X, np.ones((1, N))), axis=0))
x[0, :] = x[0, :] / x[2, :]
x[1, :] = x[1, :] / x[2, :]

# test K, R, t 
# P_est= estimate_pose(x, X)
K_est, R_est, t_est = estimate_params(P)
print('Intrinsic error with clean 2D points is {}'.format(np.linalg.norm(K_est/K_est[-1, -1] - K/K[-1, -1])/np.linalg.norm(K/K[-1,-1])))
print('Rotation error with clean 2D points is {}'.format(np.linalg.norm(R_est - R)))
print('Translation error with clean 2D points is {}'.format(np.linalg.norm(t_est - t)))