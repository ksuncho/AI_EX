# Codes are heavily borrowed from Chen Kong
import numpy as np
from utils import estimate_pose

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


PClean = estimate_pose(x, X)
xProj = np.matmul(PClean, np.concatenate((X, np.ones((1, N))), axis=0))
xProj[0, :] = xProj[0, :] / xProj[2, :]
xProj[1, :] = xProj[1, :] / xProj[2, :]
print('Reprojected Error with clean 2D points is {}'.format(np.linalg.norm(xProj - x)/np.linalg.norm(x)))
print('Pose Error with clean 2D Points is {}'.format(np.linalg.norm(PClean/PClean[-1, -1] - P/P[-1, -1])))
