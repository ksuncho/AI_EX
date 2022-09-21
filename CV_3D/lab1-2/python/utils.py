import numpy as np

def estimate_pose(x, X):
    """
    Args:
        x: 2D points with shape [2, N]
        X: 3D points with shape [3, N]
    Output:
        P: Camera matrix with shape [3, 4]
    """
    ###############################################################################################################
    ## TODO : You need to find Camera matrix P given 2D and 3D points x and X.
    # Hint: use linear regression for elements of P (assuming that P[2, 3] = 1)
    '''
    N = x.shape[1]
    X_homogeneous = np.concatenate((x, np.ones((1, N))), axis=0)
    X1 = X * x[0, :]
    X2 = X * x[1, :]
    A1 = np.concatenate((X_homogeneous, np.zeros((4, N)), -X1), axis=0)
    A2 = np.concatenate((np.zeros((4, N)), X_homogeneous, -X2), axis=0)
    A = np.concatenate((A1, A2), axis=1)
    B = np.concatenate((x[0, :], x[1, :]), axis=0).reshape(1, 2 * N)
    P_tmp = np.matmul(B, np.transpose(A))
    P_flat = np.matmul(P_tmp, np.linalg.inv(np.matmul(A, np.transpose(A)))).reshape(11,)
    P = np.zeros((3, 4))
    P[0, :] = P_flat[:4]
    P[1, :] = P_flat[4:8]
    P[2, :3] = P_flat[8:]
    P[2, 3] = 1
    return P
    '''
    N = x.shape[1]
    X_homogeneous = np.concatenate((X, np.ones((1, N))), axis=0)
    X1 = X * x[0, :]
    X2 = X * x[1, :]
    A1 = np.concatenate((X_homogeneous, np.zeros((4, N)), -X1), axis=0)
    A2 = np.concatenate((np.zeros((4, N)), X_homogeneous, -X2), axis=0)
    A = np.concatenate((A1, A2), axis=1)
    B = np.concatenate((x[0, :], x[1, :]), axis=0).reshape(1, 2 * N)
    P_tmp = np.matmul(B, np.transpose(A))
    P_flat = np.matmul(P_tmp, np.linalg.inv(np.matmul(A, np.transpose(A)))).reshape(11,)
    P = np.zeros((3, 4))
    P[0, :] = P_flat[:4]
    P[1, :] = P_flat[4:8]
    P[2, :3] = P_flat[8:]
    P[2, 3] = 1

    return P
def estimate_params(P):
    """
    Args:
        P: Camera matrix with shape [3,4]
    Output:
        (K, R, t) : intrinsic matrix K with shape [3,3],
                    rotation matrix R with shape[3,3],
                    translation t with shape [3,1]
    """
    ###############################################################################################################
    ## TODO : You need to estimate intrinsic matrix K, rotation matrix R, and translation t from given camera matrix P.
    A = P[:3, :3]
    c= np.matmul(-np.linalg.inv(A), P[:, 3])
    E = np.fliplr(np.eye(3))
    A0 = np.matmul(E, A)
    Q0, R0 = np.linalg.qr(np.transpose(A0))
    K_tmp = np.matmul(E, np.transpose(R0))
    K = np.matmul(K_tmp, E)
    R = np.matmul(E, np.transpose(Q0))
    if np.linalg.det(R) < 0:
        K = -K
        R = -R
    t = -np.matmul(R, c)
    return (K, R, t)

def project(P, X):
    N = X.shape[1]
    xProj = np.matmul(P, np.concatenate((X, np.ones((1, N))), axis=0))
    xProj[0, :] = xProj[0, :] / xProj[2, :]
    xProj[1, :] = xProj[1, :] / xProj[2, :]
    return xProj[:2, :]