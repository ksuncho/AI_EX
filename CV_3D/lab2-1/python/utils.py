import numpy as np

def find_correspondence(M, S, R, t, dist_thres):
    """
    params:
        M: Model Point Set, (3, N_M) array
        S: Scene Point Set, (3, N_S) array
        R: Rotation matrix, (3, 3) array
        t: translation vector, (3, 1) array
        dist_thres: distance threshold to reject some pairs
    returns:
        corr:
            correspondence, (N_p, 2) array, N_p is number of pairs
            first column : index of Scene Point set
            second column: index of Model Point set  
    """
    ###############################################################################################################
    ## TODO : You need to find valid (distance is less than threshold) correspondence pairs
    return corr

def find_registration(M, S, corr):
    """
    params:
        M: Model Point Set
        S: Scene Point Set
        corr:
            correspondence, (N_p, 2) array
            first column : index of Scene Point set
            second column: index of Model Point set  
    returns:
        R: rotation matrix, (3, 3) array
        t: translation vector, (3, 1) array
        MSE
    """
    ###############################################################################################################
    ## TODO : You need to find Rotation matrix and translation vector. Also calculate MSE for estimated R and t.

    return R, t, MSE

def random_sampling(pc, sample_num):
    """
    sample point cloud randomly
    params:
        pc: point set, (3, N) array
        sample_num: number of samples
    returns:
        sampled_pc: sampled point set
    """
    if pc.shape[1] >= sample_num:
        sampled_pc = np.transpose(np.random.permutation(np.transpose(pc)))[:, :sample_num]
    else:
        raise ValueError
    return sampled_pc
