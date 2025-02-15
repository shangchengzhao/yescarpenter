import numpy as np
from scipy.stats import spearmanr
from scipy.spatial.distance import cdist, pdist

def construct_RDM(data, n_target, method = "cityblock"):
    '''
    For one column data, calculate Manhattan distance; for multiple columns, calculate 1-spearman correlation coefficients
    Usage:
        construct_RDM(data, n_target, method = "cityblock")
    '''
    if data.shape[0] == n_target:
        pass
    elif data.shape[1] == n_target:
        data = data.T
    else:
        print(f'The input data does not have {n_target} observations. It has {data.shape[0]} rows and {data.shape[1]} columns. Please check the input data.')

    if method == "spearman":
        corr_matrix, _ = spearmanr(data, nan_policy='omit')
        rdm = 1 - corr_matrix
    elif method == "cityblock":
        rdm = cdist(data, data, metric='cityblock')
    
    return rdm


def do_RSA(rdm1, rdm2, n_perm=1000):
    '''
    calculate the Spearman correlation between two RDMs(lower triangle)
        and do permutation
        
    do_RSA(rdm1, rdm2, n_perm=1000)
    '''
    # rdm: n x n matrix

    # remove the upper triangle
    ind = np.tril_indices(rdm1.shape[0], k=-1)

    rdm1_f = rdm1[ind]
    rdm2_f = rdm2[ind]

    # calculate the correlation
    r, _ = spearmanr(rdm1_f, rdm2_f)

    # permutation
    # n_perm = 1000
    perm_r = np.zeros(n_perm)
    for i in range(n_perm):
        perm_r[i], _ = spearmanr(rdm1_f, np.random.permutation(rdm2_f))
    perm_p = float(np.sum(perm_r > r) / n_perm)
    print(f"p = {np.sum(perm_r > r)} / {n_perm}")

    return [r, perm_p]