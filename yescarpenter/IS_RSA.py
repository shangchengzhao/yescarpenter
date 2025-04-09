import numpy as np
from scipy.stats import spearmanr
from scipy.spatial.distance import cdist, pdist
import matplotlib.pyplot as plt

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
        corr_matrix, _ = spearmanr(data, axis = 1, nan_policy='omit')
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
    permutation_histogram(r, perm_r)

    return [r, perm_p]

def permutation_histogram(r, perm_r):
    '''
    Plot the histogram of permutation results on a created figure.
    '''
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(perm_r, bins=50, color='gray')
    ax.axvline(r, color='red', alpha=0.5)
    ax.text(r-0.01, 5, f'Observed r = {r:.2}', color='red', fontsize=16)
    ax.set_xlabel('Permutation r', fontsize=20)
    ax.set_ylabel('Frequency', fontsize=20)
    ax.tick_params(axis='both', labelsize=18)
    # ax.set_title('Permutation distribution', fontsize=22)
    plt.show()

def maximal_permutation_test(data, iv_single, iv_multiplecomp, n_perm = 1000):
    '''
    This fuction is used to address multiple comparison, \
        which provides an alternative of Bonferroni correction.
    
    - data: For IS-RSA, each row is a subject, while each column is a variable. \
        For example, if you have 20 subjects and 5 variables, the shape of data is (20, 5).
    - iv_single: the independent variable that will be shuffled and compare across iv_multiplecomp
    - iv_multiplecomp: the independent variable that are inter-related and elicit the multiple comparison problem
    - n_perm: number of permutation
    '''

    # construct the RDMs
    # convert the data into array
    ivSarray = data[iv_single].values.reshape(-1, 1)
    rdmS = construct_RDM(ivSarray, data.shape[0], method = "cityblock")

    # remove the upper triangle
    ind = np.tril_indices(rdmS.shape[0], k=-1)
    rdmS_f = rdmS[ind]

    # observed_r: dictionary to store the observed correlation
    observed_r = {}
    for ivM in iv_multiplecomp:
        # Construct the RDM for the independent variable component ivM.
        ivM_array = data[ivM].values.reshape(-1, 1)
        rdmM = construct_RDM(ivM_array, data.shape[0], method="cityblock")
        rdmM_f = rdmM[ind]
        
        # Compute the observed correlation between rdmS_f and rdmM_f.
        r, _ = spearmanr(rdmS_f, rdmM_f)
        observed_r[ivM] = r

    perm_r = np.zeros(n_perm)    
    for iperm in range(n_perm):
        # define the max r
        max_r_null = -np.inf

        # shuffle the data
        rdmS_shuffled = np.random.permutation(rdmS_f)

        for ivM in iv_multiplecomp:
            # construct the RDMs
            ivMarray = data[ivM].values.reshape(-1, 1)
            rdmM = construct_RDM(ivMarray, data.shape[0], method = "cityblock")
            rdmM_f = rdmM[ind]

            # calculate the psudo correlation
            r, _ = spearmanr(rdmS_shuffled, rdmM_f)

            # update the max r
            if r > max_r_null:
                max_r_null = r
            
        perm_r[iperm] = max_r_null
            
    # calculate the p-value
    perm_p = float(np.sum(perm_r > observed_r[ivM]) / n_perm)
    print(f"p = {np.sum(perm_r > observed_r[ivM])} / {n_perm}")

    return [perm_r, perm_p, observed_r]

