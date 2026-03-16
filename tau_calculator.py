# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 15:06:27 2026

@author: usouu
"""

import numpy as np

def estimate_tau_from_matrix_percentile(matrix, percentile=75):
    """
    Estimate tau (PLV gating threshold) from PLV distribution using percentile.

    Parameters
    ----------
    plv_matrices : ndarray
        PLV matrices with shape (n_samples, n_channels, n_channels)

    percentile : float
        Percentile used to estimate tau (e.g., 50, 75, 80, 90)

    Returns
    -------
    tau : float
        Estimated PLV threshold
    """

    n_channels, _ = matrix.shape

    # extract upper triangle indices
    iu = np.triu_indices(n_channels, k=1)
    
    # collect all PLV edges
    values = matrix[iu[0], iu[1]].reshape(-1)

    # compute percentile
    tau = np.percentile(values, percentile)

    return tau

def estimate_tau_from_matrices_percentile(matrices, percentile=75):
    """
    Estimate tau (PLV gating threshold) from PLV distribution using percentile.

    Parameters
    ----------
    plv_matrices : ndarray
        PLV matrices with shape (n_samples, n_channels, n_channels)

    percentile : float
        Percentile used to estimate tau (e.g., 50, 75, 80, 90)

    Returns
    -------
    tau : float
        Estimated PLV threshold
    """

    n_samples, n_channels, _ = matrices.shape

    # extract upper triangle indices
    iu = np.triu_indices(n_channels, k=1)
    
    # collect all PLV edges
    values = matrices[:, iu[0], iu[1]].reshape(-1)

    # compute percentile
    tau = np.percentile(values, percentile)

    return tau

from utils import utils_feature_loading

# Experiments
# pcc = utils_feature_loading.read_fcs('seed', 'sub1ex1', 'pcc')
# plv = utils_feature_loading.read_fcs('seed', 'sub1ex3', 'plv')
    
# pcc_alpha, pcc_beta, pcc_gamma = pcc['alpha'], pcc['beta'], pcc['gamma']
# plv_alpha, plv_beta, plv_gamma = plv['alpha'], plv['beta'], plv['gamma']

# tau = estimate_tau_from_matrices_percentile(plv_gamma)
# print('tau:', tau)

# Avg
# pcc = utils_feature_loading.read_fcs_global_average('seed', 'pcc')
plv = utils_feature_loading.read_fcs_global_average('seed', 'plv')

# pcc_alpha, pcc_beta, pcc_gamma = pcc['alpha'], pcc['beta'], pcc['gamma']
plv_alpha, plv_beta, plv_gamma = plv['alpha'], plv['beta'], plv['gamma']

tau_1 = estimate_tau_from_matrix_percentile(plv_alpha)
print('tau:', tau_1) # 0.43926288553892395

tau_2 = estimate_tau_from_matrix_percentile(plv_beta)
print('tau:', tau_2) # 0.36658529865891054

tau_3 = estimate_tau_from_matrix_percentile(plv_gamma)
print('tau:', tau_3) # 0.32064941325729696
