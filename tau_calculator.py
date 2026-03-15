# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 02:18:14 2026

@author: 18307
"""

import numpy as np
import pandas as pd

def estimate_tau_from_plv_percentile(plv_matrices, percentile=75):
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

    n_samples, n_channels, _ = plv_matrices.shape

    # extract upper triangle indices
    iu = np.triu_indices(n_channels, k=1)

    # collect all PLV edges
    plv_values = plv_matrices[:, iu[0], iu[1]].reshape(-1)

    # compute percentile
    tau = np.percentile(plv_values, percentile)

    return tau

import feature_fusion
from utils import utils_feature_loading
from utils import utils_visualization

pcc = utils_feature_loading.read_fcs('seed', 'sub1ex2', 'pcc')
plv = utils_feature_loading.read_fcs('seed', 'sub1ex2', 'plv')

pcc = utils_feature_loading.read_fcs_global_average('seed', 'pcc')
plv = utils_feature_loading.read_fcs_global_average('seed', 'plv')

pcc_alpha, pcc_beta, pcc_gamma = pcc['alpha'].reshape(1,62,62), pcc['beta'].reshape(1,62,62), pcc['gamma'].reshape(1,62,62)
plv_alpha, plv_beta, plv_gamma = plv['alpha'].reshape(1,62,62), plv['beta'].reshape(1,62,62), plv['gamma'].reshape(1,62,62)

labels = utils_feature_loading.read_labels('seed', header=True)

#
tau_a = estimate_tau_from_plv_percentile(plv_alpha)
print(tau_a)

tau_b = estimate_tau_from_plv_percentile(plv_beta)
print(tau_b)

tau_g = estimate_tau_from_plv_percentile(plv_gamma)
print(tau_g)