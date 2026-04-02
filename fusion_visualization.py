# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 17:03:23 2026

@author: usouu
"""

import numpy as np

from utils import utils_feature_loading
from utils import utils_visualization

#
channels = utils_feature_loading.read_distribution("seed")["channel"]
params = [channels, channels, False, 10]

# original
pcc_sample = utils_feature_loading.read_fcs_global_average("seed", "pcc")
plv_sample = utils_feature_loading.read_fcs_global_average("seed", "plv")
pli_sample = utils_feature_loading.read_fcs_global_average("seed", "pli")

pcc_alpha_sample = pcc_sample["alpha"]
plv_alpha_sample = plv_sample["alpha"]
pli_alpha_sample = pli_sample["alpha"]

np.fill_diagonal(pcc_alpha_sample, np.mean(pcc_alpha_sample))
np.fill_diagonal(plv_alpha_sample, np.mean(plv_alpha_sample))
np.fill_diagonal(pli_alpha_sample, np.mean(pli_alpha_sample))

utils_visualization.draw_projection(pcc_alpha_sample, "PCC Alpha Sample", *params)
utils_visualization.draw_projection(plv_alpha_sample, "PLV Alpha Sample", *params)
utils_visualization.draw_projection(pli_alpha_sample, "PLI Alpha Sample", *params)

# general fusion
additive_alpha_sample_cv = pcc_alpha_sample + plv_alpha_sample
multiplicative_alpha_sample_cv = pcc_alpha_sample * plv_alpha_sample

additive_alpha_sample_ci = pcc_alpha_sample + pli_alpha_sample
multiplicative_alpha_sample_ci = pcc_alpha_sample * pli_alpha_sample

np.fill_diagonal(additive_alpha_sample_cv, np.mean(additive_alpha_sample_cv))
np.fill_diagonal(multiplicative_alpha_sample_cv, np.mean(multiplicative_alpha_sample_cv))
np.fill_diagonal(additive_alpha_sample_ci, np.mean(additive_alpha_sample_ci))
np.fill_diagonal(multiplicative_alpha_sample_ci, np.mean(multiplicative_alpha_sample_ci))

utils_visualization.draw_projection(additive_alpha_sample_cv, "PCC + PLV Alpha Sample", *params)
utils_visualization.draw_projection(multiplicative_alpha_sample_cv, "PCC x PLV Alpha Sample", *params)
utils_visualization.draw_projection(additive_alpha_sample_ci, "PCC + PLI Alpha Sample", *params)
utils_visualization.draw_projection(multiplicative_alpha_sample_ci, "PCC x PLI Alpha Sample", *params)

# splicing fusion
splicing1_sample_cv = np.tril(pcc_alpha_sample, k=0) + np.triu(plv_alpha_sample, k=0)
splicing1_sample_ci = np.tril(pcc_alpha_sample, k=0) + np.triu(pli_alpha_sample, k=0)

np.fill_diagonal(splicing1_sample_cv, np.mean(splicing1_sample_cv))
np.fill_diagonal(splicing1_sample_ci, np.mean(splicing1_sample_ci))

utils_visualization.draw_projection(splicing1_sample_cv, "Splicing-1, PCC & PLV Alpha Sample", *params)
utils_visualization.draw_projection(splicing1_sample_ci, "Splicing-1, PCC & PLI Alpha Sample", *params)

# splicing fusion
length = len(pcc_alpha_sample)
A = np.zeros([2*length, 2*length])
B, C = A.copy(), A.copy()

A[:length, :length] = pcc_alpha_sample
B[length:, length:] = plv_alpha_sample
C[length:, length:] = pli_alpha_sample

splicing2_sample_cv = A + B
splicing2_sample_ci = A + C

np.fill_diagonal(splicing2_sample_cv, np.mean(splicing2_sample_cv))
np.fill_diagonal(splicing2_sample_ci, np.mean(splicing2_sample_ci))

channels_ = list(channels)*2
params_ = [channels_, channels_, False, 15]

utils_visualization.draw_projection(splicing2_sample_cv, "Splicing-2, PCC & PLV Alpha Sample", *params_)
utils_visualization.draw_projection(splicing2_sample_ci, "Splicing-2, PCC & PLI Alpha Sample", *params_)

# PC-AEC
import feature_fusion

params_pcaec={'k': 20.0, # gate sharpness
              'percentile': 30, # confidence threshold
              'normalization': True, 'scale': (0, 1)}

PC_AEC_alpha_sample_cv = feature_fusion.feature_fusion_sigmoid_gating(pcc_alpha_sample, plv_alpha_sample, params_pcaec)
PC_AEC_alpha_sample_ci = feature_fusion.feature_fusion_sigmoid_gating(pcc_alpha_sample, pli_alpha_sample, params_pcaec)

np.fill_diagonal(PC_AEC_alpha_sample_cv, np.mean(PC_AEC_alpha_sample_cv))
np.fill_diagonal(PC_AEC_alpha_sample_ci, np.mean(PC_AEC_alpha_sample_ci))

utils_visualization.draw_projection(PC_AEC_alpha_sample_cv, "PC-AEC, PCC * α(PLV) Alpha Sample", *params)
utils_visualization.draw_projection(PC_AEC_alpha_sample_ci, "PC-AEC, PCC * α(PLI) Alpha Sample", *params)

