# -*- coding: utf-8 -*-
"""
Created on Mon May 11 20:36:43 2026

@author: 18307
"""
import numpy as np
from utils import utils_feature_loading
from utils import utils_visualization

pcc_sample = utils_feature_loading.read_fcs_global_average("seed", "pcc")["alpha"]
plv_sample = utils_feature_loading.read_fcs_global_average("seed", "plv")["alpha"]
pli_sample = utils_feature_loading.read_fcs_global_average("seed", "pli")["alpha"]

np.fill_diagonal(pcc_sample, np.nan)
np.fill_diagonal(plv_sample, np.nan)
np.fill_diagonal(pli_sample, np.nan)

utils_visualization.draw_projection(pcc_sample, "", "", "", show_colorbar=False, cmap="RdBu_r")

from matplotlib import cm
from matplotlib.colors import ListedColormap
rdBu_r = cm.get_cmap("RdBu_r", 256)
plv_colors = rdBu_r(np.linspace(0.5, 1.0, 128))
cmap_plv = ListedColormap(plv_colors, name="RdBu_r_positive_half")

utils_visualization.draw_projection(plv_sample, "", "", "", show_colorbar=False, cmap=cmap_plv)
utils_visualization.draw_projection(pli_sample, "", "", "", show_colorbar=False, cmap=cmap_plv)

# fusion
import feature_engineering
fused_additive_sample = feature_engineering.normalize_matrix(pcc_sample) +  plv_sample
np.fill_diagonal(fused_additive_sample, np.nan)
utils_visualization.draw_projection(fused_additive_sample, "", "", "", show_colorbar=False, cmap=cmap_plv)

fused_multi_sample = pcc_sample*plv_sample
np.fill_diagonal(fused_multi_sample, np.nan)
utils_visualization.draw_projection(fused_multi_sample, "", "", "", show_colorbar=False, cmap="RdBu_r")

import feature_fusion
fused_spliced_sample = feature_fusion.feature_fusion_diagonal_blocking(pcc_sample, plv_sample)
np.fill_diagonal(fused_spliced_sample, np.nan)
utils_visualization.draw_projection(fused_spliced_sample, "", "", "", show_colorbar=False, cmap="RdBu_r")

fused_spliced_sample = feature_fusion.feature_fusion_triangle_blocking(pcc_sample, plv_sample)
np.fill_diagonal(fused_spliced_sample, np.nan)
utils_visualization.draw_projection(fused_spliced_sample, "", "", "", show_colorbar=False, cmap="RdBu_r")

# PC-AEC
params_4_PCAEC ={'fusion_type': 'sigmoid_gating',
                 'k': 10.0, # gate sharpness
                 'percentile': 25, # confidence threshold
                 'power': 1, # for power gating variant
                 'normalization_basis': False,
                 'normalization_modifier': False,
                 'scale': (0, 1)}

pc_aec = feature_fusion.feature_fusion_sigmoid_gating(pcc_sample, plv_sample, params_4_PCAEC)
utils_visualization.draw_projection(pc_aec, "", "", "", show_colorbar=False, cmap="RdBu_r")
