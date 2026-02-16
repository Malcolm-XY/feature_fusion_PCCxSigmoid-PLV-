# -*- coding: utf-8 -*-
"""
Created on Sun Feb 15 23:24:35 2026

@author: 18307
"""
import numpy as np

def redundancy_4_matrix(matrix, absolute=True):
    matrix = np.array(matrix)
    width_1, width_2 = matrix.shape
    repeat = width_1

    corr = []
    for i in range(repeat):
        for j in range(repeat):
            if i != j:
                edge_1, edge_2 = matrix[i], matrix[j]
                _corr = np.corrcoef(edge_1, edge_2)[0, 1]
                
                if absolute:
                   _corr = np.abs(_corr)
                    
                corr.append(_corr)
                
    redundancy_ = (1/(width_1*(width_1 - 1))) * np.sum(corr)
    return redundancy_

import feature_fusion
def redundancy_4_fns(k, tau,
                    avg_pcc_alpha, avg_plv_alpha,
                    avg_pcc_beta,  avg_plv_beta,
                    avg_pcc_gamma, avg_plv_gamma,
                    return_detail=False):
    params = {'k': float(k), 'tau': float(tau)}

    fused_alpha = feature_fusion.feature_fusion_sigmoid_gating(avg_pcc_alpha, avg_plv_alpha, params=params)
    fused_beta  = feature_fusion.feature_fusion_sigmoid_gating(avg_pcc_beta,  avg_plv_beta,  params=params)
    fused_gamma = feature_fusion.feature_fusion_sigmoid_gating(avg_pcc_gamma, avg_plv_gamma, params=params)

    r_a = redundancy_4_matrix(fused_alpha)
    r_b = redundancy_4_matrix(fused_beta)
    r_g = redundancy_4_matrix(fused_gamma)

    r_mean = np.mean([r_a, r_b, r_g])

    if return_detail:
        return r_mean, (r_a, r_b, r_g)
    return r_mean

if __name__ == '__main__':
    from utils import utils_feature_loading
    avg_pcc = utils_feature_loading.read_fcs_global_average('seed', 'pcc')
    avg_plv = utils_feature_loading.read_fcs_global_average('seed', 'plv')
    
    avg_pcc_alpha, avg_pcc_beta, avg_pcc_gamma = avg_pcc['alpha'], avg_pcc['beta'], avg_pcc['gamma']
    avg_plv_alpha, avg_plv_beta, avg_plv_gamma = avg_pcc['alpha'], avg_pcc['beta'], avg_pcc['gamma']
    
    params={'k': 100, 'tau': 0.2}
    avg_fused_alpha = feature_fusion.feature_fusion_sigmoid_gating(avg_pcc_alpha, avg_plv_alpha, params=params)
    avg_fused_beta = feature_fusion.feature_fusion_sigmoid_gating(avg_pcc_beta, avg_plv_beta, params=params)
    avg_fused_gamma = feature_fusion.feature_fusion_sigmoid_gating(avg_pcc_gamma, avg_plv_gamma, params=params)
    
    # redundancy
    redundancy_fused_alpha = redundancy_4_matrix(avg_fused_alpha)
    redundancy_fused_beta = redundancy_4_matrix(avg_fused_beta)
    redundancy_fused_gamma = redundancy_4_matrix(avg_fused_gamma)
    redundancy_fused = np.mean([redundancy_fused_alpha, redundancy_fused_beta, redundancy_fused_gamma])
    
    # 
    r_mean = redundancy_4_fns(100, 0.2, avg_pcc_alpha, avg_plv_alpha, 
                              avg_pcc_beta, avg_plv_beta, 
                              avg_pcc_gamma, avg_plv_gamma)
    
    # optimization
    