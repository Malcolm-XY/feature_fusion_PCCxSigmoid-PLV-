# -*- coding: utf-8 -*-
"""
Created on Thu Jan  1 16:55:00 2026

@author: 18307
"""

import numpy as np

import feature_engineering
# competitors: additive, multiplicative, color_blocking
def feature_fusion_color_blocking(fn_basis, fn_modifier, params={'normalization': True, 'scale': (0,1)}):
    normalization = params.get('normalization', True)
    scale = params.get('scale', (0,1))
    
    # print('Normalization: ', normalization)
    
    if normalization:
        fn_basis = feature_engineering.normalize_matrix(fn_basis, 'minmax', param={'target_range': scale})
        fn_modifier = feature_engineering.normalize_matrix(fn_modifier, 'minmax', param={'target_range': scale})
    else: 
        fn_basis = np.array(fn_basis)
        fn_modifier = np.array(fn_modifier)
    
    upper = np.triu(fn_basis)
    lower = np.tril(fn_modifier)
    
    fn_fussed = upper + lower
    
    return fn_fussed

def feature_fusion_additive(fn_basis, fn_modifier, params={'normalization': True, 'scale': (0,1)}):
    normalization = params.get('normalization', True)
    scale = params.get('scale', (0,1))
    
    if normalization:
        fn_basis = feature_engineering.normalize_matrix(fn_basis, 'minmax', param={'target_range': scale})
        fn_modifier = feature_engineering.normalize_matrix(fn_modifier, 'minmax', param={'target_range': scale})
    else: 
        fn_basis = np.array(fn_basis)
        fn_modifier = np.array(fn_modifier)
    
    fn_fussed = fn_basis + fn_modifier
    
    return fn_fussed

def feature_fusion_multiplicative(fn_basis, fn_modifier, params={'normalization': True, 'scale': (0,1)}):
    normalization = params.get('normalization', True)
    scale = params.get('scale', (0,1))
    
    if normalization:
        fn_basis = feature_engineering.normalize_matrix(fn_basis, 'minmax', param={'target_range': scale})
        fn_modifier = feature_engineering.normalize_matrix(fn_modifier, 'minmax', param={'target_range': scale})
    else: 
        fn_basis = np.array(fn_basis)
        fn_modifier = np.array(fn_modifier)
    
    fn_fussed = fn_basis * fn_modifier
    
    return fn_fussed

# proposed
from utils import utils_feature_loading
def feature_fusion_power_gating(fn_basis, fn_modifier, params={'power': 1, 'normalization': True, 'scale': (0,1)}):
    power = params.get('power', 1)
    normalization = params.get('normalization', True)
    scale = params.get('scale', (0,1))
    
    # normalization
    if normalization:
        fn_basis = feature_engineering.normalize_matrix(fn_basis, 'minmax', param={'target_range': scale})
        fn_modifier = feature_engineering.normalize_matrix(fn_modifier, 'minmax', param={'target_range': scale})
    else: 
        fn_basis = np.array(fn_basis)
        fn_modifier = np.array(fn_modifier)
    
    # operation
    alpha = fn_modifier**power
    fn_fussed = fn_basis * alpha
    
    return fn_fussed

def feature_fusion_power_gating_parameterized(fn_basis, fn_modifier=None, params={'power': 1, 'modifier_parameterization': 'plv', 
                                                                                   'normalization': True, 'scale': (0,1)}):
    power = params.get('power', 1)
    parameterization_source = params.get('modifier_parameterization', 'plv')    
    normalization = params.get('normalization', True)
    scale = params.get('scale', (0,1))

    # construct modifer
    fcs_global_averaged = utils_feature_loading.read_fcs_global_average('seed', parameterization_source, 'joint', range(1,6))
    alpha_global_averaged = fcs_global_averaged['alpha']
    beta_global_averaged = fcs_global_averaged['beta']
    gamma_global_averaged = fcs_global_averaged['gamma']
    
    fn_modifier = alpha_global_averaged + beta_global_averaged + gamma_global_averaged
    
    # normalization
    if normalization:
        fn_basis = feature_engineering.normalize_matrix(fn_basis, 'minmax', param={'target_range': scale})
        fn_modifier = feature_engineering.normalize_matrix(fn_modifier, 'minmax', param={'target_range': scale})
    else: 
        fn_basis = np.array(fn_basis)
        fn_modifier = np.array(fn_modifier)
    
    # operation
    alpha = fn_modifier**power
    fn_fussed = fn_basis * alpha
    
    return fn_fussed

def feature_fusion_sigmoid_gating(fn_basis, fn_modifier, params={'k': 10.0, # gate sharpness
                                                                 'tau': 0.5, # confidence threshold
                                                                 'normalization': True, 'scale': (0, 1)}):
    k = params.get('k', 10.0)
    tau = params.get('tau', 0.5)
    normalization = params.get('normalization', True)
    scale = params.get('scale', (0, 1))

    # normalization
    if normalization:
        fn_basis = feature_engineering.normalize_matrix(fn_basis, 'minmax', param={'target_range': scale})
        fn_modifier = feature_engineering.normalize_matrix(fn_modifier, 'minmax', param={'target_range': scale})
    else:
        fn_basis = np.array(fn_basis)
        fn_modifier = np.array(fn_modifier)
    
    # sigmoid confidence gate
    alpha = 1.0 / (1.0 + np.exp(-k * (fn_modifier - tau)))
    
    # opertion
    fn_fused = fn_basis * alpha
    
    return fn_fused

def feature_fusion_sigmoid_gating_parameterized(fn_basis, fn_modifier=None, params={'k': 10.0, # gate sharpness
                                                                                    'tau': 0.5, # confidence threshold
                                                                                    'modifier_parameterization': 'plv',
                                                                                    'normalization': True, 'scale': (0, 1)}):
    k = params.get('k', 10.0)
    tau = params.get('tau', 0.5)
    parameterization_source = params.get('modifier_parameterization', 'plv')
    normalization = params.get('normalization', True)
    scale = params.get('scale', (0, 1))
    
    # construct modifer
    fcs_global_averaged = utils_feature_loading.read_fcs_global_average('seed', parameterization_source, 'joint', range(1,6))
    alpha_global_averaged = fcs_global_averaged['alpha']
    beta_global_averaged = fcs_global_averaged['beta']
    gamma_global_averaged = fcs_global_averaged['gamma']
    
    fn_modifier = alpha_global_averaged + beta_global_averaged + gamma_global_averaged

    # normalization
    if normalization:
        fn_basis = feature_engineering.normalize_matrix(fn_basis, 'minmax', param={'target_range': scale})
        fn_modifier = feature_engineering.normalize_matrix(fn_modifier, 'minmax', param={'target_range': scale})
    else:
        fn_basis = np.array(fn_basis)
        fn_modifier = np.array(fn_modifier)
    
    # sigmoid confidence gate
    alpha = 1.0 / (1.0 + np.exp(-k * (fn_modifier - tau)))
    
    # opertion
    fn_fused = fn_basis * alpha
    
    return fn_fused

# executor
def feature_fusion(fns_1, fns_2, params={}):
    fussion_type = params.get('fussion_type', None).lower()
    
    fussion_type_valid = {'color_blocking', 'additive', 'multiplicative', 
                          'power_gating', 'power_gating_parameterized', 
                          'sigmoid_gating', 'sigmoid_gating_parameterized'}
    if fussion_type not in fussion_type_valid:
        raise ValueError(f"Invalid filter '{fussion_type}'. Allowed filters: {fussion_type_valid}")
        
    # competitors: additive, multiplicative, color_blocking
    elif fussion_type == 'additive':
        fn_fussed = feature_fusion_additive(fns_1, fns_2)
    elif fussion_type == 'multiplicative':
        fn_fussed = feature_fusion_multiplicative(fns_1, fns_2)
    elif fussion_type == 'color_blocking':
        fn_fussed = feature_fusion_color_blocking(fns_1, fns_2, params)
    
    # proposed power_gating
    elif fussion_type == 'power_gating':
        fn_fussed = feature_fusion_power_gating(fns_1, fns_2, params)
    elif fussion_type == 'power_gating_parameterized':
        fn_fussed = feature_fusion_power_gating_parameterized(fns_1, fns_2, params)
    
    # proposed PCC(sigmoid(PLV))
    elif fussion_type == 'sigmoid_gating':
        fn_fussed = feature_fusion_sigmoid_gating(fns_1, fns_2, params)
    elif fussion_type == 'sigmoid_gating_parameterized':
        fn_fussed = feature_fusion_sigmoid_gating_parameterized(fns_1, fns_2, params)
    
    return fn_fussed

if __name__ == "__main__":
    # from utils import utils_feature_loading
    from utils import utils_visualization
    
    feature_basis='pcc'
    feature_modifier='plv'
    
    fcs_basis_global_averaged = utils_feature_loading.read_fcs_global_average('seed', feature_basis, 'joint', range(1,6))
    alpha_basis_global_averaged = fcs_basis_global_averaged['alpha']
    beta_basis_global_averaged = fcs_basis_global_averaged['beta']
    gamma_basis_global_averaged = fcs_basis_global_averaged['gamma']
    
    fcs_modifier_global_averaged = utils_feature_loading.read_fcs_global_average('seed', feature_modifier, 'joint', range(1,6))
    alpha_modifier_global_averaged = fcs_modifier_global_averaged['alpha']
    beta_modifier_global_averaged = fcs_modifier_global_averaged['beta']
    gamma_modifier_global_averaged = fcs_modifier_global_averaged['gamma']
    
    utils_visualization.draw_projection(alpha_basis_global_averaged)
    utils_visualization.draw_projection(alpha_modifier_global_averaged)
    
    #
    params = {'fussion_type': 'additive', 'normalization': True}
    alpha_fussed = feature_fusion(alpha_basis_global_averaged, alpha_modifier_global_averaged, params)
    utils_visualization.draw_projection(alpha_fussed)
    
    params = {'fussion_type': 'multiplicative', 'normalization': True}
    alpha_fussed = feature_fusion(alpha_basis_global_averaged, alpha_modifier_global_averaged, params)
    utils_visualization.draw_projection(alpha_fussed)
    
    params = {'fussion_type': 'color_blocking', 'normalization': True}
    alpha_fussed = feature_fusion(alpha_basis_global_averaged, alpha_modifier_global_averaged, params)
    utils_visualization.draw_projection(alpha_fussed)
    
    # 
    params={'fussion_type': 'sigmoid_gating', 
            'k': 10, 'tau': 0.2,
            'normalization': True}
    alpha_fussed = feature_fusion(alpha_basis_global_averaged, alpha_modifier_global_averaged, params)
    utils_visualization.draw_projection(alpha_fussed)
    
    params={'fussion_type': 'sigmoid_gating_parameterized', 
            'k': 10, 'tau': 0.2,
            'modifier_parameterization': 'plv',
            'normalization': True}
    alpha_fussed = feature_fusion(alpha_basis_global_averaged, None, params)
    utils_visualization.draw_projection(alpha_fussed)