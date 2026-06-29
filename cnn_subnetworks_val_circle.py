# -*- coding: utf-8 -*-
"""
Created on Thu May 22 09:21:23 2025

@author: usouu
"""
import numpy as np
import pandas as pd

import torch

import cnn_validation
import feature_engineering
from models import models
from utils import utils_feature_loading
from utils import utils_tools

# from tool_read_params_save_xlsx import read_params
from tool_read_params_save_xlsx import save_to_xlsx_sheet
# from tool_read_params_save_xlsx import save_to_xlsx_fitting

# %% cnn subnetworks evaluation circle common
def cnn_subnetworks_evaluation_circle_original_cm(feature_cm='pcc', 
                                                  normalization_for_train=True, valid_type='cross_validation', # 'hold_one_out_validation'
                                                  subject_range=range(6,16), experiment_range=range(1,4),
                                                  node_retention_list=None,
                                                  save=False):
    if node_retention_list is not None:
        channel_selects = {'gamma': node_retention_list,
                           'beta': node_retention_list,
                           'alpha': node_retention_list
                           }
        node_retention_number = len(node_retention_list)
    
    # for traning and testing in CNN
    # labels
    labels = utils_feature_loading.read_labels(dataset='seed', header=True)
    y = torch.tensor(np.array(labels)).view(-1)
    
    # data and evaluation circle
    all_results_list = []
    for sub in subject_range:
        for ex in experiment_range:
            subject_id = f"sub{sub}ex{ex}"
            print(f"Evaluating {subject_id}...")

            # CM/H5
            features = utils_feature_loading.read_fcs('seed', subject_id, feature_cm)
            alpha = features['alpha']
            beta = features['beta']
            gamma = features['gamma']

            # Selected CM           
            alpha_selected = alpha[:,channel_selects['alpha'],:][:,:,channel_selects['alpha']]
            beta_selected = beta[:,channel_selects['beta'],:][:,:,channel_selects['beta']]
            gamma_selected = gamma[:,channel_selects['gamma'],:][:,:,channel_selects['gamma']]
            
            # Normalization before training
            if normalization_for_train:
                alpha_selected = feature_engineering.normalize_matrix(alpha_selected)
                beta_selected = feature_engineering.normalize_matrix(beta_selected)
                gamma_selected = feature_engineering.normalize_matrix(gamma_selected)
            
            x_selected = np.stack((alpha_selected, beta_selected, gamma_selected), axis=1)
            
            # cnn model
            cnn_model = models.CNN_2layers_adaptive_maxpool_3()
            
            # traning and testing
            if valid_type == 'cross_validation':    
                result_CM = cnn_validation.cnn_cross_validation(cnn_model, x_selected, y)
            elif valid_type == 'hold_one_out_validation':
                result_CM = cnn_validation.cnn_validation(cnn_model, x_selected, y)
            else:
                raise NotImplementedError("valid type must be defined.")
            
            # Flatten result and add identifier
            result_flat = {'Identifier': subject_id, **result_CM}
            all_results_list.append(result_flat)
            
    # Convert list of dicts to DataFrame
    df_results = pd.DataFrame(all_results_list)
    
    # Compute mean of all numeric columns (excluding Identifier)
    mean_row = df_results.select_dtypes(include=[np.number]).mean().to_dict()
    mean_row['Identifier'] = 'Average'
    
    # Std
    std_row = df_results.select_dtypes(include=[np.number]).std(ddof=0).to_dict()
    std_row['Identifier'] = 'Std'
    
    df_results = pd.concat([df_results, pd.DataFrame([mean_row, std_row])], ignore_index=True)
    
    # Summary
    df_summary = pd.DataFrame([mean_row, std_row])
    summary_transpose = {
        "accuracy_avg": df_summary["accuracy"][0],
        "acc_std": df_summary["accuracy"][1],

        "f1_score_avg": df_summary["f1_score"][0],
        "f1_std": df_summary["f1_score"][1],

        "recall_avg": df_summary["recall"][0],
        "recall_std": df_summary["recall"][1],

        "loss_avg": df_summary["loss"][0],
        "loss_std": df_summary["loss"][1],
    }

    df_summary_transpose = pd.DataFrame([summary_transpose])
    
    # Save
    if save:        
        folder_name = 'results_(stress_test)_original'
        file_name = f'cnn_evaluation(stress_test)_{feature_cm}_origin.xlsx'
        sheet_name = f'nrn_{node_retention_number}'
        
        save_to_xlsx_sheet(df_results, folder_name, file_name, sheet_name)
        
        # Save Summary
        save_to_xlsx_sheet(df_summary, folder_name, file_name, "summary")
        save_to_xlsx_sheet(df_summary_transpose, folder_name, file_name, "summary_t")  
        
    return df_results

import feature_fusion
def cnn_subnetworks_evaluation_circle_feature_fusion(feature_basis='pcc', feature_modifier='plv',
                                                     params={'fusion_type': 'triangle_blocking',
                                                             'normalization_basis': False, 
                                                             'normalization_modifier': False,
                                                             'scale': (0, 1)},
                                                     normalization_for_train=False,
                                                     valid_type='cross_validation', # 'hold_one_out_validation'
                                                     subject_range=range(6,16), experiment_range=range(1,4),
                                                     node_retention_list=None,
                                                     save=False):
    # subnetworks selects;channel selects------start
    if node_retention_list is not None:
        channel_selects = {'gamma': node_retention_list,
                           'beta': node_retention_list,
                           'alpha': node_retention_list
                           }
        node_retention_number = len(node_retention_list)
    # subnetworks selects;channel selects------end
    
    # for training and testing in CNN------start
    # labels
    labels = utils_feature_loading.read_labels(dataset='seed', header=True)
    y = torch.tensor(np.array(labels)).view(-1)
    
    # data and evaluation circle
    all_results_list = []
    for sub in subject_range:
        for ex in experiment_range:
            subject_id = f"sub{sub}ex{ex}"
            print(f"Evaluating {subject_id}...")

            # FN/H5
            features_basis = utils_feature_loading.read_fcs('seed', subject_id, feature_basis)
            alpha_basis = features_basis['alpha']
            beta_basis = features_basis['beta']
            gamma_basis = features_basis['gamma']
            
            if feature_modifier is not None:
                features_modifier = utils_feature_loading.read_fcs('seed', subject_id, feature_modifier)
                alpha_modifier = features_modifier['alpha']
                beta_modifier = features_modifier['beta']
                gamma_modifier = features_modifier['gamma']
            elif feature_modifier is None:
                alpha_modifier = None
                beta_modifier = None
                gamma_modifier = None
                
            # fussed FN
            alpha_fussed = feature_fusion.feature_fusion(alpha_basis, alpha_modifier, params)
            beta_fussed = feature_fusion.feature_fusion(beta_basis, beta_modifier, params)
            gamma_fussed = feature_fusion.feature_fusion(gamma_basis, gamma_modifier, params)
            
            alpha_fussed = alpha_fussed[:,channel_selects['alpha'],:][:,:,channel_selects['alpha']]
            beta_fussed = beta_fussed[:,channel_selects['beta'],:][:,:,channel_selects['beta']]
            gamma_fussed = gamma_fussed[:,channel_selects['gamma'],:][:,:,channel_selects['gamma']]
            
            # Normalization before training
            if normalization_for_train:
                alpha_fussed = feature_engineering.normalize_matrix(alpha_fussed)
                beta_fussed = feature_engineering.normalize_matrix(beta_fussed)
                gamma_fussed = feature_engineering.normalize_matrix(gamma_fussed)
            
            x_rebuild = np.stack((alpha_fussed, beta_fussed, gamma_fussed), axis=1)
            
            # cnn model
            cnn_model = models.CNN_2layers_adaptive_maxpool_3()
            
            # traning and testing
            if valid_type == 'cross_validation':    
                result_RCM = cnn_validation.cnn_cross_validation(cnn_model, x_rebuild, y)
            elif valid_type == 'hold_one_out_validation':
                result_RCM = cnn_validation.cnn_validation(cnn_model, x_rebuild, y)
            else:
                raise NotImplementedError("valid type must be defined.")
            
            # Flatten result and add identifier
            result_flat = {'Identifier': subject_id, **result_RCM}
            all_results_list.append(result_flat)
            
    # Convert list of dicts to DataFrame
    df_results = pd.DataFrame(all_results_list)
    
    # Compute mean of all numeric columns (excluding Identifier)
    mean_row = df_results.select_dtypes(include=[np.number]).mean().to_dict()
    mean_row['Identifier'] = 'Average'
    
    # Std
    std_row = df_results.select_dtypes(include=[np.number]).std(ddof=0).to_dict()
    std_row['Identifier'] = 'Std'
    
    df_results = pd.concat([df_results, pd.DataFrame([mean_row, std_row])], ignore_index=True)
    
    # Summary
    df_summary = pd.DataFrame([mean_row, std_row])
    summary_transpose = {
        "accuracy_avg": df_summary["accuracy"][0],
        "acc_std": df_summary["accuracy"][1],

        "f1_score_avg": df_summary["f1_score"][0],
        "f1_std": df_summary["f1_score"][1],

        "recall_avg": df_summary["recall"][0],
        "recall_std": df_summary["recall"][1],

        "loss_avg": df_summary["loss"][0],
        "loss_std": df_summary["loss"][1],
    }

    df_summary_transpose = pd.DataFrame([summary_transpose])
    
    # Save
    if save:
        fusion_type = params.get('fusion_type', None).lower()
        if fusion_type == 'sigmoid_gating':
            folder_name = f'results_(stress_test)_{feature_basis.upper()}xSigmoid-{feature_modifier.upper()}-'
            params_desired = {'k': params['k'],
                              'p': params['percentile'],
                              'nm_basis': params['normalization_basis'],
                              'nm_modifier': params['normalization_modifier']}
        elif fusion_type == 'power_gating':
            folder_name = f'results_(stress_test)_{feature_basis.upper()}xSigmoid-{feature_modifier.upper()}-'
            params_desired = {'power': params['power'],
                              'nm_basis': params['normalization_basis'],
                              'nm_modifier': params['normalization_modifier']}
        elif fusion_type in {'triangle_blocking', 
                             'diagonal_blocking',
                             'additive', 'multiplicative'}:
            folder_name = 'results_(stress_test)_comps'
            params_desired = {'type': fusion_type,
                              'nm_basis': params['normalization_basis'],
                              'nm_modifier': params['normalization_modifier']}
        else:
            raise ValueError("'Fusion Type' Error")
        
        suffix = "_".join(f"{k}-{v}" for k, v in params_desired.items())
        file_name = f"cnn_evaluation(stress_test)_{suffix}.xlsx"

        sheet_name = f'nrr_{node_retention_number}'
        
        save_to_xlsx_sheet(df_results, folder_name, file_name, sheet_name)
        
        # Save Summary
        save_to_xlsx_sheet(df_summary, folder_name, file_name, "summary")
        save_to_xlsx_sheet(df_summary_transpose, folder_name, file_name, "summary_t")    
    
    return df_results

# %% Execute
ch_index_62 = list(range(1,63))
ch_index_32 = [1,3,4,5,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,53,55,59,60,61]
ch_index_16 = [1,3,8,10,12,24,26,28,30,32,44,46,48,59,60,61]
ch_index_8 = [1,3,26,30,44,48,59,61]
ch_index_4 = [1,3,44,48]

ch_index_62 = [ch - 1 for ch in ch_index_62]
ch_index_32 = [ch - 1 for ch in ch_index_32]
ch_index_16 = [ch - 1 for ch in ch_index_16]
ch_index_8 = [ch - 1 for ch in ch_index_8]
ch_index_4 = [ch - 1 for ch in ch_index_4]

def normal_evaluation_framework():
    # for _list in [ch_index_62, ch_index_32]: # , ch_index_16, ch_index_8, ch_index_4]:
    for _list in [ch_index_62, ch_index_32, ch_index_16, ch_index_8, ch_index_4]:
        # %% baseline: original functional networks
        # cnn_subnetworks_evaluation_circle_original_cm(feature_cm="wpli", # "pcc", "plv", "pli", "wpli", "dpli"
        #                                               normalization_for_train=False, # recommended False
        #                                               valid_type="hold_one_out_validation",
        #                                               subject_range=range(6,16), experiment_range=range(1,4), 
        #                                               node_retention_list=_list, 
        #                                               save=True) # switch to True
        
        # %% competitors: additive, multiplicative, triangle_blocking, diagonal_blocking 
        # cnn_subnetworks_evaluation_circle_feature_fusion(feature_basis="plv", # always "pcc"
        #                                                  feature_modifier="pli", # "plv" or "pli"
        #                                                  params={"fusion_type": "triangle_blocking", 
        #                                                          # "additive", "multiplicative", "triangle_blocking" or "diagonal_blocking"
        #                                                          "normalization_basis": False, # "addtive": True; others: False
        #                                                          "normalization_modifier": False, # always False
        #                                                          "scale": (0, 1)},
        #                                                  normalization_for_train=False, # always False 
        #                                                  subject_range=range(6,16), experiment_range=range(1,4),
        #                                                  node_retention_list=_list, 
        #                                                  save=True) # switch to True
            
        # %% Proposed Methods: PCCxSigmoid(PLV) or PCCxSigmoid(PLI)
        params={"fusion_type": "sigmoid_gating", # always "sigmoid_gating"
                "k": "heaviside", # waiting for assignment
                "percentile": 30, # value 30 is recommended
                "normalization_basis": False, # True or False, depended on experiments
                "normalization_modifier": False} # always False
        
        cnn_subnetworks_evaluation_circle_feature_fusion(feature_basis="pcc", # always "pcc"
                                                         feature_modifier="sdpli", # "plv" or "pli"
                                                         params=params,
                                                         normalization_for_train=False, # always False
                                                         valid_type="cross_validation", # "hold_one_out_validation",
                                                         subject_range=range(6,16), experiment_range=range(1,4),
                                                         node_retention_list=_list, 
                                                         save=True) # switch to True
        
        cnn_subnetworks_evaluation_circle_feature_fusion(feature_basis="pcc", # always "pcc"
                                                         feature_modifier="dpli", # "plv" or "pli"
                                                         params=params,
                                                         normalization_for_train=False, # always False
                                                         valid_type="cross_validation", # "hold_one_out_validation",
                                                         subject_range=range(6,16), experiment_range=range(1,4),
                                                         node_retention_list=_list, 
                                                         save=True) # switch to True
        
        # # %% Mirrors: PLVxSigmoid(PCC) or PLIxSigmoid(PCC)
        # params={"fusion_type": "sigmoid_gating", # always "sigmoid_gating"
        #         "k": None, # waiting for assignment
        #         "percentile": 30, # value 30 is recommended
        #         "normalization_basis": False, # True or False, depended on experiments
        #         "normalization_modifier": False} # always False

        # params["k"] = "heaviside" # "heaviside" or values ranges of [10, 200]
        # cnn_subnetworks_evaluation_circle_feature_fusion(feature_basis="pli", # "plv" or "pli"
        #                                                  feature_modifier="plv", 
        #                                                  params=params,
        #                                                  normalization_for_train=False, # always False
        #                                                  subject_range=range(6,16), experiment_range=range(1,4),
        #                                                  node_retention_list=_list, 
        #                                                  save=True) # switch to True
        
        # ----------------------------------------------------------------------
        
# %% Execution
if __name__ == '__main__':
    normal_evaluation_framework()
    
    # end
    utils_tools.end_program_actions(play_sound=True, shutdown=False, countdown_seconds=120)