# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 12:54:07 2026

@author: 18307
"""
import numpy as np
import pandas as pd

import torch

import cnn_validation
from models import models

from utils import utils_interaction
from utils import utils_basic_reading
from utils.utils_feature_loading import read_features, read_distribution

from utils.utils_translation import rectify_diagonal
from utils.utils_validation import Validation, PathDefinition

from utils_advanced import label_translation as lt
from utils_advanced import electrode_distribution_translation as edt

# from tool_read_params_save_xlsx import read_params
from tool_read_params_save_xlsx import save_to_xlsx_sheet
# from tool_read_params_save_xlsx import save_to_xlsx_fitting

# test
feature_sd = read_features("seed", "sub1ex1", "pcc")
feature_dp = read_features("deap", "sub1ex1", "pcc")
feature_dr = read_features("dreamer", "sub1ex1", "pcc")

# %% cnn subnetworks evaluation circle common
def cnn_subnetworks_evaluation_circle_original_cm(feature="pcc", dataset="seed", dimension="valence",
                                                  # =================================================
                                                  valid_type="cross_validation", # "hold_one_out_validation"
                                                  random_seed=42, stratified=False, shuffle=False,
                                                  # =================================================
                                                  identifier_1 = "sub1ex1", identifier_2 = "sub1ex3",
                                                  node_retention=None, diagonal_rectify=False,
                                                  save=False):
    # Validation
    dataset = Validation.validate_dataset(dataset)
    feature = Validation.validate_feature(feature)
    identifier_1 = Validation.validate_identifier(identifier_1)
    identifier_2 = Validation.validate_identifier(identifier_2)
    
    subject_range = range(utils_basic_reading.get_first_number(identifier_1), 
                          utils_basic_reading.get_first_number(identifier_2) + 1)
    experiment_range = range(utils_basic_reading.get_last_number(identifier_1),
                             utils_basic_reading.get_last_number(identifier_2) + 1)
    
    # Node retention list
    electrode_dist = edt.read_electrode_distribution(dataset, node_retention)
    node_retention_list = electrode_dist["order"].tolist()
    
    # For traning and testing in CNN; data and evaluation circle
    all_results_list = []
    for subject in subject_range:
        for experiment in experiment_range:
            _identifier = f"sub{subject}ex{experiment}"
            print(f"Evaluating {_identifier}...")
            
            # Labels/y
            labels = lt.read_label_and_translation(dataset, _identifier, "ternary")
            labels = np.array(labels[dimension])
            y = torch.tensor(labels).view(-1)
            
            # Features/functional networks/connectivity matrices/x
            features = read_features(dataset, _identifier, feature)
            alpha = features["alpha"]
            beta = features["beta"]
            gamma = features["gamma"]

            # Features/functional subnetworks/x_retained
            alpha_retained = alpha[:,node_retention_list,:][:,:,node_retention_list]
            beta_retained = beta[:,node_retention_list,:][:,:,node_retention_list]
            gamma_retained = gamma[:,node_retention_list,:][:,:,node_retention_list]
            
            x_retained = np.stack((alpha_retained, beta_retained, gamma_retained), axis=1)
            
            # Diagonal rectify
            if diagonal_rectify:
                x_retained = rectify_diagonal(x_retained)
            
            # CNN model
            cnn_model = models.LeNet5Modern(num_classes=3)
            
            # Traning and testing
            if valid_type == "cross_validation":
                result_CM = cnn_validation.cnn_cross_validation(cnn_model, 
                                                                x_retained, y, 
                                                                folds=5, seed=random_seed, 
                                                                stratified=stratified, 
                                                                stratified_shuffle=shuffle)
                
                result_CM = cnn_validation.cnn_cross_validation(cnn_model, x_retained, y)
                
            elif valid_type == "hold_one_out_validation":
                result_CM = cnn_validation.cnn_validation(cnn_model, 
                                                          x_retained, y, 
                                                          0.6, random_seed=random_seed, 
                                                          stratified=stratified, 
                                                          stratified_shuffle=shuffle)
            else:
                raise NotImplementedError("valid type must be defined.")
            
            # Flatten result and add identifier
            result_flat = {"Identifier": _identifier, **result_CM}
            all_results_list.append(result_flat)
            
    # Convert list of dicts to DataFrame
    df_results = pd.DataFrame(all_results_list)
    
    # Compute mean of all numeric columns (excluding Identifier)
    mean_row = df_results.select_dtypes(include=[np.number]).mean().to_dict()
    mean_row["Identifier"] = "Average"
    
    # Std
    std_row = df_results.select_dtypes(include=[np.number]).std(ddof=0).to_dict()
    std_row["Identifier"] = "Std"
    
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
        folder_name = "results_(stress_test)_original"
        val = "cross" if valid_type == "cross_validation" else "hold"
        file_name = f"dleval(stress_test)_f[{feature}]_val[{val}]_seed[{random_seed}]_origin.xlsx"
        sheet_name = f"nrn_{node_retention}"
        
        save_to_xlsx_sheet(df_results, folder_name, file_name, sheet_name)
        
        # Save Summary
        save_to_xlsx_sheet(df_summary, folder_name, file_name, "summary")
        save_to_xlsx_sheet(df_summary_transpose, folder_name, file_name, "summary_t")  
        
    return df_results

# %% Usage
if __name__ == "__main__":
    for _node_retention in [8]: # [62, 32, 16, 8, 4]:
        cnn_subnetworks_evaluation_circle_original_cm(feature="pli", dataset="seed", dimension="valence",
                                                      # =================================================
                                                      valid_type="cross_validation", # "hold_one_out_validation"
                                                      random_seed=42, stratified=False, shuffle=False,
                                                      # =================================================
                                                      identifier_1 = "sub6ex1", identifier_2 = "sub15ex3",
                                                      node_retention=_node_retention, diagonal_rectify=False,
                                                      save=True)
    
    utils_interaction.end_program_actions(play_sound=True, shutdown=False, countdown_seconds=120)
    