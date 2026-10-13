# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 20:30:23 2026

@author: 18307
"""

import os
import sys
import time
import h5py

import numpy as np
import pandas as pd

from scipy.signal import hilbert

#
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils.utils_validation import Validation, PathDefinition
from utils import (
    utils_basic_reading,
    utils_feature_loading,
    utils_interaction,
    utils_eeg_loading,
)

def labels_encoding(labels, obj_dimension="ternary", bound="ceil"):
    is_df = isinstance(labels, pd.DataFrame)
    values = np.asarray(labels, dtype=float)

    if obj_dimension not in ("binary", "ternary", "continuous"):
        raise ValueError(f"Invalid obj_dimension: {obj_dimension}")

    if bound not in ("ceil", "floor"):
        raise ValueError(f"Invalid bound: {bound}")

    # Infer rating scale
    scale_max = 9 if np.max(values) > 5 else 5
    midpoint = (1 + scale_max) / 2

    if obj_dimension == "continuous":
        result = (values - 1) / (scale_max - 1)

    elif obj_dimension == "binary":
        if bound == "ceil":
            result = (values >= midpoint).astype(int)
        else:
            result = (values > midpoint).astype(int)

    elif obj_dimension == "ternary":
        lower = midpoint - 1
        upper = midpoint + 1

        if bound == "ceil":
            result = np.where(
                values < lower, -1,
                np.where(values < upper, 0, 1)
            ).astype(int)
        else:
            result = np.where(
                values <= lower, -1,
                np.where(values <= upper, 0, 1)
            ).astype(int)

    if is_df:
        return pd.DataFrame(
            result,
            index=labels.index,
            columns=labels.columns
        )

    return result

def labels_upsampling(labels, durations: int | list):
    is_df = isinstance(labels, pd.DataFrame)
    values = np.asarray(labels)

    if isinstance(durations, (int, np.integer)):
        durations = np.full(len(values), durations)
    else:
        durations = np.asarray(durations)

    if durations.ndim != 1 or len(durations) != len(values):
        raise ValueError("durations must have the same length as labels.")

    if (
        not np.issubdtype(durations.dtype, np.number)
        or not np.all(np.isfinite(durations))
        or np.any(durations < 0)
        or not np.all(durations == np.floor(durations))
    ):
        raise ValueError("durations must contain non-negative integers.")

    durations = durations.astype(int)
    result = np.repeat(values, durations, axis=0)

    if is_df:
        result = pd.DataFrame(
            result,
            index=labels.index.repeat(durations),
            columns=labels.columns
        )

        result["trial"] = np.repeat(
            np.arange(len(labels)),
            durations
        )

    return result

def labels_encoding_save_batch(dataset, identifier_1, identifier_2):
    dataset = Validation.validate_dataset(dataset)
    identifier_1 = Validation.validate_identifier(identifier_1)
    identifier_2 = Validation.validate_identifier(identifier_2)
    
    subject_range = range(utils_basic_reading.get_first_number(identifier_1), 
                          utils_basic_reading.get_first_number(identifier_2) + 1)
    experiment_range = range(utils_basic_reading.get_last_number(identifier_1),
                             utils_basic_reading.get_last_number(identifier_2) + 1)
    
    if dataset == "seed":
        print("No Labels Encoding Requirement for SEED")
        return None
    
    elif dataset == "deap":
        for subject in subject_range:
            for experiment in experiment_range:
                _identifier = f"sub{subject}ex{experiment}"
                print(f"Processing: {_identifier}.")
                
                # read dataset                
                _, labels_dp, _ = utils_eeg_loading.read_eeg_raw_dataset_and_parse("deap", _identifier)
                
                # extract scores
                valence_dp, arousal_dp = labels_dp[:,0], labels_dp[:,1]
                dominance_dp, liking_dp = labels_dp[:,2], labels_dp[:,3]
                
                # extract duration information
                freq = Validation.SAMPLING_RATES["deap"]
                data_dp_ori, _ = utils_eeg_loading.read_eeg_raw_dataset("deap", _identifier)
                durations = [data_dp_ori["data"].shape[2]/freq] * data_dp_ori["data"].shape[0]
                
                # dataframe for saving
                labels_df = {
                    "valence": valence_dp, 
                    "arousal": arousal_dp, 
                    "dominance": dominance_dp, 
                    "liking": liking_dp,
                    "duration": durations
                    }
                
                labels_df = pd.DataFrame(labels_df)
                
                # retrieve path for saving
                path_labels_folder = PathDefinition.retrieve_path("labels", "deap")
                
                # save
                path_save_file = os.path.join(path_labels_folder, f"{_identifier}.txt")
                labels_df.to_csv(path_save_file, sep='\t', index=False)
                
    elif dataset == "dreamer":
        print("test")
        for subject in subject_range:
            for experiment in experiment_range:
                _identifier = f"sub{subject}ex{experiment}"
                print(f"Processing: {_identifier}.")
                
                # read dataset
                _, labels_dr, _ = utils_eeg_loading.read_eeg_raw_dataset_and_parse("dreamer", _identifier)
                
                # extract scores
                valence_dr = labels_dr["valence"][:,0]
                arousal_dr = labels_dr["arousal"][:,0]
                dominance_dr = labels_dr["dominance"][:,0]
                
                # extract duration information
                freq = Validation.SAMPLING_RATES["dreamer"]
                data_dr_ori, _ = utils_eeg_loading.read_eeg_raw_dataset("dreamer", _identifier)
                
                subj_index = 0
                eeg_stimulis = data_dr_ori["DREAMER"]["Data"][subj_index]["EEG"]["stimuli"]

                durations = []
                for _index, _trial in enumerate(eeg_stimulis):
                    _duration = len(_trial)/freq
                    
                    durations.append(_duration)
                
                # dataframe for saving
                labels_df = {
                    "valence": valence_dr,
                    "arousal": arousal_dr,
                    "dominance": dominance_dr,
                    "duration": durations
                    }
                
                labels_df = pd.DataFrame(labels_df)
                
                # retrieve path for saving
                path_labels_folder = PathDefinition.retrieve_path("labels", "dreamer")
                
                # save
                path_save_file = os.path.join(path_labels_folder, f"{_identifier}.txt")
                labels_df.to_csv(path_save_file, sep='\t', index=False)

def read_label_and_translation(dataset, identifier="sub1ex1", obj_dimension="ternary", bound="ceil"):
    dataset = Validation.validate_dataset(dataset)
    identifier = Validation.validate_identifier(identifier)
    
    # path
    path_read_folder = PathDefinition.retrieve_path("labels", dataset)
    if dataset == "seed":
        path_read_file = os.path.join(path_read_folder, "labels_seed.txt")
        # read
        labels_df = utils_basic_reading.read_txt(path_read_file, header=True)
        
        return labels_df
    else:
        path_read_file = os.path.join(path_read_folder, f"{identifier}.txt")
    
    # read
    labels_df = utils_basic_reading.read_txt(path_read_file, header=True)
    
    # translation
    durations = labels_df["duration"]
    labels_encoded = labels_encoding(labels_df.copy().drop(columns="duration"), obj_dimension, bound)
    # labels_encoded = labels_encoding(labels_df, obj_dimension, bound)
    labels_translated = labels_upsampling(labels_encoded, durations)
    
    return labels_translated
    
# %% Usage
if __name__ == "__main__":
    # labels_encoding_save_batch("dreamer", "sub1ex1", "sub23ex1")
    
    labels_sd = read_label_and_translation("seed", "sub1ex1")
    labels_dp = read_label_and_translation("deap", "sub1ex1")
    labels_dr = read_label_and_translation("dreamer", "sub1ex1")
    