# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 14:01:53 2026

@author: 18307
"""
from utils import utils_eeg_loading
from utils import utils_interaction

# %% Validation
from utils.utils_validation import Validation, PathDefinition

print("Validation")
Validation.report()
PathDefinition.report()

# Usage
path_dataset = PathDefinition.retrieve_path("dataset")
path_original_eeg = PathDefinition.retrieve_path("eeg_converted")
path_preprocessed_eeg = PathDefinition.retrieve_path("eeg_preprocessed")
path_decomposed_eeg = PathDefinition.retrieve_path("eeg_decomposed")

# %% Converting; preprocessing; decomposition
# from utils_advanced.preprocessing_decomposition import (
#     converting_and_save_circle, 
#     preprocessing_and_save_circle, 
#     decomposition_and_save_circle)

# # Raw eeg
# print("Raw EEG")
# raw_dataset_sample, _ = utils_eeg_loading.read_eeg_raw_dataset_and_parse(
#     "seed", "sub1ex1", "RawEDF") # "ndarray")

# # Converting eeg
# print("Converting")
# path_save_converted_sample, _ = converting_and_save_circle(
#     "seed", "sub1ex1", "sub1ex1", verbose=True, save=False)

# # Preprocessing eeg
# print("Preprocessing")
# path_save_preprocessed_sample, _ = preprocessing_and_save_circle(
#     "seed", "sub1ex1", "sub1ex1", verbose=True, save=False)

# # Decomposition
# print("Decomposition")
# path_save_decomposed_file, _ = decomposition_and_save_circle(
#     "seed", "sub1ex1", "sub1ex1", verbose=True, save=False)

# # Reading, correspondance check by visualization
# # Raw dataset (.mat, ......)->Converted EEG (RawEDF)->Preprocessed EEG (RawEDF)->Decomposed EEG (RawEDF)
# print("Correspondance check")

# eeg_raw_sample, path_0 = utils_eeg_loading.read_eeg_raw_dataset_and_parse(
#     "seed", "sub1ex1", "RawEDF")

# eeg_converted_sample, path_1 = utils_eeg_loading.read_eeg_converted(
#     "seed", "sub1ex1", "eeg_converted")

# eeg_preprocessed_sample, path_2 = utils_eeg_loading.read_eeg_converted(
#     "seed", "sub1ex1", "eeg_preprocessed")

# eeg_decomposed_sample, path_3 = utils_eeg_loading.read_eeg_decomposed(
#     "seed", "sub1ex1", return_type="RawEDF")
# eeg_decomposed_sample_a = eeg_decomposed_sample["alpha"]
# eeg_decomposed_sample_b = eeg_decomposed_sample["beta"]
# eeg_decomposed_sample_g = eeg_decomposed_sample["gamma"]

# eeg_raw_sample.plot()
# eeg_converted_sample.plot()
# eeg_preprocessed_sample.plot()

# eeg_decomposed_sample_a.plot()
# eeg_decomposed_sample_b.plot()
# eeg_decomposed_sample_g.plot()

# %% Feature Engineering
from utils_advanced.feature_engineering import (
    compute_fc_matrices_batch, 
    compute_average_fc_matrix)

# Feature engineering
compute_fc_matrices_batch("dreamer", "sub1ex1", "sub23ex1", feature="pli", band="joint", save=True, verbose=True)
compute_fc_matrices_batch("dreamer", "sub1ex1", "sub23ex1", feature="wpli", band="joint", save=True, verbose=True)
compute_fc_matrices_batch("dreamer", "sub1ex1", "sub23ex1", feature="dpli", band="joint", save=True, verbose=True)
compute_fc_matrices_batch("dreamer", "sub1ex1", "sub23ex1", feature="sdpli", band="joint", save=True, verbose=True)

# Average connectivity matrices
# compute_average_fc_matrix("seed", "sub1ex1", "sub1ex3", "pli", band="joint", save=True, verbose=True)

# End program
utils_interaction.end_program_actions(play_sound=True, shutdown=True, countdown_seconds=30)