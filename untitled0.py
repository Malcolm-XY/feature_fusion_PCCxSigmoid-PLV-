# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 16:20:39 2026

@author: usouu
"""

# %% Validation
print("Validation")

# from utils import utils_validation

# utils_validation.Validation.report()
# utils_validation.PathDefinition.report()

# path_dataset = utils_validation.PathDefinition.retrive_path("dataset")
# path_original_eeg = utils_validation.PathDefinition.retrive_path("converted_eeg")
# path_preprocessed_eeg = utils_validation.PathDefinition.retrive_path("preprocessed_eeg")
# path_decomposed_eeg = utils_validation.PathDefinition.retrive_path("decomposed")

# %% Raw dataset
print("Raw dataset demonstration")

# from utils import utils_eeg_loading

# raw_seed_sample, _ = utils_eeg_loading.read_eeg_raw_dataset_and_parse(
#     "seed", "sub1ex1", "RawEDF") # "ndarray")

# raw_dr_sample, _ = utils_eeg_loading.read_eeg_raw_dataset_and_parse(
#     "dreamer", "sub1ex1", "RawEDF") # "ndarray")

# raw_dp_sample, _ = utils_eeg_loading.read_eeg_raw_dataset_and_parse(
#     "deap", "sub1ex1", "RawEDF") # "ndarray")

# raw_seed_sample.plot()
# raw_dr_sample.plot()
# raw_dp_sample.plot()

# %% Converting eeg
print("Convert EEG")

# from preprocessing_decomposition import converting_and_save_circle

# path_save_file, path_read_file = converting_and_save_circle(
#     "seed", "sub1ex1", "sub2ex1", verbose=True, save=False)

# path_save_file, path_read_file = converting_and_save_circle(
#     "dreamer", "sub1ex1", "sub2ex1", verbose=True, save=False)

# path_save_file, path_read_file = converting_and_save_circle(
#     "deap", "sub1ex1", "sub2ex1", verbose=True, save=False)

# %% Preprocessing eeg
print("Preprocess EEG")

# from preprocessing_decomposition import preprocessing_and_save_circle

# path_save_file, path_read_file = preprocessing_and_save_circle(
#     "deap", "sub1ex1", "sub2ex1", verbose=True, save=False)

# path_save_file, path_read_file = preprocessing_and_save_circle(
#     "seed", "sub1ex1", "sub2ex1", verbose=True, save=False)

# path_save_file, path_read_file = preprocessing_and_save_circle(
#     "dreamer", "sub1ex1", "sub2ex1", verbose=True, save=False)

# %% Decomposition
print("Decomposition")

# from preprocessing_decomposition import decomposition_and_save_circle

# path_save_file, path_read_file = decomposition_and_save_circle(
#     "deap", "sub1ex1", "sub2ex1", verbose=True, save=False)

# path_save_file, path_read_file = decomposition_and_save_circle(
#     "seed", "sub1ex1", "sub2ex3", verbose=True, save=False)

# path_save_file, path_read_file = decomposition_and_save_circle(
#     "dreamer", "sub1ex1", "sub2ex1", verbose=True, save=False)

# %% Reading; Correspondance check
print("Correspondance check")

from utils import utils_eeg_loading

eeg_raw_sample, path_0 = utils_eeg_loading.read_eeg_raw_dataset_and_parse(
    "seed", "sub1ex1", "RawEDF")

eeg_converted_sample, path_1 = utils_eeg_loading.read_eeg_converted(
    "seed", "sub1ex1", "converted")

eeg_preprocessed_sample, path_2 = utils_eeg_loading.read_eeg_converted(
    "seed", "sub1ex1", "preprocessed")

eeg_decomposed_sample, path_3 = utils_eeg_loading.read_eeg_decomposed(
    "seed", "sub1ex1", return_type="RawEDF")
eeg_decomposed_sample_a = eeg_decomposed_sample["alpha"]
eeg_decomposed_sample_b = eeg_decomposed_sample["beta"]
eeg_decomposed_sample_g = eeg_decomposed_sample["gamma"]

eeg_raw_sample.plot()
eeg_converted_sample.plot()
eeg_preprocessed_sample.plot()

eeg_decomposed_sample_a.plot()
eeg_decomposed_sample_b.plot()
eeg_decomposed_sample_g.plot()