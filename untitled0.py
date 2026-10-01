# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 16:20:39 2026

@author: usouu
"""

from utils import utils_eeg_loading

# eeg_seed_sample = utils_eeg_loading.read_eeg_original_dataset(dataset='seed', identifier='sub1ex1')

# eeg_dr_sample = utils_eeg_loading.read_eeg_original_dataset(dataset='dreamer', identifier=None)
# eeg_dreamer_1 = utils_eeg_loading.read_and_parse_dreamer("sub1")
# eeg_dreamer_23 = utils_eeg_loading.read_and_parse_dreamer("sub23")

#eeg_seed_sample = utils_eeg_loading.read_and_parse_seed("sub1ex1")
#eeg_dreamer = utils_eeg_loading.read_and_parse_dreamer("sub1ex1")

# %% EEG filtering
# from feature_engineering import filter_eeg_and_save_batch
# filter_eeg_and_save_batch("dreamer", range(1,2), range(1,2), verbose=True, save=False)

# %% Validation
# from utils import utils_validation

# utils_validation.Validation.report()
# utils_validation.PathDefinition.report()

# path_dataset = utils_validation.PathDefinition.retrive_path("dataset")
# path_original_eeg = utils_validation.PathDefinition.retrive_path("original_eeg")
# path_preprocessed_eeg = utils_validation.PathDefinition.retrive_path("preprocessed_eeg")
# path_decomposed_eeg = utils_validation.PathDefinition.retrive_path("decomposed")

# raw dataset
# test_seed_1, _ = utils_eeg_loading.read_eeg_raw_dataset_and_parse("seed", "sub6ex3", "RawEDF") # "ndarray")
# test_dr_1, _ = utils_eeg_loading.read_eeg_raw_dataset_and_parse("dreamer", "sub1ex1", "RawEDF") # "ndarray")
# test_dp_1, _ = utils_eeg_loading.read_eeg_raw_dataset_and_parse("deap", "sub1ex1", "RawEDF") # "ndarray")

# test_seed_1.plot()
# test_dr_1.plot()
# test_dp_1.plot()

# path
from utils.utils_validation import Validation, PathDefinition
path_preprocessed = PathDefinition.PREPROCESSED["deap"]
path_decomposed = PathDefinition.DECOMPOSED

import preprocessing_decomposition
path_save_fold, path_raw_dataset = preprocessing_decomposition.converting_and_save_circle("deap", "sub1ex1", "sub32ex1", verbose=True, save=True)