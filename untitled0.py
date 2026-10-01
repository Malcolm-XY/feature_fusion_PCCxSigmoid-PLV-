# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 16:20:39 2026

@author: usouu
"""

# from utils import utils_eeg_loading

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
# path_original_eeg = utils_validation.PathDefinition.retrive_path("converted_eeg")
# path_preprocessed_eeg = utils_validation.PathDefinition.retrive_path("preprocessed_eeg")
# path_decomposed_eeg = utils_validation.PathDefinition.retrive_path("decomposed")

# %% Raw dataset
from utils import utils_eeg_loading
# raw_seed_sample, _ = utils_eeg_loading.read_eeg_raw_dataset_and_parse("seed", "sub1ex1", "RawEDF") # "ndarray")
# raw_dr_sample, _ = utils_eeg_loading.read_eeg_raw_dataset_and_parse("dreamer", "sub1ex1", "RawEDF") # "ndarray")
# raw_dp_sample, _ = utils_eeg_loading.read_eeg_raw_dataset_and_parse("deap", "sub1ex1", "RawEDF") # "ndarray")

# raw_seed_sample.plot()
# raw_dr_sample.plot()
# raw_dp_sample.plot()

# %% Converted eeg
import preprocessing_decomposition
# path_save_file, path_read_file = preprocessing_decomposition.converting_and_save_circle("seed", "sub1ex1", "sub2ex1", verbose=True, save=False)
# path_save_file, path_read_file = preprocessing_decomposition.converting_and_save_circle("dreamer", "sub1ex1", "sub2ex1", verbose=True, save=False)
# path_save_file, path_read_file = preprocessing_decomposition.converting_and_save_circle("deap", "sub1ex1", "sub2ex1", verbose=True, save=False)

# %% Preprocessing eeg
# path_save_file, path_read_file = preprocessing_decomposition.preprocessing_and_save_circle("deap", "sub1ex1", "sub2ex1", verbose=True, save=False)
# path_save_file, path_read_file = preprocessing_decomposition.preprocessing_and_save_circle("seed", "sub1ex1", "sub2ex1", verbose=True, save=False)
# path_save_file, path_read_file = preprocessing_decomposition.preprocessing_and_save_circle("dreamer", "sub1ex1", "sub2ex1", verbose=True, save=False)

# %% Decomposition
path_save_file, path_read_file = preprocessing_decomposition.decomposition_and_save_circle("deap", "sub1ex1", "sub32ex1", verbose=True, save=True)


# test
# raw_seed_sample, path_0 = utils_eeg_loading.read_eeg_raw_dataset_and_parse("deap", "sub1ex1", "RawEDF")
# eeg_1, path_1 = utils_eeg_loading.read_eeg_converted("deap", "sub1ex1", "converted")
# eeg_2, path_2 = utils_eeg_loading.read_eeg_converted("deap", "sub1ex1", "preprocessed")

# raw_seed_sample.plot()
# eeg_1.plot()
# eeg_2.plot()

# %% path
# from utils.utils_validation import Validation, PathDefinition
# path_preprocessed = PathDefinition.PREPROCESSED["deap"]
# path_decomposed = PathDefinition.DECOMPOSED

# import preprocessing_decomposition
# # path_save_file, path_read_file = preprocessing_decomposition.converting_and_save_circle("deap", "sub1ex1", "sub2ex1", verbose=True, save=True)
# path_save_file, path_read_file = preprocessing_decomposition.preprocessing_and_save_circle("deap", "sub1ex1", "sub2ex1", verbose=True, save=True)

# # path_read = PathDefinition.retrieve_stage_dataset("deap", "converted", "sub1ex1")
# # from utils import utils_basic_reading
# # eeg_1 = utils_eeg_loading.read_eeg_converted("deap", "sub1ex1", "converted")
# # eeg_1.plot()

# # eeg_2 = utils_eeg_loading.read_eeg_converted("deap", "sub1ex1", "preprocessed")
# # eeg_2.plot()