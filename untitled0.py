# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 16:20:39 2026

@author: usouu
"""

from utils import utils_eeg_loading

# eeg_seed_sample = utils_eeg_loading.read_eeg_original_dataset(dataset='seed', identifier='sub1ex1')

# eeg_dr_sample = utils_eeg_loading.read_eeg_original_dataset(dataset='dreamer', identifier=None)
eeg_dreamer_1 = utils_eeg_loading.read_and_parse_dreamer("sub1")
eeg_dreamer_23 = utils_eeg_loading.read_and_parse_dreamer("sub23")