# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 23:44:56 2026

@author: 18307
"""

import os
import sys

#
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils import utils_basic_reading
from utils.utils_validation import Validation, PathDefinition

def read_electrode_distribution(dataset, number_channels=None):
    # Validation
    dataset = Validation.validate_dataset(dataset)

    #
    if number_channels is None:
        number_channels = Validation.DATASET_INFO[dataset]["channel numbers"]
        
    #
    path_read_files = PathDefinition.ELECTRODE_DISTRIBUTION_FILE[dataset]
    path_read_file = path_read_files[number_channels]
    electrode_distribution = utils_basic_reading.read_txt(path_read_file, header=True)
    
    # Offset
    electrode_distribution["order"] = electrode_distribution["order"] -1
    
    return electrode_distribution

# %% Usage
if __name__ == "__main__":
    dataset = "seed" # "seed", "deap", "dreamer"
    number_channels = 62 # None, 62, 32, 16, 8, 4
    
    ele_dist = read_electrode_distribution(dataset, number_channels)
    print(ele_dist)
    
    order = ele_dist["order"].tolist()
    print(order)