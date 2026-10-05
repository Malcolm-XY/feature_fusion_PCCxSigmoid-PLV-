# -*- coding: utf-8 -*-
"""
Created on Sat Mar  1 00:17:25 2025

@author: 18307
"""

import os

from . import utils_basic_reading
from .utils_validation import Validation, PathDefinition

# %% Read Feature Functions
def read_features(dataset, identifier, feature, 
                  feature_type="functional_connectivity", band="joint"):
    """
    dataset: str, "seed", "dreamer", "deap";
    identifier: str, "sub<number>ex<number>" for connectivity matrices;
    identifier: str, "avg_sub<number>ex<number>_sub<number>ex<number>" 
                for global averaged connectivity matrices;
    feature: str, functional connectivity or channel feature, "pcc", "plv", ......;
    """
    # Validation
    dataset = Validation.validate_dataset(dataset)
    feature = Validation.validate_feature(feature)
    band = Validation.validate_bands(band)
    identifier = Validation.validate_identifier(identifier)
    stage = Validation.validate_file_stages(feature_type)
    
    # Path
    path_read_folder = PathDefinition.retrieve_path(stage, dataset, feature)
    path_read_file = os.path.join(path_read_folder, f"{identifier}.h5")
    
    data = utils_basic_reading.load_file(path_read_file)

    return data if band == "joint" else data.get(band, {})

# %% Read Labels Functions
def read_labels(dataset, header=False):
    """
    Reads emotion labels for a specified dataset.
    
    Parameters:
    - dataset (str): The dataset name (e.g., 'SEED', 'DREAMER').
    
    Returns:
    - pd.DataFrame: DataFrame containing label data.
    
    Raises:
    - ValueError: If the dataset is not supported.
    """
    path_parent_parent = os.path.dirname(os.path.dirname(os.getcwd()))
    if dataset.lower() == 'seed':
        path_labels = os.path.join(path_parent_parent, 'Research_Data', 'SEED', 'labels', 'labels_seed.txt')
    elif dataset.lower() == 'dreamer':
        path_labels = os.path.join(path_parent_parent, 'Research_Data', 'DREAMER', 'labels', 'labels_dreamer.txt')
    else:
        raise ValueError('Currently only support SEED and DREAMER')
    return utils_basic_reading.read_txt(path_labels, header)

# %% Read Distributions
def read_distribution(dataset, mapping_method='auto', header=True):
    """
    Read the electrode distribution file for a given EEG dataset and mapping method.

    Parameters:
    dataset (str): The EEG dataset name ('SEED' or 'DREAMER').
    mapping_method (str): The mapping method ('auto' for automatic mapping, 'manual' for manual mapping).
                          Default is 'auto'.

    Returns:
    list or pandas.DataFrame:
        - The parsed electrode distribution data, depending on how `utils_basic_reading.read_txt` processes it.

    Raises:
    ValueError: If the dataset or mapping method is invalid.
    FileNotFoundError: If the distribution file does not exist.
    """
    # Define valid parameters
    valid_datasets = ['SEED', 'DREAMER']
    valid_mapping_methods = ['auto', 'manual']

    # Normalize inputs
    dataset = dataset.upper()
    mapping_method = mapping_method.lower()

    # Validate inputs
    if dataset not in valid_datasets:
        raise ValueError(f"Invalid dataset: {dataset}. Choose from {', '.join(valid_datasets)}.")
    
    if mapping_method not in valid_mapping_methods:
        raise ValueError(f"Invalid mapping method: {mapping_method}. Choose from {', '.join(valid_mapping_methods)}.")

    # Define the base path
    base_path = os.path.abspath(os.path.join(os.getcwd(), "../../Research_Data", dataset, "electrode distribution"))

    # Determine the correct file based on dataset and mapping method
    file_map = {
        ('SEED', 'auto'): "biosemi64_62_channels_original_distribution.txt",
        ('SEED', 'manual'): "biosemi64_62_channels_manual_distribution.txt",
        ('DREAMER', 'auto'): "biosemi64_14_channels_original_distribution.txt",
        ('DREAMER', 'manual'): "biosemi64_14_channels_manual_distribution.txt",
    }

    path_distr = os.path.join(base_path, file_map[(dataset, mapping_method)])

    # Check if file exists before reading
    if not os.path.exists(path_distr):
        raise FileNotFoundError(f"Distribution file not found: {path_distr}. Check dataset and mapping method.")

    # Read and return the distribution file
    distribution = utils_basic_reading.read_txt(path_distr, header)
    
    return distribution

# %% Read Channel Rankings
def read_ranking(ranking='all'):
    """
    Read electrode ranking information from a predefined Excel file.
    
    Parameters:
    ranking (str): The type of ranking to return. Options:
                  - 'label_driven_mi'
                  - 'data_driven_mi'
                  - 'data_driven_pcc' 
                  - 'data_driven_plv'
                  - 'all': returns all rankings (default)
    
    Returns:
    pandas.DataFrame or pandas.Series: The requested ranking data.
    
    Raises:
    ValueError: If an invalid ranking type is specified.
    FileNotFoundError: If the ranking file cannot be found.
    """
    import os
    
    # Valid ranking options
    valid_rankings = ['label_driven_mi', 'data_driven_mi', 'data_driven_pcc', 'data_driven_plv', 'all']
    
    # Validate input
    if ranking not in valid_rankings:
        raise ValueError(f"Invalid ranking type: '{ranking}'. Choose from {', '.join(valid_rankings)}.")
    
    # Define path
    path_current = os.getcwd()
    path_ranking = os.path.join(path_current, 'Distribution', 'electrodes_ranking.xlsx')
    
    # Check if file exists
    if not os.path.exists(path_ranking):
        raise FileNotFoundError(f"Ranking file not found at: {path_ranking}")
    
    try:
        # Read xlsx; electrodes ranking
        if ranking == 'all':
            result = utils_basic_reading.read_xlsx(path_ranking)
        else:
            result = utils_basic_reading.read_xlsx(path_ranking)[ranking]
            
        return result
    
    except KeyError:
        raise KeyError(f"Ranking type '{ranking}' not found in the Excel file.")
    except Exception as e:
        raise Exception(f"Error reading ranking data: {str(e)}")
