# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 02:14:56 2025

@author: 18307
"""

import os

import numpy as np
import pandas as pd

import mne

from . import utils_basic_reading
from .utils_validation import Validation, PathDefinition

# %% Read Original EEG/.mat
def read_eeg_raw_dataset(dataset, identifier=None):
    """
    Read original EEG data from specified dataset.
    
    Parameters:
    dataset (str): Dataset name ('SEED' or 'DREAMER').
    identifier (str, optional): Subject/session identifier, required for SEED dataset, ignored for DREAMER.
    
    Returns:
    dict or mat object: The loaded EEG data in the specified format.
    
    Raises:
    FileNotFoundError: If the expected file does not exist.
    """
    # Validate and normalize inputs
    dataset = Validation.validate_dataset(dataset)
    identifier = Validation.validate_identifier(identifier)
    
    # 
    if identifier is not None:
        path_raw_dataset = PathDefinition.retrive_raw_dataset(dataset, 
                                                              utils_basic_reading.get_first_number(identifier),
                                                              utils_basic_reading.get_last_number(identifier))
    else: 
        path_raw_dataset = PathDefinition.retrive_raw_dataset(dataset)
    
    eeg_raw_dataset = utils_basic_reading.load_file(path_raw_dataset)
    
    return eeg_raw_dataset

def read_eeg_raw_dataset_and_parse(dataset, identifier, return_type="numpy_array"): # default to "numpy_array"
    # Validate and normalize inputs
    dataset = Validation.validate_dataset(dataset)
    identifier = Validation.validate_identifier(identifier)
    return_type = Validation.validate_file_type(return_type)
    
    # read raw dataset
    eeg_raw_dataset = read_eeg_raw_dataset(dataset, identifier)
    
    # transform
    match dataset:
        case "seed":
            eeg_parsed = np.hstack([eeg_raw_dataset[key] for key in eeg_raw_dataset])
        case "dreamer":
            eeg_list = [np.vstack(trial["EEG"]["stimuli"]) for trial in eeg_raw_dataset["DREAMER"]["Data"]]
            eeg_list_transposed = [matrix.T for matrix in eeg_list]
            eeg_dict = {i: eeg_list_transposed[i] for i in range(len(eeg_list_transposed))}
            
            # Extract specific EEG
            key = utils_basic_reading.get_first_number(identifier) - 1
            print(f"identifier: {identifier}", f"key: {key}")
            eeg_parsed = eeg_dict[key]
    
    match return_type:
        case "numpy_array":
            print()
        case "pandas_dataframe":
            eeg_parsed = pd.DataFrame(eeg_parsed)
        case "mne":
            sfreq = Validation.DATASET_INFO[dataset]["sfreq"]
            ch_names = [f"Ch{i}" for i in range(eeg_parsed.shape[0])]
            
            info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
            info["description"] = str(Validation.DATASET_INFO[dataset])
            
            eeg_parsed = mne.io.RawArray(eeg_parsed, info)
        
    return eeg_parsed

# %% Read Filtered EEG/.fif
def read_eeg_filtered(dataset, identifier, freq_band='joint', object_type='pandas_dataframe'):
    """
    Read filtered EEG data for the specified experiment and frequency band.

    Parameters:
    dataset (str): Dataset name (e.g., 'SEED', 'DREAMER').
    identifier (str): Identifier for the subject/session.
    freq_band (str): Frequency band to load ("alpha", "beta", "gamma", "delta", "theta", or "joint").
                     Default is "joint", which loads all bands.
    object_type (str): Desired output format: 'pandas_dataframe', 'numpy_array', or 'mne'.

    Returns:
    mne.io.Raw | dict | pandas.DataFrame | numpy.ndarray:
        - If 'mne', returns the MNE Raw object (or a dictionary of them for 'joint').
        - If 'pandas_dataframe', returns a DataFrame with EEG data.
        - If 'numpy_array', returns a NumPy array with EEG data.

    Raises:
    ValueError: If the specified frequency band is not valid.
    FileNotFoundError: If the expected file does not exist.
    """
    # Valide and normalize inputs
    dataset = Validation.validate_dataset(dataset)
    identifier = Validation.validate_identifier(identifier)
    freq_band = Validation.validate_bands(freq_band)
    object_type = Validation.validate_file_type(object_type)
    
    # Construct base path
    path_parent_parent = os.path.dirname(os.path.dirname(os.getcwd()))
    base_path = os.path.join(path_parent_parent, 'Research_Data', dataset, 'original eeg', 'Filtered_EEG')
    
    # Function to process a single frequency band
    def process_band(band):
        file_path = os.path.join(base_path, f'{identifier}_{band.capitalize()}_eeg.fif')
        try:
            raw_data = mne.io.read_raw_fif(file_path, preload=True)
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}. Check the path and file existence.")
            
        if object_type == 'pandas_dataframe':
            return pd.DataFrame(raw_data.get_data(), index=raw_data.ch_names)
        elif object_type == 'numpy_array':
            return raw_data.get_data()
        else:  # Default to MNE Raw object / .fif object
            return raw_data
    
    # Handle joint vs. single band request
    if freq_band == 'joint':
        result = {}
        for band in ['alpha', 'beta', 'gamma', 'delta', 'theta']:
            result[band] = process_band(band)
        return result
    else:
        return process_band(freq_band)

# %% Example Usage
if __name__ == '__main__':
    # EEG from original dataset
    eeg_dreamer = read_eeg_raw_dataset(dataset="dreamer", identifier=None)
    eeg_dreamer_ = read_eeg_raw_dataset_and_parse("dreamer", "sub1ex1")
    eeg_seed_sample = read_eeg_raw_dataset(dataset='seed', identifier='sub1ex1')
    eeg_seed_sample_ = read_eeg_raw_dataset_and_parse("seed", "sub1ex1")
    
    # Filtered EEG
    # filtered_eeg_dreamer_sample1 = read_eeg_filtered(dataset='dreamer', identifier='sub1', freq_band='alpha')
    # filtered_eeg_dreamer_sample2 = read_eeg_filtered(dataset='dreamer', identifier='sub1', freq_band='beta')
    # filtered_eeg_seed_sample1 = read_eeg_filtered(dataset='seed', identifier='sub1ex1', freq_band='alpha')
    # filtered_eeg_seed_sample2 = read_eeg_filtered(dataset='seed', identifier='sub1ex2', freq_band='beta')