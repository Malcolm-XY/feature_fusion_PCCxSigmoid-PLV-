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
    
    return eeg_raw_dataset, path_raw_dataset

def read_eeg_raw_dataset_and_parse(dataset, identifier, return_type="ndarray"): # default to "ndarray"
    # Validate and normalize inputs
    dataset = Validation.validate_dataset(dataset)
    identifier = Validation.validate_identifier(identifier)
    return_type = Validation.validate_file_type(return_type)
    
    # read raw dataset
    eeg_raw_dataset, path_raw_dataset = read_eeg_raw_dataset(dataset, identifier)
    
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
        case "deap":
            eeg_parsed = eeg_raw_dataset["data"]
            eeg_parsed = np.concatenate(eeg_parsed, axis=1)[0:32,:]
            
    # Convert voltage values to the recommended unit: volts (V)
    def convert_unit(eeg_parsed):
        unit = Validation.DATASET_INFO[dataset]["source_unit"]
        unit_scale = {"V": 1.0, "mV": 1e-3, "uV": 1e-6,}
        eeg_parsed = eeg_parsed * unit_scale[unit]
        return eeg_parsed
    
    # Convert variable type for returning
    match return_type:
        case "RawEDF":
            if isinstance(eeg_parsed, mne.io.BaseRaw):
                pass
    
            elif isinstance(eeg_parsed, (np.ndarray, pd.DataFrame)):
                if isinstance(eeg_parsed, pd.DataFrame):
                    eeg_parsed = eeg_parsed.to_numpy()
                
                # sampling rate
                sfreq = Validation.DATASET_INFO[dataset]["sfreq"]
                
                # channel names
                try:
                    path_ch_names = PathDefinition.ELECTRODE_DISTRIBUTION_FILE[dataset]
                    ch_names = utils_basic_reading.read_txt(path_ch_names, header=True)["channel"].tolist()
                
                    if len(ch_names) != eeg_parsed.shape[0]:
                        raise ValueError("Channel-name count does not match EEG channel count.")
                
                except (KeyError, FileNotFoundError, ValueError, TypeError) as e:
                    print(f"Warning: {e}. Using default channel names.")
                    ch_names = [f"Ch{i}" for i in range(eeg_parsed.shape[0])]


                info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg",)
                info["description"] = str(Validation.DATASET_INFO[dataset])
                
                eeg_parsed = convert_unit(eeg_parsed)
                
                eeg_parsed = mne.io.RawArray(eeg_parsed, info)
    
            else:
                raise TypeError(f"Cannot convert {type(eeg_parsed).__name__} to RawEDF.")
    
        case "ndarray":
            if isinstance(eeg_parsed, np.ndarray):
                eeg_parsed = convert_unit(eeg_parsed)
                pass

            elif isinstance(eeg_parsed, mne.io.BaseRaw):
                eeg_parsed = eeg_parsed.get_data()
                eeg_parsed = convert_unit(eeg_parsed)
                
            elif isinstance(eeg_parsed, pd.DataFrame):
                eeg_parsed = eeg_parsed.to_numpy()
                eeg_parsed = convert_unit(eeg_parsed)
            else:
                raise TypeError(f"Cannot convert {type(eeg_parsed).__name__} to ndarray.")
    
        case "DataFrame":
            if isinstance(eeg_parsed, pd.DataFrame):
                eeg_parsed = convert_unit(eeg_parsed)
                pass
    
            elif isinstance(eeg_parsed, mne.io.BaseRaw):
                eeg_parsed = eeg_parsed.get_data()
                eeg_parsed = pd.DataFrame(convert_unit(eeg_parsed))
    
            elif isinstance(eeg_parsed, np.ndarray):
                eeg_parsed = convert_unit(eeg_parsed)
                eeg_parsed = pd.DataFrame(eeg_parsed)
    
            else:
                raise TypeError(f"Cannot convert {type(eeg_parsed).__name__} to DataFrame.")
    
        case _:
            raise ValueError(f"Unsupported return_type: {return_type!r}. "
                             "Expected 'RawEDF', 'ndarray', or 'DataFrame'.")

    return eeg_parsed, path_raw_dataset

# %% Read Converted EEG/fif/gz
def read_eeg_converted(dataset, identifier, file_stage, verbose=False):
    # Valide and normalize inputs
    dataset = Validation.validate_dataset(dataset)
    identifier = Validation.validate_identifier(identifier)
    file_stage = Validation.validate_file_stages(file_stage)
    
    # 
    path_file = PathDefinition.retrieve_stage_dataset(dataset, file_stage, identifier)
    try:
        raw_data = mne.io.read_raw_fif(path_file, preload=True, verbose=verbose)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {path_file}. Check the path and file existence.")
    
    return raw_data, path_file

# %% Read Decomposed EEG/fif/gz
# revise here
def read_eeg_decomposed(dataset, identifier, band="joint", verbose=False):
    # Valide and normalize inputs
    dataset = Validation.validate_dataset(dataset)
    identifier = Validation.validate_identifier(identifier)
    band = Validation.validate_bands(band)
    
    def retrieve_band(identifier, band, object_type="DataFrame"):
        _identifier = "_".join([identifier, band])
        path_file = PathDefinition.retrieve_stage_dataset(dataset, "decomposed", _identifier)
        try:
            raw_data = mne.io.read_raw_fif(path_file, preload=True, verbose=verbose)
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {path_file}. Check the path and file existence.")
        
        match object_type:
            case "RawEDF":
                eeg_data = raw_data.copy()
            case "ndarray":
                eeg_data = raw_data.get_data().copy()
            case "DataFrame":
                eeg_data = pd.DataFrame(raw_data.get_data(), index=raw_data.ch_names).copy()
                
        return eeg_data, path_file
        
    if band != "joint":
        raw_data, path_file = retrieve_band(identifier, band)
        return raw_data, path_file
    
    elif band == "joint":
        dict_eeg_decomposed = {}
        for _band in ["theta", "delta", "alpha", "beta", "gamma"]:
            raw_data, path_file = retrieve_band(identifier, _band)            
            dict_eeg_decomposed.update({_band: raw_data})
                
        return dict_eeg_decomposed, path_file
    


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
    # EEG from raw dataset
    raw_dreamer_sample = read_eeg_raw_dataset(dataset="dreamer", identifier=None)
    raw_dreamer_sample_ = read_eeg_raw_dataset_and_parse("dreamer", "sub1ex1")
    
    raw_seed_sample = read_eeg_raw_dataset(dataset='seed', identifier='sub1ex1')
    raw_seed_sample_ = read_eeg_raw_dataset_and_parse("seed", "sub1ex1")

    # Converted EEG
    raw_converted_seed_sample = read_eeg_converted("seed", "sub1ex1", "converted")
    raw_preprocessed_seed_sample = read_eeg_converted("seed", "sub1ex1", "preprocessed")