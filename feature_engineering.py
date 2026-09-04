# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 23:15:11 2025

@author: 18307
"""

import os
import time
import h5py

import numpy as np
import pandas as pd

import mne
from scipy.signal import hilbert

from utils import utils_feature_loading, utils_visualization, utils_eeg_loading, utils_tools

# %% Filter EEG
def filter_eeg(eeg, sampling_rate:int, verbose:bool=False):
    """
    Filter raw EEG data into standard frequency bands using MNE.

    Parameters:
    eeg (numpy.ndarray): Raw EEG data array with shape (n_channels, n_samples).
    freq (int): Sampling frequency of the EEG data. Default is 128 Hz.
    verbose (bool): If True, prints progress messages. Default is False.

    Returns:
    dict:
        A dictionary where keys are frequency band names ("Delta", "Theta", "Alpha", "Beta", "Gamma")
        and values are the corresponding MNE Raw objects filtered to that band.
    """
    # Create MNE info structure and Raw object from the EEG array
    info = mne.create_info(ch_names=[f"Ch{i}" for i in range(eeg.shape[0])], sfreq=sampling_rate, ch_types='eeg')
    mne_eeg = mne.io.RawArray(eeg, info)
    
    # Define frequency bands
    freq_bands = {
        "Delta": (0.5, 4),
        "Theta": (4, 8),
        "Alpha": (8, 13),
        "Beta": (13, 30),
        "Gamma": (30, 50),
    }
    
    band_filtered_eeg = {}
    
    # Filter EEG data for each frequency band
    for band, (low_freq, high_freq) in freq_bands.items():
        filtered_eeg = mne_eeg.copy().filter(l_freq=low_freq, h_freq=high_freq, method="fir", phase="zero-double")
        band_filtered_eeg[band] = filtered_eeg
        if verbose:
            print(f"{band} band filtered: {low_freq}–{high_freq} Hz")
    
    return band_filtered_eeg

class validation:
    names_dataset = ("SEED", "DEAP", "DREAMER")
    names_feature = ("pcc", "plv", "mi", "pli", "wpli", "dpli", "sdpli")
    bands = ("joint", "theta", "delta", "alpha", "beta", "gamma")
    sampling_rates = {"SEED": 200, "DEAP": 128, "DREAMER": 128}
    
def filter_eeg_4_dataset(dataset:str, identifier:str, 
                         verbose:bool=True, save:bool=False):
    """
    Load, filter, and optionally save SEED dataset EEG data into frequency bands.

    Parameters:
    identifier (str): Identifier for the subject/session.
    freq (int): SEED: 200 Hz. DREAMER: 128 Hz. DEAP: 128 Hz.
    verbose (bool): If True, prints progress messages. Default is True.
    save (bool): If True, saves the filtered EEG data to disk. Default is False.

    Returns:
    dict:
        A dictionary where keys are frequency band names and values are the filtered MNE Raw objects.

    Raises:
    FileNotFoundError: If the SEED data file cannot be found.
    """
    # Validation
    validation_ = validation
    
    # Normalize parameters
    dataset_ = dataset.upper()
    identifier_ = identifier.lower()
    funcs_parse = {"SEED": utils_eeg_loading.read_and_parse_seed, 
                   # "DEAP": utils_eeg_loading.read_and_parse_deap, 
                   "DREAMER": utils_eeg_loading.read_and_parse_dreamer}
    
    # Validate dataset
    if dataset_ not in validation_.names_dataset:
        raise ValueError(f"Invalid dataset: {dataset_}. Choose from {', '.join(validation_.names_dataset)}.")
    
    # Load raw EEG data using the provided utility function
    eeg = funcs_parse[dataset_](identifier_)
    
    # Construct the output folder path for filtered data
    base_path = os.path.abspath(os.path.join(os.getcwd(), f"../../Research_Data/{dataset_}/original eeg/Filtered_EEG"))
    os.makedirs(base_path, exist_ok=True)
    
    # Filter the EEG data into different frequency bands
    filtered_eeg_dict = filter_eeg(eeg, sampling_rate=validation_.sampling_rates[dataset_], verbose=verbose)
    
    # Save filtered EEG data if requested
    if save:
        for band, filtered_eeg in filtered_eeg_dict.items():
            path_file = os.path.join(base_path, f"{identifier_}_{band}_eeg.fif")
            filtered_eeg.save(path_file, overwrite=True)
            if verbose:
                print(f"Saved {band} band filtered EEG to {path_file}")
    
    return filtered_eeg_dict

def filter_eeg_and_save_batch(dataset:str, subject_range:range, experiment_range:range, 
                              # subject_range:range=None, experiment_range:range=None, 
                              verbose:bool=True, save:bool=False):
    # Validation
    validation_ = validation
    
    # Normalize parameters
    dataset_ = dataset.upper()

    # Validate dataset
    if dataset_ not in validation_.names_dataset:
        raise ValueError(f"Invalid dataset: {dataset_}. Choose from {', '.join(validation_.names_dataset)}.")
    
    if subject_range is None or experiment_range is None:
        raise ValueError("Error of unexpected subject or experiment range designation.")
    
    # Batch operation
    for subject in subject_range:
        for experiment in experiment_range:
            identifier = f"sub{subject}ex{experiment}"
            print(f"Processing: {identifier}.")
            filter_eeg_4_dataset(dataset_, identifier, verbose=verbose, save=save)

# %% Feature Engineering
def compute_fc_matrices_batch(dataset, subject_range=range(1, 2), experiment_range=range(1, 2),
                              feature='pcc', band='joint', save=False, verbose=True):
    """
    Computes functional connectivity matrices for EEG datasets.

    Features:
    - Computes connectivity matrices based on the selected feature and frequency band.
    - Records total and average computation time.
    - Optionally saves results in HDF5 format.

    Parameters:
    - dataset (str): Dataset name ('SEED' or 'DREAMER').
    - subject_range (range): Range of subject IDs (default: range(1, 2)).
    - experiment_range (range): Range of experiment IDs (default: range(1, 2)).
    - feature (str): Connectivity feature ('pcc', 'plv', 'mi').
    - band (str): Frequency band ('delta', 'theta', 'alpha', 'beta', 'gamma', or 'joint').
    - save (bool): Whether to save results (default: False).
    - verbose (bool): Whether to print timing information (default: True).

    Returns:
    - dict: Dictionary containing computed functional connectivity matrices.
    """
    # Validation
    validation_ = validation
    
    # Normalize parameters
    dataset_ = dataset.upper()
    feature_ = feature.lower()
    band_ = band.lower()

    # Validate dataset
    if dataset_ not in validation_.names_dataset:
        raise ValueError(f"Invalid dataset '{dataset_}'. Supported datasets: {validation_.names_dataset}")
    if feature_ not in validation_.names_feature:
        raise ValueError(f"Invalid feature '{feature_}'. Supported features: {validation_.names_feature}")
    if band_ not in validation_.bands:
        raise ValueError(f"Invalid band '{band_}'. Supported bands: {validation_.bands}")
        
        
    # **************************
        
    
    fc_matrices = {}
    start_time = time.time()
    total_experiment_time = 0
    experiment_count = 0
    
    
    for subject in subject_range:
        for experiment in experiment_range:
            experiment_start = time.time()
            experiment_count += 1
            
            identifier = f"sub{subject}ex{experiment}"
            eeg_data = utils_eeg_loading.read_eeg_filtered(dataset_, identifier)
            
    
    if dataset == 'SEED': 
        sampling_rate = 200
        experiments = experiment_range
    elif dataset == 'DREAMER':
        sampling_rate = 128
        experiments = [None]

    for subject in subject_range:
        for experiment in experiments:
            experiment_start = time.time()
            experiment_count += 1
            
            identifier = f"sub{subject}ex{experiment}"
            eeg_data = utils_eeg_loading.read_eeg_filtered(dataset_, identifier)
            
            bands_to_process = ['delta', 'theta', 'alpha', 'beta', 'gamma'] if band == 'joint' else [band]

            fc_matrices[identifier] = {} if band == 'joint' else None

            # Test
            global data_
            data_ = eeg_data

            funcs = {"pcc": compute_corr_matrices, "plv": compute_plv_matrices,
                     "mi": compute_mi_matrices, "pli": compute_pli_matrices,
                     "wpli": compute_wpli_matrices, "dpli": compute_dpli_matrices,
                     "sdpli": compute_sdpli_matrices}

            for current_band in bands_to_process:
                data = np.array(eeg_data[current_band])

                result = funcs[feature](data, sampling_rate)
                
                if band == 'joint':
                    fc_matrices[identifier][current_band] = result
                else:
                    fc_matrices[identifier] = result

            experiment_duration = time.time() - experiment_start
            total_experiment_time += experiment_duration

            if verbose:
                print(f"Experiment {identifier} completed in {experiment_duration:.2f} seconds")

            if save:
                save_results(dataset, feature, identifier, fc_matrices[identifier])

    total_time = time.time() - start_time
    avg_experiment_time = total_experiment_time / experiment_count if experiment_count else 0

    if verbose:
        print(f"\nTotal time taken: {total_time:.2f} seconds")
        print(f"Average time per experiment: {avg_experiment_time:.2f} seconds")

    return fc_matrices

def save_results(dataset, feature, identifier, data):
    """Saves functional connectivity matrices to an HDF5 file."""
    path_parent = os.path.dirname(os.getcwd())
    path_parent_parent = os.path.dirname(path_parent)
    base_path = os.path.join(path_parent_parent, 'Research_Data', dataset, 'functional connectivity', f'{feature}_h5')
    os.makedirs(base_path, exist_ok=True)
    
    file_path = os.path.join(base_path, f"{identifier}.h5")
    with h5py.File(file_path, 'w') as f:
        if isinstance(data, dict):  # Joint band case
            for band, matrix in data.items():
                f.create_dataset(band, data=matrix, compression="gzip")
        else:  # Single band case
            f.create_dataset("connectivity", data=data, compression="gzip")

    print(f"Data saved to {file_path}")

from tqdm import tqdm
def compute_corr_matrices(eeg_data, sampling_rate, window=1, overlap=0, verbose=True, visualization=True):
    """
    Compute correlation matrices for EEG data using a sliding window approach.
    
    Parameters:
        eeg_data (numpy.ndarray): EEG data with shape (channels, time_samples).
        sampling_rate (int): Sampling rate of the EEG data in Hz.
        window (float): Window size in seconds for segmenting EEG data.
        overlap (float): Overlap fraction between consecutive windows (0 to 1).
        verbose (bool): If True, shows progress bar.
        visualization (bool): If True, displays correlation matrices.
    
    Returns:
        list of numpy.ndarray: List of correlation matrices for each window.
    """
    # Compute step size and segment length
    step = int(sampling_rate * window * (1 - overlap))
    segment_length = int(sampling_rate * window)

    # Generate overlapping segments
    split_segments = [
        eeg_data[:, i:i + segment_length]
        for i in range(0, eeg_data.shape[1] - segment_length + 1, step)
    ]

    # Compute correlation matrices with tqdm progress bar
    corr_matrices = []
    iterator = tqdm(enumerate(split_segments), total=len(split_segments), disable=not verbose, desc="Computing Corr Matrices")

    for idx, segment in iterator:
        if segment.shape[1] < segment_length:
            continue
        corr_matrix = np.corrcoef(segment)
        
        # Test here
        np.fill_diagonal(corr_matrix, 0)
        
        corr_matrices.append(corr_matrix)

    # Visualization
    if visualization and corr_matrices:
        avg_corr_matrix = np.mean(corr_matrices, axis=0)
        utils_visualization.draw_projection(avg_corr_matrix)

    return corr_matrices

def compute_plv_matrices(eeg_data, sampling_rate, window=1, overlap=0, verbose=True, visualization=True):
    """
    Compute Phase Locking Value (PLV) matrices for EEG data using a sliding window approach.

    Parameters:
        eeg_data (numpy.ndarray): EEG data with shape (channels, time_samples).
        sampling_rate (int): Sampling rate of the EEG data in Hz.
        window (float): Window size in seconds for segmenting EEG data.
        overlap (float): Overlap fraction between consecutive windows (0 to 1).
        verbose (bool): If True, shows progress bar.
        visualization (bool): If True, displays average PLV matrix.

    Returns:
        list of numpy.ndarray: List of PLV matrices for each window.
    """
    step = int(sampling_rate * window * (1 - overlap))
    segment_length = int(sampling_rate * window)

    # Split EEG data into overlapping windows
    split_segments = [
        eeg_data[:, i:i + segment_length]
        for i in range(0, eeg_data.shape[1] - segment_length + 1, step)
    ]

    plv_matrices = []

    iterator = tqdm(enumerate(split_segments), total=len(split_segments), disable=not verbose, desc="Computing PLV Matrices")

    for idx, segment in iterator:
        if segment.shape[1] < segment_length:
            continue  # Skip incomplete segments

        # Hilbert transform to extract phase
        analytic_signal = hilbert(segment, axis=1)
        phase_data = np.angle(analytic_signal)

        num_channels = phase_data.shape[0]
        plv_matrix = np.zeros((num_channels, num_channels))

        for ch1 in range(num_channels):
            for ch2 in range(num_channels):
                phase_diff = phase_data[ch1, :] - phase_data[ch2, :]
                plv_matrix[ch1, ch2] = np.abs(np.mean(np.exp(1j * phase_diff)))
                
        # Test here
        np.fill_diagonal(plv_matrix, 0)        
        
        plv_matrices.append(plv_matrix)

    # Visualization
    if visualization and plv_matrices:
        avg_plv_matrix = np.mean(plv_matrices, axis=0)
        utils_visualization.draw_projection(avg_plv_matrix)

    return plv_matrices

def compute_pli_matrices(eeg_data, sampling_rate, window=1, overlap=0, verbose=True, visualization=True):
    """
    Compute Phase Lag Index (PLI) matrices for EEG data using a sliding window approach.

    Parameters:
        eeg_data (numpy.ndarray): EEG data with shape (channels, time_samples).
        sampling_rate (int): Sampling rate of the EEG data in Hz.
        window (float): Window size in seconds for segmenting EEG data.
        overlap (float): Overlap fraction between consecutive windows (0 to 1).
        verbose (bool): If True, shows progress bar.
        visualization (bool): If True, displays average PLI matrix.

    Returns:
        list of numpy.ndarray: List of PLI matrices for each window.
    """
    step = int(sampling_rate * window * (1 - overlap))
    segment_length = int(sampling_rate * window)

    # Generate overlapping segments
    split_segments = [
        eeg_data[:, i:i + segment_length]
        for i in range(0, eeg_data.shape[1] - segment_length + 1, step)
    ]

    pli_matrices = []

    iterator = tqdm(enumerate(split_segments), total=len(split_segments), disable=not verbose, desc="Computing PLI Matrices")

    for idx, segment in iterator:
        if segment.shape[1] < segment_length:
            continue

        analytic_signal = hilbert(segment, axis=1)
        phase_data = np.angle(analytic_signal)

        num_channels = phase_data.shape[0]
        pli_matrix = np.zeros((num_channels, num_channels))

        for ch1 in range(num_channels):
            for ch2 in range(num_channels):
                if ch1 == ch2:
                    continue
                phase_diff = phase_data[ch1] - phase_data[ch2]
                pli = np.abs(np.mean(np.sign(np.sin(phase_diff))))
                pli_matrix[ch1, ch2] = pli
        
        # Test here
        np.fill_diagonal(pli_matrix, 0)  
        
        pli_matrices.append(pli_matrix)

    if visualization and pli_matrices:
        avg_pli_matrix = np.mean(pli_matrices, axis=0)
        utils_visualization.draw_projection(avg_pli_matrix)

    return pli_matrices

def compute_dpli_matrices(eeg_data, sampling_rate, window=1, overlap=0, verbose=True, visualization=True):
    """
    Compute directed Phase Lag Index (dPLI) matrices for EEG data using a sliding window approach.

    Parameters:
        eeg_data (numpy.ndarray): EEG data with shape (channels, time_samples).
        sampling_rate (int): Sampling rate of the EEG data in Hz.
        window (float): Window size in seconds for segmenting EEG data.
        overlap (float): Overlap fraction between consecutive windows (0 to <1).
        verbose (bool): If True, shows progress bar.
        visualization (bool): If True, displays average dPLI matrix.

    Returns:
        list of numpy.ndarray: List of dPLI matrices for each window.
    """
    segment_length = int(sampling_rate * window)
    step = int(segment_length * (1 - overlap))

    if step <= 0:
        raise ValueError("overlap must be less than 1, resulting in a positive step size.")

    split_segments = [
        eeg_data[:, i:i + segment_length]
        for i in range(0, eeg_data.shape[1] - segment_length + 1, step)
    ]

    dpli_matrices = []

    iterator = tqdm(
        enumerate(split_segments),
        total=len(split_segments),
        disable=not verbose,
        desc="Computing dPLI Matrices"
    )

    for idx, segment in iterator:
        analytic_signal = hilbert(segment, axis=1)
        phase_data = np.angle(analytic_signal)

        phase_diff = phase_data[:, None, :] - phase_data[None, :, :]

        dpli_matrix = np.mean(
            np.heaviside(np.sin(phase_diff), 0.5),
            axis=-1
        )
        
        # Test here
        np.fill_diagonal(dpli_matrix, 0)

        dpli_matrices.append(dpli_matrix)

    if visualization and dpli_matrices:
        avg_dpli_matrix = np.mean(dpli_matrices, axis=0)
        utils_visualization.draw_projection(avg_dpli_matrix)

    return dpli_matrices

def compute_sdpli_matrices(eeg_data, sampling_rate, window=1, overlap=0, verbose=True, visualization=True):
    """
    Compute signed directed Phase Lag Index (sdPLI) matrices for EEG data using a sliding window approach.

    Parameters:
        eeg_data (numpy.ndarray): EEG data with shape (channels, time_samples).
        sampling_rate (int): Sampling rate of the EEG data in Hz.
        window (float): Window size in seconds for segmenting EEG data.
        overlap (float): Overlap fraction between consecutive windows (0 to <1).
        verbose (bool): If True, shows progress bar.
        visualization (bool): If True, displays average dPLI matrix.

    Returns:
        list of numpy.ndarray: List of dPLI matrices for each window.
    """
    segment_length = int(sampling_rate * window)
    step = int(segment_length * (1 - overlap))

    if step <= 0:
        raise ValueError("overlap must be less than 1, resulting in a positive step size.")

    split_segments = [
        eeg_data[:, i:i + segment_length]
        for i in range(0, eeg_data.shape[1] - segment_length + 1, step)
    ]

    sdpli_matrices = []

    iterator = tqdm(
        enumerate(split_segments),
        total=len(split_segments),
        disable=not verbose,
        desc="Computing sdPLI Matrices"
    )

    for idx, segment in iterator:
        analytic_signal = hilbert(segment, axis=1)
        phase_data = np.angle(analytic_signal)

        phase_diff = phase_data[:, None, :] - phase_data[None, :, :]

        dpli_matrix = np.mean(
            np.heaviside(np.sin(phase_diff), 0.5),
            axis=-1
        )
        
        sdpli_matrix = 2 * dpli_matrix - 1
        
        # Test here
        np.fill_diagonal(sdpli_matrix, 0)
        
        sdpli_matrices.append(sdpli_matrix)

    if visualization and sdpli_matrices:
        avg_sdpli_matrix = np.mean(sdpli_matrices, axis=0)
        utils_visualization.draw_projection(avg_sdpli_matrix)

    return sdpli_matrices

def compute_wpli_matrices(eeg_data, sampling_rate, window=1, overlap=0, verbose=True, visualization=True):
    """
    Compute weighted Phase Lag Index (wPLI) matrices for EEG data using a sliding window approach.

    Parameters:
        eeg_data (numpy.ndarray): EEG data with shape (channels, time_samples).
        sampling_rate (int): Sampling rate of the EEG data in Hz.
        window (float): Window size in seconds for segmenting EEG data.
        overlap (float): Overlap fraction between consecutive windows (0 to 1).
        verbose (bool): If True, shows progress bar.
        visualization (bool): If True, displays average wPLI matrix.

    Returns:
        list of numpy.ndarray: List of wPLI matrices for each window.
    """
    step = int(sampling_rate * window * (1 - overlap))
    segment_length = int(sampling_rate * window)

    # Create sliding window segments
    split_segments = [
        eeg_data[:, i:i + segment_length]
        for i in range(0, eeg_data.shape[1] - segment_length + 1, step)
    ]

    wpli_matrices = []
    iterator = tqdm(enumerate(split_segments), total=len(split_segments), disable=not verbose, desc="Computing wPLI Matrices")

    for idx, segment in iterator:
        if segment.shape[1] < segment_length:
            continue

        analytic_signal = hilbert(segment, axis=1)
        
        num_channels = analytic_signal.shape[0]
        wpli_matrix = np.zeros((num_channels, num_channels))

        for ch1 in range(num_channels):
            for ch2 in range(num_channels):
                if ch1 == ch2:
                    continue

                csd = analytic_signal[ch1] * np.conj(analytic_signal[ch2])
                im_part = np.imag(csd)

                numerator = np.abs(np.mean(im_part))
                denominator = np.mean(np.abs(im_part)) # avoid divide-by-zero
                if denominator == 0:
                    denominator = 1e-10
                wpli = numerator / denominator
                wpli_matrix[ch1, ch2] = wpli
        
        # Test here
        np.fill_diagonal(wpli_matrix, 0)
        
        wpli_matrices.append(wpli_matrix)

    if visualization and wpli_matrices:
        avg_wpli_matrix = np.mean(wpli_matrices, axis=0)
        utils_visualization.draw_projection(avg_wpli_matrix)

    return wpli_matrices

from sklearn.metrics import mutual_info_score
def compute_mi_matrices(eeg_data, sampling_rate, window=1, overlap=0, verbose=True, visualization=True, bins=16):
    """
    Compute Mutual Information (MI) matrices for EEG data using a sliding window approach.

    Parameters:
        eeg_data (numpy.ndarray): EEG data with shape (channels, time_samples).
        sampling_rate (int): Sampling rate of the EEG data in Hz.
        window (float): Window size in seconds for segmenting EEG data.
        overlap (float): Overlap fraction between consecutive windows (0 to 1).
        verbose (bool): If True, shows progress bar.
        visualization (bool): If True, displays average MI matrix.
        bins (int): Number of bins for discretizing EEG signals before MI computation.

    Returns:
        list of numpy.ndarray: List of MI matrices for each window.
    """
    step = int(sampling_rate * window * (1 - overlap))
    segment_length = int(sampling_rate * window)

    # Create overlapping segments
    split_segments = [
        eeg_data[:, i:i + segment_length]
        for i in range(0, eeg_data.shape[1] - segment_length + 1, step)
    ]

    mi_matrices = []
    iterator = tqdm(enumerate(split_segments), total=len(split_segments), disable=not verbose, desc="Computing MI Matrices")

    for idx, segment in iterator:
        if segment.shape[1] < segment_length:
            continue

        num_channels = segment.shape[0]
        mi_matrix = np.zeros((num_channels, num_channels))

        # Discretize each channel
        discretized = np.array([
            np.digitize(segment[ch], bins=np.histogram_bin_edges(segment[ch], bins=bins))
            for ch in range(num_channels)
        ])

        for ch1 in range(num_channels):
            for ch2 in range(num_channels):
                if ch1 == ch2:
                    continue
                mi = mutual_info_score(discretized[ch1], discretized[ch2])
                mi_matrix[ch1, ch2] = mi
        
        # Test here
        np.fill_diagonal(mi_matrix, 0)
        
        mi_matrices.append(mi_matrix)

    if visualization and mi_matrices:
        avg_mi_matrix = np.mean(mi_matrices, axis=0)
        utils_visualization.draw_projection(avg_mi_matrix)

    return mi_matrices

def compute_average_fcs(dataset, subjects=range(1, 16), experiments=range(1, 4), 
                        feature='pcc', band='joint', in_file_type='.h5', 
                        save=False, verbose=False, visualization=False):
    """
    Computes and optionally saves or visualizes the averaged functional connectivity matrices.

    Parameters
    ----------
    dataset : str
        Dataset name (e.g., 'seed').
    subjects : iterable
        List or range of subject indices.
    experiments : iterable
        List or range of experiment indices.
    feature : str
        Feature type, e.g., 'pcc', 'plv', 'pli'.
    band : str
        Frequency band or 'joint' for all bands.
    in_file_type : str
        Input file type, '.h5' or '.mat'.
    out_file_type : str
        Output file type, '.h5' or '.mat'.
    save : bool
        Whether to save the result.
    verbose : bool
        Whether to print verbose output.
    visualization : bool
        Whether to visualize the global averaged matrix.

    Returns
    -------
    np.ndarray
        The global averaged functional connectivity matrix.
    """
    
    assert dataset.lower() in {'seed'}, "Unsupported dataset."
    assert feature.lower() in {'pcc', 'plv', 'mi', 'pli', 'wpli', 'dpli', 'sdpli'}, "Invalid feature."
    assert band.lower() in {'joint', 'alpha', 'beta', 'gamma', 'delta', 'theta'}, "Invalid band."
    assert in_file_type in {'.h5', '.mat'}, "Unsupported input file type."

    fcs_averaged_dict, fcs_averaged_dict_ = [], {'alpha': [], 'beta': [], 'gamma': [], 'delta': [], 'theta': []}
    
    for subject in subjects:
        for experiment in experiments:
            identifier = f"sub{subject}ex{experiment}"
            if verbose:
                print(f"Processing: {identifier}")
            
            features = utils_feature_loading.read_fcs(dataset, identifier, feature, band, in_file_type)
            
            if band == 'joint':
                try:
                    avg_bands = [{"average": np.mean(features[b], axis=0), 
                                  "band": b, "subject": subject, "experiment": experiment} 
                                 for b in ['alpha', 'beta', 'gamma', 'delta', 'theta']]
                    
                    fcs_averaged_dict.append(avg_bands)
                    for entry in avg_bands:                        
                        fcs_averaged_dict_[entry["band"]].append(entry["average"])
                    
                    # Correct theta/delta swap if necessary
                except KeyError:
                    avg_bands = [{"average": np.mean(features[b], axis=0), 
                                  "band": b, "subject": subject, "experiment": experiment} 
                                 for b in ['alpha', 'beta', 'gamma']]
                    
                    fcs_averaged_dict.append(avg_bands)
                    for entry in avg_bands:                        
                        fcs_averaged_dict_[entry["band"]].append(entry["average"])

    # Compute global average
    try:
        fcs_global_averaged = {b: np.mean(fcs_averaged_dict_[b], axis=0)
                               for b in ['alpha', 'beta', 'gamma', 'delta', 'theta']}
    except KeyError:
        fcs_global_averaged = {b: np.mean(fcs_averaged_dict_[b], axis=0)
                               for b in ['alpha', 'beta', 'gamma']}
        
    if visualization:
        for fc in fcs_global_averaged.values():
            utils_visualization.draw_projection(fc)

    if save:
        save_results(dataset, feature, f'global_averaged_{subject}_15', fcs_global_averaged)
        
        if verbose:
            print("Results saved to .h5 and .mat")

    return fcs_global_averaged, fcs_averaged_dict_
        
# %% Label Engineering
def labels_upsampling(labels, categories="binary", ratio=63):
    """
    Transform trial-level labels using adaptive thresholds.

    Parameters
    ----------
    labels : array-like
        A 2D array with shape (n_trials, n_dimensions).

    categories : {"binary", "ternary", "continuous"}, default="binary"
        Label transformation method:
        - binary:
            Values <= the median of each dimension are mapped to 0,
            and values > the median are mapped to 1.
        - ternary:
            Values are divided into low, middle, and high classes using
            the 1/3 and 2/3 quantiles of each dimension.
        - continuous:
            Labels are returned as float32 without discretization.

    ratio : int, default=63
        Number of windows corresponding to each trial.

    Returns
    -------
    transformed_labels : np.ndarray
        Transformed labels with shape
        (n_trials * ratio, n_dimensions).
    """
    labels = np.asarray(labels)

    # Support arbitrary label dimensions; input shape is (n_trials, n_dimensions)

    if labels.ndim == 1:
        labels = np.expand_dims(labels, axis=1)
    elif labels.ndim > 2:
        raise ValueError(
            "labels must be a 2D array with shape "
            f"(n_trials, n_dimensions), but got {labels.shape}"
        )

    if labels.shape[0] == 0:
        raise ValueError("labels must contain at least one trial.")

    if labels.shape[1] == 0:
        raise ValueError("labels must contain at least one label dimension.")

    if not np.issubdtype(labels.dtype, np.number):
        raise TypeError("labels must contain numeric values.")

    if not np.all(np.isfinite(labels)):
        raise ValueError("labels must not contain NaN or infinite values.")

    if not isinstance(ratio, (int, np.integer)) or isinstance(ratio, bool):
        raise TypeError("ratio must be an integer.")

    if ratio <= 0:
        raise ValueError("ratio must be a positive integer.")

    if not isinstance(categories, str):
        raise TypeError("categories must be a string.")

    categories = categories.lower().strip()

    if categories == "binary":
        # Compute the median separately for each label dimension
        # thresholds has shape (1, n_dimensions) and broadcasts automatically
        thresholds = np.median(
            labels,
            axis=0,
            keepdims=True,
        )

        # Less than or equal to the median: 0
        # Greater than the median: 1
        transformed_labels = (
                labels > thresholds
        ).astype(np.int64)

    elif categories == "ternary":
        # Compute the 1/3 and 2/3 quantiles separately for each label dimension
        lower_thresholds = np.quantile(
            labels,
            q=1 / 3,
            axis=0,
            keepdims=True,
        )

        upper_thresholds = np.quantile(
            labels,
            q=2 / 3,
            axis=0,
            keepdims=True,
        )

        # <= lower quantile: 0 (low)
        # > lower quantile and <= upper quantile: 1 (middle)
        # > upper quantile: 2 (high)
        transformed_labels = np.where(
            labels <= lower_thresholds,
            0,
            np.where(
                labels <= upper_thresholds,
                1,
                2,
            ),
        ).astype(np.int64)

    elif categories == "continuous":
        transformed_labels = labels.astype(
            np.float32,
            copy=False,
        )

    else:
        raise ValueError(
            "categories must be 'binary', 'ternary', or 'continuous'."
        )

    # Repeat each trial label ratio times to match all windows of that trial
    transformed_labels = np.repeat(
        transformed_labels,
        repeats=ratio,
        axis=0,
    )

    return transformed_labels

# %% Normalize
from scipy.stats import boxcox, yeojohnson
def normalize_matrix(matrix, method='minmax', epsilon=1e-8, param=None):
    """
    对矩阵或批量矩阵进行归一化或变换处理。
    
    支持方法包括：minmax, max, mean, z-score, boxcox, yeojohnson, sqrt, log, none。
    可输入单个矩阵 (H, W) 或批量矩阵 (N, H, W)。

    参数:
        matrix (np.ndarray): 输入矩阵或批量矩阵。
        method (str): 归一化方法。
        epsilon (float): 防止除零的极小值。
        param (dict): 额外参数，如 target_range 或 lmbda。
    """
    if param is None:
        param = {}
    a, b = param.get('target_range', (0, 1))
    lmbda = param.get('lmbda', None)

    # 判断是否批处理
    is_batch = matrix.ndim == 3
    matrices = matrix if is_batch else matrix[None, ...]

    normalized = []
    for mat in matrices:
        mat = mat.copy()

        if method == 'minmax':
            min_val, max_val = np.min(mat), np.max(mat)
            scale = max(max_val - min_val, epsilon)
            mat = ((mat - min_val) / scale) * (b - a) + a

        elif method == 'max':
            max_val = max(np.max(np.abs(mat)), epsilon)
            mat = mat / max_val

        elif method == 'mean':
            mean_val = max(np.mean(mat), epsilon)
            mat = mat / mean_val

        elif method == 'z-score':
            mean_val, std_val = np.mean(mat), np.std(mat)
            mat = (mat - mean_val) / max(std_val, epsilon)

        elif method == 'boxcox':
            mat += epsilon
            if np.any(mat <= 0):
                raise ValueError("Box-Cox 要求所有值 > 0")
            mat = boxcox(mat.flatten(), lmbda=lmbda)[0].reshape(mat.shape)

        elif method == 'yeojohnson':
            mat = yeojohnson(mat.flatten(), lmbda=lmbda)[0].reshape(mat.shape)

        elif method == 'sqrt':
            if np.any(mat < 0):
                raise ValueError("平方根要求非负值")
            mat = np.sqrt(mat + epsilon)

        elif method == 'log':
            if np.any(mat <= 0):
                raise ValueError("对数要求值 > 0")
            mat = np.log(mat + epsilon)

        elif method == 'none':
            pass

        else:
            raise ValueError(f"不支持的归一化方法: {method}")

        normalized.append(mat)

    result = np.stack(normalized) if is_batch else normalized[0]
    return result

# %% Tools
def remove_idx_manual(A, manual_idxs=[]):
    if len(A.shape) == 1:
        A = np.delete(A, manual_idxs, axis=0)
    elif len(A.shape) == 2:
        A = np.delete(A, manual_idxs, axis=0)
        A = np.delete(A, manual_idxs, axis=1)
    elif len(A.shape) == 3:
        A = np.delete(A, manual_idxs, axis=1)
        A = np.delete(A, manual_idxs, axis=2)
    return A

def insert_idx_manual(A, manual_idxs=[], value=0):
    if len(A.shape) == 1:
        for idx in manual_idxs:
            if idx >= len(A):
                A = np.append(A, value)
            else:
                A = np.insert(A, idx, value)
                
    return A

def compute_electrode_retention_list(ele_strengths_comprehensive, err):
    _ele_strengths_comprehensive = ele_strengths_comprehensive
    _err = err

    # The importance of each electrode/channel is equal to its corresponding node strength
    k = max(1, int(len(_ele_strengths_comprehensive) * _err))  # err (persentage) to top k (int)

    _ele_strengths_comprehensive = {"strengths": _ele_strengths_comprehensive}
    ele_importances = pd.DataFrame(_ele_strengths_comprehensive)
    ele_importances.sort_values(by=["strengths"], ascending=False, inplace=True)
    electrode_retention_list_df = ele_importances.iloc[:k]
    electrode_retention_list_ar = np.array(electrode_retention_list_df.index.tolist())

    return electrode_retention_list_ar, electrode_retention_list_df

# %% Example usage
if __name__ == "__main__":
    # %% Example for SEED
    # Read original EEG
    eeg_sample = utils_eeg_loading.read_eeg_original_dataset("seed", "sub1ex1")

    # Frequency band decomposition
    filtered_eeg_sample = filter_eeg_seed("sub1ex1", sampling_rate=200, verbose=True, save=False)

    # Feature engineering; distance matrix
    channel_names, distance_matrix = compute_distance_matrix("seed")

    plot_settings = {"xticklabels": channel_names, "yticklabels": channel_names,
                     "show_colorbar": True, "max_labels": 20,
                     "title": "Inter-Electrode Distance Matrix, for SEED",
                     "title_position": "upper", "cmap": "RdBu", }  # cmap="RdBu_r", cmap="viridis"

    utils_visualization.draw_projection(distance_matrix, **plot_settings)

    # Feature engineering; compute functional connectivities
    eeg_sample_parsed = utils_eeg_loading.read_and_parse_seed("sub1ex1")

    fcs_pcc_sample = compute_corr_matrices(eeg_sample_parsed, sampling_rate=200, verbose=True, visualization=False)
    plot_settings["cmap"] = "RdBu_r"
    plot_settings["title"] = "Functional Connectivity Matrix, \n averaged across temporal windows (PCC, SEED)"
    utils_visualization.draw_projection(np.mean(fcs_pcc_sample, axis=0), **plot_settings)

    # fcs_plv_sample = compute_plv_matrices(eeg_sample_parsed, sampling_rate=200, verbose=True, visualization=False)
    # plot_settings["title"] = "Functional Connectivity Matrix, averaged across temporal windows (PLV, SEED)"
    # utils_visualization.draw_projection(np.mean(fcs_plv_sample, axis=0), **plot_settings)

    # Label engineering
    labels_seed = utils_feature_loading.read_labels("seed", header=True, identifier="valence")

    # Feature engineering; batched computation
    fc_pcc_matrices_seed = compute_fc_matrices_batch("seed", feature="pcc", subject_range=range(1, 2),
                                                     experiment_range=range(1, 4), save=False)
    # fc_plv_matrices_seed = fc_matrices_circle("seed", feature="plv", subject_range=range(1, 2), experiment_range=range(1, 4), save=False)

    # Feature engineering; compute globally averaged fucntional matrices
    fcs_globally_averaged, _ = compute_average_fcs("seed", feature="pcc", subjects=range(1, 6), experiments=range(1, 4),
                                                   save=False,
                                                   verbose=True, visualization=False)

    fcs_globally_averaged_sample = fcs_globally_averaged["alpha"]
    plot_settings[
        "title"] = "Functional Connectivity Matrix, \n globally averaged across temporal windows and recordings (PCC, SEED)"
    utils_visualization.draw_projection(fcs_globally_averaged_sample, **plot_settings)

    # %% Example for DREAMER

    # %% End program actions
    utils_tools.end_program_actions(play_sound=True, shutdown=False, countdown_seconds=120)