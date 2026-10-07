# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 23:15:11 2025

@author: 18307
"""

import os
import sys
import time
import h5py

import numpy as np

from scipy.signal import hilbert

#
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils.utils_validation import Validation, PathDefinition
from utils import (
    utils_basic_reading,
    utils_feature_loading,
    utils_interaction,
    utils_eeg_loading,
)

# %% Feature Engineering; Batch
def compute_fc_matrices_batch(dataset, identifier_1, identifier_2, feature, 
                              band="joint", save=False, verbose=True):
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
    dataset = Validation.validate_dataset(dataset)
    feature = Validation.validate_feature(feature)
    band = Validation.validate_bands(band)
    
    sampling_rate = Validation.DATASET_INFO[dataset]["sfreq"]
    
    identifier_1 = Validation.validate_identifier(identifier_1)
    identifier_2 = Validation.validate_identifier(identifier_2)
    subject_range = range(utils_basic_reading.get_first_number(identifier_1), 
                          utils_basic_reading.get_first_number(identifier_2) + 1)
    experiment_range = range(utils_basic_reading.get_last_number(identifier_1),
                             utils_basic_reading.get_last_number(identifier_2) + 1)
    
    # Functional connectivity calculator
    funcs = {"pcc": compute_pcc_matrices, "plv": compute_plv_matrices,
             "mi": compute_mi_matrices, "pli": compute_pli_matrices,
             "wpli": compute_wpli_matrices, "dpli": compute_dpli_matrices,
             "sdpli": compute_sdpli_matrices}
    
    func = funcs[feature]
    
    # A built-in function for saving connectivity results
    def save_connectivity_results(dataset, feature, identifier, data):
        # Create folder if it does not already exist
        path_save_folder = PathDefinition.retrieve_path("functional_connectivity", dataset, feature)
        os.makedirs(path_save_folder, exist_ok=True)
    
        path_save_file = os.path.join(path_save_folder, f"{identifier}.h5")
    
        with h5py.File(path_save_file, "w") as f:
            if isinstance(data, dict):  # Joint band case
                for band, matrix in data.items():
                    f.create_dataset(band, data=matrix, compression="gzip")
            else:  # Single band case
                f.create_dataset("connectivity", data=data, compression="gzip")
    
        print(f"Data saved to {path_save_file}")
    
    # Batch circle
    start_time = time.time()
    total_experiment_time = 0
    experiment_count = 0
    
    con_matrices = {}
    for subject in subject_range:
        for experiment in experiment_range:
            _identifier = f"sub{subject}ex{experiment}"
            print(f"Processing: {_identifier}.")
            
            con_matrices.update({_identifier: {}})
            
            experiment_start = time.time()
            experiment_count += 1
            
            eeg_bands, _ = utils_eeg_loading.read_eeg_decomposed(dataset, _identifier, return_type="ndarray")
            
            if band == "joint":
                for _current_band, _eeg_current_band in eeg_bands.items():
                    connectivities = func(_eeg_current_band, sampling_rate)
                    
                    con_matrices[_identifier].update({_current_band: connectivities})
            
            elif band != "joint":
                connectivities = func(_eeg_current_band, sampling_rate)
                
                con_matrices[_identifier].update({band: connectivities})

            experiment_duration = time.time() - experiment_start
            total_experiment_time += experiment_duration

            if verbose:
                print(f"Experiment {_identifier} completed in {experiment_duration:.2f} seconds")

            if save:
                save_connectivity_results(dataset, feature, _identifier, con_matrices[_identifier])

    total_time = time.time() - start_time
    avg_experiment_time = total_experiment_time / experiment_count if experiment_count else 0

    if verbose:
        print(f"\nTotal time taken: {total_time:.2f} seconds")
        print(f"Average time per experiment: {avg_experiment_time:.2f} seconds")

    return con_matrices

# %% Feature Engineering; Global average
def compute_average_fc_matrix(dataset, identifier_1, identifier_2, feature, 
                              band="joint", save=False, verbose=True, visualization=True):
    # Validation
    dataset = Validation.validate_dataset(dataset)
    feature = Validation.validate_feature(feature)
    band = Validation.validate_bands(band)
    
    identifier_1 = Validation.validate_identifier(identifier_1)
    identifier_2 = Validation.validate_identifier(identifier_2)
    subject_range = range(utils_basic_reading.get_first_number(identifier_1), 
                          utils_basic_reading.get_first_number(identifier_2) + 1)
    experiment_range = range(utils_basic_reading.get_last_number(identifier_1),
                             utils_basic_reading.get_last_number(identifier_2) + 1)

    # A built-in function for saving connectivity results
    def save_connectivity_results(dataset, feature, identifier, data):
        # Create folder if it does not already exist
        path_save_folder = PathDefinition.retrieve_path("functional_connectivity", dataset, feature)
        os.makedirs(path_save_folder, exist_ok=True)
    
        path_save_file = os.path.join(path_save_folder, f"{identifier}.h5")
    
        with h5py.File(path_save_file, "w") as f:
            if isinstance(data, dict):  # Joint band case
                for band, matrix in data.items():
                    f.create_dataset(band, data=matrix, compression="gzip")
            else:  # Single band case
                f.create_dataset("connectivity", data=data, compression="gzip")
    
        print(f"Data saved to {path_save_file}")
    
    # Averaging across matrices
    con_matrices = {}
    for subject in subject_range:
        for experiment in experiment_range:
            _identifier = f"sub{subject}ex{experiment}"
            print(f"Processing: {_identifier}.")
            
            con_matrices_cur_id = utils_feature_loading.read_features(dataset, _identifier, feature)
            
            con_matrices.update({_identifier: con_matrices_cur_id})
    
    con_matrix_avg = {}
    
    bands = next(iter(con_matrices.values())).keys()
    
    for band in bands:
        matrices = [
            con_matrices[_id][band]
            for _id in con_matrices
        ]
    
        # Shape: (subjects, N, W, H)
        matrices = np.stack(matrices, axis=0)
    
        # Average across subjects and N
        con_matrix_avg[band] = matrices.mean(axis=(0, 1))
    
    if save:
        save_connectivity_results(dataset, feature, f"avg_{identifier_1}_{identifier_2}", con_matrix_avg)
    
    return con_matrix_avg

# %% Functional connectivity calculator
from tqdm import tqdm
from sklearn.metrics import mutual_info_score

def compute_pcc_matrices(eeg_data, sampling_rate, window=1, overlap=0, verbose=True, visualization=True):
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
        
        corr_matrices.append(corr_matrix)

    # Visualization
    if visualization and corr_matrices:
        avg_corr_matrix = np.mean(corr_matrices, axis=0)
        utils_interaction.draw_projection(avg_corr_matrix)

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
        
        plv_matrices.append(plv_matrix)

    # Visualization
    if visualization and plv_matrices:
        avg_plv_matrix = np.mean(plv_matrices, axis=0)
        utils_interaction.draw_projection(avg_plv_matrix)

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
        
        pli_matrices.append(pli_matrix)

    if visualization and pli_matrices:
        avg_pli_matrix = np.mean(pli_matrices, axis=0)
        utils_interaction.draw_projection(avg_pli_matrix)

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

        dpli_matrices.append(dpli_matrix)

    if visualization and dpli_matrices:
        avg_dpli_matrix = np.mean(dpli_matrices, axis=0)
        utils_interaction.draw_projection(avg_dpli_matrix)

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
        
        sdpli_matrices.append(sdpli_matrix)

    if visualization and sdpli_matrices:
        avg_sdpli_matrix = np.mean(sdpli_matrices, axis=0)
        utils_interaction.draw_projection(avg_sdpli_matrix)

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
        
        wpli_matrices.append(wpli_matrix)

    if visualization and wpli_matrices:
        avg_wpli_matrix = np.mean(wpli_matrices, axis=0)
        utils_interaction.draw_projection(avg_wpli_matrix)

    return wpli_matrices

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
        
        mi_matrices.append(mi_matrix)

    if visualization and mi_matrices:
        avg_mi_matrix = np.mean(mi_matrices, axis=0)
        utils_interaction.draw_projection(avg_mi_matrix)

    return mi_matrices

# %% Example usage
if __name__ == "__main__":
    # %% Functional connectivity
    # compute_fc_matrices_batch("seed", "sub10ex1", "sub15ex3", feature="pli", band="joint", save=True, verbose=True)

    # End program actions
    # utils_interaction.end_program_actions(play_sound=True, shutdown=False, countdown_seconds=30)
    
    # %% Average connectivity matrices
    # compute_average_fc_matrix("seed", "sub1ex1", "sub1ex3", "pli", band="joint", save=False, verbose=True)
    
    # End program actions
    utils_interaction.end_program_actions(play_sound=True, shutdown=False, countdown_seconds=30)