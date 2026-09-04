# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 16:38:27 2025

@author: usouu
"""
import os
import h5py
import mne
import scipy
import numpy as np
import pandas as pd

# %% Basic File Reading Functions
def read_txt(path_file, header=False, encoding='utf-8'):
    """
    Reads a text file and returns its content as a Pandas DataFrame.

    Parameters:
    - path_file (str): Path to the text file.
    - header (bool or int): Whether the file contains a header row.
                            If True, use the first row as header.
                            If False, no header is assumed.
                            If int, use that row number as header.
    - encoding (str): Encoding to use when reading the file. Default is 'utf-8'.

    Returns:
    - pd.DataFrame: DataFrame containing the parsed text data.
    """
    if not os.path.isfile(path_file):
        raise FileNotFoundError(f"File not found: {path_file}")

    if isinstance(header, bool):
        header_value = 0 if header else None
    elif isinstance(header, int):
        header_value = header
    else:
        raise ValueError("`header` must be a boolean or an integer.")

    try:
        txt = pd.read_csv(path_file, sep=r'\s+', engine='python', header=header_value, encoding=encoding)
    except Exception as e:
        raise ValueError(f"Failed to read file '{path_file}': {e}")
    
    return txt

def read_xlsx(path_file):
    xls = pd.ExcelFile(path_file)
    dfs = {sheet: xls.parse(sheet) for sheet in xls.sheet_names}
    return dfs

import pickle
def read_hdf5(path_file):
    """
    Reads an HDF5 file and returns its contents as a dictionary.
    
    Parameters:
    - path_file (str): Path to the HDF5 file.
    
    Returns:
    - dict: Parsed data from the HDF5 file.
    
    Raises:
    - FileNotFoundError: If the file does not exist.
    - TypeError: If the file is not a valid HDF5 format.
    """
    if not os.path.exists(path_file):
        raise FileNotFoundError(f"File not found: {path_file}")

    try:
        with h5py.File(path_file, 'r') as f:
            return {key: simplify_mat_structure(f[key]) for key in f.keys()}
    except OSError:
        raise TypeError(f"File '{path_file}' is not in HDF5 format.")

def read_dat(path_file):
    if not os.path.exists(path_file):
        raise FileNotFoundError(f"File not found: {path_file}")
        
    try:
        with open(path_file, "rb") as file:
            dataset = pickle.load(file, encoding="latin1")
            return dataset
    except OSError:
        raise TypeError(f"File '{path_file}' is not in DAT format.")
        
def read_mat(path_file, simplify=True):
    """
    Reads a MATLAB .mat file, supporting both HDF5 and older formats.
    
    Parameters:
    - path_file (str): Path to the .mat file.
    - simplify (bool): Whether to simplify the data structure (default: True).
    
    Returns:
    - dict: Parsed MATLAB file data.
    
    Raises:
    - FileNotFoundError: If the file does not exist.
    - TypeError: If the file format is invalid.
    """
    if not os.path.exists(path_file):
        raise FileNotFoundError(f"File not found: {path_file}")
    
    try:
        # Attempt to read as HDF5 format
        with h5py.File(path_file, 'r') as f:
            return {key: simplify_mat_structure(f[key]) for key in f.keys()} if simplify else f
    except OSError:
        try:
            # Read as non-HDF5 .mat file
            mat_data = scipy.io.loadmat(path_file, squeeze_me=simplify, struct_as_record=not simplify)
            return {key: simplify_mat_structure(value) for key, value in mat_data.items() if not key.startswith('_')} if simplify else mat_data
        except Exception as e:
            raise TypeError(f"Failed to read '{path_file}': {e}")

def simplify_mat_structure(data):
    """
    Recursively processes and simplifies MATLAB data structures.
    
    Converts:
    - HDF5 datasets to NumPy arrays or scalars.
    - HDF5 groups to Python dictionaries.
    - MATLAB structs to Python dictionaries.
    - Cell arrays to Python lists.
    - NumPy arrays are squeezed to remove unnecessary dimensions.
    
    Parameters:
    - data: Input data (HDF5, MATLAB struct, NumPy array, etc.).
    
    Returns:
    - Simplified Python data structure.
    """
    if isinstance(data, h5py.Dataset):
        return data[()]
    elif isinstance(data, h5py.Group):
        return {key: simplify_mat_structure(data[key]) for key in data.keys()}
    elif isinstance(data, scipy.io.matlab.mat_struct):
        return {field: simplify_mat_structure(getattr(data, field)) for field in data._fieldnames}
    elif isinstance(data, np.ndarray):
        if data.dtype == 'object':
            return [simplify_mat_structure(item) for item in data]
        return np.squeeze(data)
    return data

def read_bdf(path_file, preload=True):
    """
    Reads a BioSemi BDF file using MNE.

    Parameters
    ----------
    path_file : str
        Path to the .bdf file.
    preload : bool
        Whether to preload data into memory.

    Returns
    -------
    mne.io.Raw
        MNE Raw object.
    """
    if not os.path.exists(path_file):
        raise FileNotFoundError(f"File not found: {path_file}")

    try:
        return mne.io.read_raw_bdf(
            path_file,
            preload=preload,
            verbose=False
        )
    except Exception as e:
        raise TypeError(f"Failed to read BDF file '{path_file}': {e}")

def read_fif(path_file, preload=True):
    """
    Reads an MNE FIF file.

    Parameters
    ----------
    path_file : str
        Path to the .fif file.
    preload : bool
        Whether to preload data into memory.

    Returns
    -------
    mne.io.Raw
        MNE Raw object.
    """
    if not os.path.exists(path_file):
        raise FileNotFoundError(f"File not found: {path_file}")

    try:
        return mne.io.read_raw_fif(
            path_file,
            preload=preload,
            verbose=False
        )
    except Exception as e:
        raise TypeError(f"Failed to read FIF file '{path_file}': {e}")

def load_file(path_file, simplify=True, preload=True):
    """
    Automatically selects the appropriate reader
    according to the file extension.

    Supported formats:
    - .mat
    - .h5
    - .hdf5
    - .dat
    - .bdf
    - .fif
    """
    if not os.path.exists(path_file):
        raise FileNotFoundError(f"File not found: {path_file}")

    extension = os.path.splitext(path_file)[1].lower()

    readers = {
        ".mat": lambda path: read_mat(path, simplify=simplify),
        ".h5": read_hdf5,
        ".hdf5": read_hdf5,
        ".dat": read_dat,
        ".bdf": lambda path: read_bdf(path, preload=preload),
        ".fif": lambda path: read_fif(path, preload=preload),
    }

    try:
        reader = readers[extension]
    except KeyError:
        raise ValueError(
            f"Unsupported file format '{extension}'. "
            f"Supported formats: {', '.join(readers.keys())}"
        )

    return reader(path_file)

# %% Tools
import re

def get_last_number(text):
    matches = re.findall(r'\d+', text)
    return int(matches[-1]) if matches else None

def get_first_number(text):
    matches = re.findall(r'\d+', text)
    return int(matches[0]) if matches else None