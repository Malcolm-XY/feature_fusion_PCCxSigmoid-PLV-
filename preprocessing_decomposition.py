# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 01:22:29 2026

@author: 18307
"""
import os

from utils import utils_basic_reading
from utils import utils_eeg_loading
# from utils import utils_preprocessing

from utils.utils_validation import Validation, PathDefinition
def converting_and_save_circle(dataset, identifier_1, identifier_2, verbose=True, save=False):
    # validation
    dataset = Validation.validate_dataset(dataset)
    identifier_1 = Validation.validate_identifier(identifier_1)
    identifier_2 = Validation.validate_identifier(identifier_2)
    
    subject_range = range(utils_basic_reading.get_first_number(identifier_1), 
                          utils_basic_reading.get_first_number(identifier_2) + 1)
    experiment_range = range(utils_basic_reading.get_last_number(identifier_1),
                             utils_basic_reading.get_last_number(identifier_2) + 1)
    
    for subject in subject_range:
        for experiment in experiment_range:
            _identifier = f"sub{subject}ex{experiment}"
            print(f"Processing: {_identifier}.")
            
            # retrieve RawEDF
            edf, path_raw_dataset = utils_eeg_loading.read_eeg_raw_dataset_and_parse(dataset, _identifier, "RawEDF")
            
            # save
            if save:
                path_save_fold = PathDefinition.PREPROCESSED[dataset]
                path_save_file = os.path.join(path_save_fold, f"{_identifier}.fif.gz")
                
                edf.save(path_save_file, overwrite=True)
            
                if verbose:
                    print(f"[INFO] Dataset       : {dataset}")
                    print(f"[INFO] Identifier    : {_identifier}")
                    print(f"[INFO] Channel number: {edf.info["nchan"]}")
                    print(f"[INFO] Sampling Hz   : {edf.info["sfreq"]}")
    
    if save:
        return path_save_file, path_raw_dataset
    else:
        return None
        
def converting_and_save_circle_(dataset_key, subject_range, experiment_range, verbose=True, save=False):
    # Prune inputs
    dataset_key = dataset_key.lower()
    
    valid_dataset_keys = ["seed", "dreamer", "deap"]
    
    if dataset_key not in valid_dataset_keys:
        raise ValueError(f"{dataset_key} is not a valid dataset. Valid datasets are: {valid_dataset_keys}")
    
    path_save_files = []
    if dataset_key in valid_dataset_keys and subject_range is not None and experiment_range is not None:
        for subject in subject_range:
            for experiment in experiment_range:
                identifier = f"sub{subject}ex{experiment}"
                print(f"Processing: {identifier}.")
                
                # Retrieve eeg
                _, eeg_mne, path_read_file = utils_eeg_loading.read_mne_origin(dataset_key, identifier, 
                                                                               verbose=verbose)
                
                # Save
                if save:
                    identifier_save = f"sub{subject}ex{experiment}"
                    path_save_file = utils_eeg_loading.save_eeg_mne(eeg_mne, dataset_key, 
                                                                    identifier_save, item="original")
                    path_save_files.append(path_save_file)
    
    return path_save_files
    
def preprocessing_and_save_circle(dataset_key, subject_range, experiment_range, verbose=True, save=False):
    # Prune inputs
    dataset_key = dataset_key.lower()
    
    valid_dataset_keys = ["seed", "dreamer", "deap"]
    
    if dataset_key not in valid_dataset_keys:
        raise ValueError(f"{dataset_key} is not a valid dataset. Valid datasets are: {valid_dataset_keys}")
    
    path_save_files = []
    if dataset_key in valid_dataset_keys and subject_range is not None and experiment_range is not None:
        for subject in subject_range:
            for experiment in experiment_range:
                identifier = f"sub{subject}ex{experiment}"
                print(f"Processing: {identifier}.")
                
                # Retrieve converted eeg
                eeg_converted, _ = utils_eeg_loading.read_eeg_mne(dataset_key, identifier, 
                                                                  "original", verbose=False)
                
                # Preprocessing
                steps = utils_preprocessing.StepsPreprocessing.retrieve(dataset_key)
                eeg_pred, _ = utils_preprocessing.eeg_preprocessing(eeg_converted.copy(), 
                                                                    steps, verbose=verbose)
                
                # Save
                if save:
                    identifier_save = f"sub{subject}ex{experiment}"
                    path_save_file = utils_eeg_loading.save_eeg_mne(eeg_pred, dataset_key, 
                                                                    identifier_save, item="preprocessed")
                    path_save_files.append(path_save_file)

    return path_save_files
    
def decomposition_and_save_circle(dataset_key, subject_range, experiment_range, verbose=True, save=False):
    # Prune inputs
    dataset_key = dataset_key.lower()
    
    valid_dataset_keys = ["seed", "dreamer", "deap"]
    
    if dataset_key not in valid_dataset_keys:
        raise ValueError(f"{dataset_key} is not a valid dataset. Valid datasets are: {valid_dataset_keys}")
    
    path_save_files = []
    if dataset_key in valid_dataset_keys and subject_range is not None and experiment_range is not None:
        for subject in subject_range:
            for experiment in experiment_range:
                identifier = f"sub{subject}ex{experiment}"
                print(f"Processing: {identifier}.")
                
                # Retrieve converted eeg
                eeg_pred, _ = utils_eeg_loading.read_eeg_mne(dataset_key, identifier, 
                                                             "preprocessed", verbose=False)
                
                bands_def = utils_preprocessing.DefinationEEGBands.retrieve_by_dataset(dataset_key)
                eeg_decomposed = utils_preprocessing.eeg_decomposition(eeg_pred.copy(), bands_def, verbose=verbose)
                
                if save:
                    for key_band in eeg_decomposed:
                        identifier_compact = f"{identifier}_{key_band}"
                        path_save_file = utils_eeg_loading.save_eeg_mne(eeg_decomposed[key_band], dataset_key, 
                                                                        identifier_compact, item="decomposed")
                        path_save_files.append(path_save_file)
    
    return path_save_files

    # %% Converting
if __name__ == "__main__":
    # path_save_files_seed = converting_and_save_circle("seed", range(1,2), range(1,4), verbose=True, save=False)
    # path_save_files_dreamer = converting_and_save_circle("dreamer", range(1,2), range(1,2), verbose=True, save=False)
    # path_save_files_deap = converting_and_save_circle("deap", range(1,2), range(1,2), verbose=True, save=False)
    
    # Read validation
    # _, eeg_origin, path_origin = utils_eeg_loading.read_mne_origin("deap", "sub1ex1", verbose=True)
    # eeg_converted, path_converted = utils_eeg_loading.read_eeg_mne("deap", "sub1ex1", "original", verbose=True)
    
    # %% Preprocessing; Work from here
    # path_save_files = preprocessing_and_save_circle("seed", range(1,2), range(1,2), verbose=True, save=False)
    # path_save_files = preprocessing_and_save_circle("dreamer", range(1,2), range(1,2), verbose=True, save=False)    
    path_save_files = preprocessing_and_save_circle("deap", range(1,2), range(1,2), verbose=True, save=False)
    
    # %% Decomposition
    # path_save_files = decomposition_and_save_circle("seed", range(1,2), range(1,2), verbose=True, save=False)
    # path_save_files = decomposition_and_save_circle("dreamer", range(1,2), range(1,2), verbose=True, save=False)
    path_save_files = decomposition_and_save_circle("deap", range(1,2), range(1,2), verbose=True, save=False)
    
    # Read validation
    # eeg_decomposed, path_decomposed = utils_eeg_loading.read_eeg_mne("seed", "sub1ex1"+"_"+"alpha", "decomposed", verbose=True)
    # eeg_decomposed, path_decomposed = utils_eeg_loading.read_eeg_mne("dreamer", "sub1ex1"+"_"+"alpha", "decomposed", verbose=True)
    # eeg_decomposed, path_decomposed = utils_eeg_loading.read_eeg_mne("deap", "sub1ex1"+"_"+"alpha", "decomposed", verbose=True)
    
    