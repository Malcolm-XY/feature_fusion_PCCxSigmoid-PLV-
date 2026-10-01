# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 01:22:29 2026

@author: 18307
"""
import os

from utils import utils_basic_reading
from utils import utils_eeg_loading
from utils import utils_preprocessing

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
                path_save_fold = PathDefinition.CONVERTED[dataset]
                path_save_file = os.path.join(path_save_fold, f"{_identifier}.fif.gz")
                
                folder = os.path.dirname(path_save_file)
                if folder:
                    os.makedirs(folder, exist_ok=True)
                edf.save(path_save_file, overwrite=True)
            
                if verbose:
                    print(f"[INFO] Dataset       : {dataset}")
                    print(f"[INFO] Identifier    : {_identifier}")
                    print(f"[INFO] Channel number: {edf.info["nchan"]}")
                    print(f"[INFO] Sampling Hz   : {edf.info["sfreq"]}")
    
    if save:
        return path_save_file, path_raw_dataset
    else:
        return None, None
           
def preprocessing_and_save_circle(dataset, identifier_1, identifier_2, verbose=True, save=False):
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
    
            # retrieve converted fif
            eeg_converted, path_read_file = utils_eeg_loading.read_eeg_converted(dataset, _identifier, "converted")
            
            # preprocessing
            steps = utils_preprocessing.StepsPreprocessing.hang_on
            eeg_pred, _ = utils_preprocessing.eeg_preprocessing(eeg_converted.copy(), steps, verbose=verbose)
            
            # save
            if save:
                path_save_fold = PathDefinition.PREPROCESSED[dataset]
                path_save_file = os.path.join(path_save_fold, f"{_identifier}.fif.gz")
                
                folder = os.path.dirname(path_save_file)
                if folder:
                    os.makedirs(folder, exist_ok=True)
                eeg_pred.save(path_save_file, overwrite=True)
                
                if verbose:
                    print(f"[INFO] Dataset       : {dataset}")
                    print(f"[INFO] Identifier    : {_identifier}")
                    print(f"[INFO] Channel number: {eeg_pred.info["nchan"]}")
                    print(f"[INFO] Sampling Hz   : {eeg_pred.info["sfreq"]}")
    
    if save:
        return path_save_file, path_read_file
    else:
        return None

def decomposition_and_save_circle(dataset, identifier_1, identifier_2, verbose=True, save=False):
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
    
            # retrieve converted fif
            eeg_converted, path_read_file = utils_eeg_loading.read_eeg_converted(dataset, _identifier, "preprocessed")
            
            # decomposition
            bands_def = utils_preprocessing.DefinationEEGBands.retrieve_by_dataset(dataset)
            eeg_decomposed, _ = utils_preprocessing.eeg_decomposition(eeg_converted.copy(), bands_def, verbose=verbose)
            
            # save
            if save:
                path_save_fold = PathDefinition.DECOMPOSED[dataset]
                # check whether directory exists
                if not os.path.isdir(path_save_fold):
                    os.makedirs(path_save_fold, exist_ok=True)
                    
                for key_band in eeg_decomposed:
                    identifier_compact = f"{_identifier}_{key_band}"
                    path_save_file = os.path.join(path_save_fold, f"{identifier_compact}.fif.gz")
                    
                    eeg_decomposed[key_band].save(path_save_file, overwrite=True)
                    
                    if verbose:
                        print(f"[INFO] Dataset       : {dataset}")
                        print(f"[INFO] Identifier    : {_identifier}")
                        print(f"[INFO] Channel number: {eeg_decomposed[key_band].info["nchan"]}")
                        print(f"[INFO] Sampling Hz   : {eeg_decomposed[key_band].info["sfreq"]}")
            
    if save:
        return path_save_file, path_read_file
    else:
        return None
    
def decomposition_and_save_circle_(dataset_key, subject_range, experiment_range, verbose=True, save=False):
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

# %% Usage
if __name__ == "__main__":
    # %% Validation; Raw dataset; Converted eeg; Preprocessing eeg
    # from utils import utils_validation
    
    # utils_validation.Validation.report()
    # utils_validation.PathDefinition.report()
    
    # path_dataset = utils_validation.PathDefinition.retrive_path("dataset")
    # path_original_eeg = utils_validation.PathDefinition.retrive_path("converted_eeg")
    # path_preprocessed_eeg = utils_validation.PathDefinition.retrive_path("preprocessed_eeg")
    # path_decomposed_eeg = utils_validation.PathDefinition.retrive_path("decomposed")
    
    # # Raw dataset
    # from utils import utils_eeg_loading
    # raw_dataset_sample, _ = utils_eeg_loading.read_eeg_raw_dataset_and_parse("seed", "sub1ex1", "RawEDF") # "ndarray")
    # raw_dataset_sample.plot()
    
    # # Converted eeg
    # path_save_file_sample, path_read_file_sample = converting_and_save_circle("seed", "sub1ex1", "sub2ex1", verbose=True, save=False)
    
    # # Preprocessing eeg
    # path_save_file_sample, path_read_file_sample = preprocessing_and_save_circle("seed", "sub1ex1", "sub2ex1", verbose=True, save=False)
    
    # %% Decomposition
    path_save_file, path_read_file = decomposition_and_save_circle("seed", "sub1ex1", "sub15ex3", verbose=True, save=True)
    # path_save_file, path_read_file = decomposition_and_save_circle("deap", "sub1ex1", "sub23ex1", verbose=True, save=True)
    # path_save_file, path_read_file = decomposition_and_save_circle("dreamer", "sub1ex1", "sub32ex1", verbose=True, save=True)