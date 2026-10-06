# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 01:22:29 2026

@author: 18307

Data Flow:
DATASET (preprocessed in most cases)
->eeg_converted
->eeg_preprocessed
->eeg_decomposed
->functional_connectivity
/->channel_features
"""
import os

from utils.utils_validation import Validation, PathDefinition

from utils import utils_basic_reading
from utils import utils_eeg_loading
from utils import utils_preprocessing
from utils import utils_interaction

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
                path_save_file = os.path.join(path_save_fold, f"{_identifier}_eeg.fif.gz")
                
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
            eeg_converted, path_read_file = utils_eeg_loading.read_eeg_converted(dataset, _identifier, "eeg_converted")
            
            # preprocessing
            steps = utils_preprocessing.StepsPreprocessing.hang_on
            eeg_pred, _ = utils_preprocessing.eeg_preprocessing(eeg_converted.copy(), steps, verbose=verbose)
            
            # save
            if save:
                path_save_fold = PathDefinition.PREPROCESSED[dataset]
                path_save_file = os.path.join(path_save_fold, f"{_identifier}_eeg.fif.gz")
                
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
            eeg_converted, path_read_file = utils_eeg_loading.read_eeg_converted(dataset, _identifier, "eeg_preprocessed")
            
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
                    path_save_file = os.path.join(path_save_fold, f"{identifier_compact}_eeg.fif.gz")
                    
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
    
# %% Usage
if __name__ == "__main__":
    # %% Validation
    print("Validation")
    # from utils import utils_validation
    
    # utils_validation.Validation.report()
    # utils_validation.PathDefinition.report()
    
    # path_dataset = utils_validation.PathDefinition.retrive_path("dataset")
    # path_original_eeg = utils_validation.PathDefinition.retrive_path("converted_eeg")
    # path_preprocessed_eeg = utils_validation.PathDefinition.retrive_path("preprocessed_eeg")
    # path_decomposed_eeg = utils_validation.PathDefinition.retrive_path("decomposed")
    
    # %% Raw dataset; Converted eeg; Preprocessing eeg
    print("Converting")
    # raw_dataset_sample, _ = utils_eeg_loading.read_eeg_raw_dataset_and_parse(
    #     "seed", "sub1ex1", "RawEDF") # "ndarray")
    # raw_dataset_sample.plot()
    
    # Converted eeg
    path_save_file_sample, path_read_file_sample = converting_and_save_circle(
        "dreamer", "sub1ex1", "sub23ex1", verbose=True, save=True)
    
    # Preprocessing eeg
    path_save_file_sample, path_read_file_sample = preprocessing_and_save_circle(
        "dreamer", "sub1ex1", "sub23ex1", verbose=True, save=True)
    
    # %% Decomposition
    print("Decomposition")
    path_save_file, path_read_file = decomposition_and_save_circle(
        "dreamer", "sub1ex1", "sub23ex1", verbose=True, save=True)
    
    # %% Reading; Correspondance check
    # Raw dataset (.mat, ......)->Converted EEG (RawEDF)->Preprocessed EEG (RawEDF)->Decomposed EEG (RawEDF)
    # print("Correspondance check")

    # eeg_raw_sample, path_0 = utils_eeg_loading.read_eeg_raw_dataset_and_parse(
    #     "seed", "sub1ex1", "RawEDF")

    # eeg_converted_sample, path_1 = utils_eeg_loading.read_eeg_converted(
    #     "seed", "sub1ex1", "converted")

    # eeg_preprocessed_sample, path_2 = utils_eeg_loading.read_eeg_converted(
    #     "seed", "sub1ex1", "preprocessed")

    # eeg_decomposed_sample, path_3 = utils_eeg_loading.read_eeg_decomposed(
    #     "seed", "sub1ex1", return_type="RawEDF")
    # eeg_decomposed_sample_a = eeg_decomposed_sample["alpha"]
    # eeg_decomposed_sample_b = eeg_decomposed_sample["beta"]
    # eeg_decomposed_sample_g = eeg_decomposed_sample["gamma"]

    # eeg_raw_sample.plot()
    # eeg_converted_sample.plot()
    # eeg_preprocessed_sample.plot()

    # eeg_decomposed_sample_a.plot()
    # eeg_decomposed_sample_b.plot()
    # eeg_decomposed_sample_g.plot()
    
    # End program actions
    utils_interaction.end_program_actions(play_sound=True, shutdown=False, countdown_seconds=30)
