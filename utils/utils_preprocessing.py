# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 23:48:24 2026

@author: 18307
"""

import numpy as np

# %% Attributes
class StepsPreprocessing:
    general_steps_seed = {
        "bad chs handling": None,
        "re-reference": None, # None or "CAR"
        "band-pass": (0.3, 99),
        "notch": 50
    }

    general_steps_dreamer = {
        "bad chs handling": None,
        "re-reference": None, # None or "CAR"
        "band-pass": (0.3, 60),
        "notch": 50
    }
    
    general_steps_deap = {
        "bad chs handling": None,
        "re-reference": None,
        "band-pass": (0.3, 99),
        "notch": 50
    }
    
    @staticmethod
    def retrieve(dataset):
        dataset = dataset.lower()

        if dataset == "seed":
            return StepsPreprocessing.general_steps_seed
        elif dataset == "dreamer":
            return StepsPreprocessing.general_steps_dreamer
        elif dataset == "deap":
            return StepsPreprocessing.general_steps_deap
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

class DefinationEEGBands:
    fre_bands_higher = {
        "delta": (0.5, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
        "gamma": (30, 99)
    }
    
    fre_bands_lower = {
        "delta": (0.5, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
        "gamma": (30, 60)
        }
    
    fre_bands_classical = {
        "delta": (0.5, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
        "gamma": (30, 50)
        }
    
    @staticmethod
    def retrieve(bands_def):
        bands_def = bands_def.lower()

        if bands_def == "higher":
            return DefinationEEGBands.fre_bands_higher
        elif bands_def == "lower":
            return DefinationEEGBands.fre_bands_lower
        elif bands_def == "classical":
            return DefinationEEGBands.fre_bands_classical
        else:
            raise ValueError(f"Unknown Defination: {bands_def}")
    
    @staticmethod
    def retrieve_by_dataset(dataset):
        dataset = dataset.lower()

        if dataset == "seed":
            return DefinationEEGBands.fre_bands_higher
        elif dataset == "dreamer":
            return DefinationEEGBands.fre_bands_lower
        elif dataset == "deap":
            return DefinationEEGBands.fre_bands_higher
        else:
            raise ValueError(f"Unknown Defination: {dataset}")
    
# %% Preprocessing
def eeg_preprocessing(eeg_mne, steps, verbose=False):
    eeg_mne_origin = eeg_mne.copy()
    sfreq = eeg_mne.info['sfreq']
    
    if verbose:
        print(f"[INFO] Shape       : {eeg_mne.get_data().shape}")
        print(f"[INFO] Sampling Hz : {sfreq}")
        # eeg_mne.plot(title="Raw EEG before preprocessing", scalings="auto")
        eeg_mne.plot()
    
    bad_ch_handling = steps.get("bad chs handling", None)
    re_reference = steps.get("re-reference", None)
    band_pass = steps.get("band-pass", None)
    notch = steps.get("notch", None)
    
    # 1) Bad channel handling
    # ------------------------------------------------------------------
    if bad_ch_handling is not None:
        if bad_ch_handling == "auto":
            raise NotImplementedError(
                "Automatic bad-channel detection is not implemented yet."
            )
            
    # 2) Re-reference
    # ------------------------------------------------------------------
    if re_reference is not None:
        # Case 1: CAR
        if isinstance(re_reference, str) and re_reference.upper() == "CAR":
            eeg_mne.set_eeg_reference(ref_channels="average", verbose="ERROR")
            if verbose:
                print("[INFO] Applied common average reference (CAR).")
    
        # Case 2: custom reference channels (e.g., linked ears)
        elif isinstance(re_reference, (list, tuple)):
            invalid = [ch for ch in re_reference if ch not in eeg_mne.ch_names]
            if len(invalid) > 0:
                raise ValueError(f"Unknown reference channels: {invalid}")
    
            eeg_mne.set_eeg_reference(ref_channels=list(re_reference), verbose="ERROR")
    
            if verbose:
                print(f"[INFO] Applied custom reference: {list(re_reference)}")
    
        else:
            raise ValueError(
                '"re-reference" must be None, "CAR", or a list of channel names.'
            )
    
    # 3) Band-pass filtering
    # ------------------------------------------------------------------
    if band_pass is not None:
        if (
            not isinstance(band_pass, (list, tuple))
            or len(band_pass) != 2
        ):
            raise ValueError('"band-pass" must be None or a tuple/list like (low, high).')

        low_freq, high_freq = band_pass

        if low_freq is None and high_freq is None:
            pass
        else:
            eeg_mne.filter(l_freq=low_freq, h_freq=high_freq, method="fir", phase="zero-double", verbose="ERROR")

            if verbose:
                print(f"[INFO] Applied band-pass filter: ({low_freq}, {high_freq}) Hz")
    
    # 4) Notch filtering
    # ------------------------------------------------------------------
    if notch is not None:
        if isinstance(notch, (int, float)):
            notch_freqs = [float(notch)]
        elif isinstance(notch, (list, tuple, np.ndarray)):
            notch_freqs = [float(f) for f in notch]
        else:
            raise TypeError('"notch" must be None, a number, or a list/tuple of numbers.')
    
        nyquist = sfreq / 2.0
        valid_notch_freqs = [f for f in notch_freqs if 0 < f < nyquist]
    
        if len(valid_notch_freqs) > 0:
            eeg_mne.notch_filter(
                freqs=valid_notch_freqs,
                method="fir",
                phase="zero-double",
                verbose="ERROR"
            )
    
            if verbose:
                print(f"[INFO] Applied notch filter at: {valid_notch_freqs} Hz")
        elif verbose:
            print(
                f"[INFO] Notch skipped because all requested frequencies "
                f"are outside valid range (< Nyquist={nyquist:.2f} Hz)."
            )
    
    if verbose:
        print("[INFO] Preprocessing finished.")
        # eeg_mne.plot(title="Preprocessed EEG", scalings="auto")
        eeg_mne.plot()
    
    return eeg_mne, eeg_mne_origin

# %% Decomposition
def eeg_decomposition(eeg_mne, bands_def, verbose=False):
    band_filtered_eeg = {}
        
    # Filter EEG data for each frequency band
    for band, (low_freq, high_freq) in bands_def.items():
        filtered_eeg = eeg_mne.copy().filter(l_freq=low_freq, h_freq=high_freq, method="fir", phase="zero-double")
        band_filtered_eeg[band] = filtered_eeg
        if verbose:
            print(f"{band} band filtered: {low_freq}–{high_freq} Hz")
            # filtered_eeg.plot(title="Decomposed EEG", scalings="auto")
            filtered_eeg.plot()
    
    return band_filtered_eeg

    # %% Test; SEED
if __name__ == "__main__":
    # from . import utils_eeg_loading
    print("utils_proprocessing")
    # MNE data
    # eeg_seed, eeg_seed_mne, path_file = utils_eeg_loading.read_eeg_mne("seed", "sub1ex1", verbose=False)
    
    # # Preprocessing
    # steps = StepsPreprocessing.retrieve("seed")
    # eeg_ori, eeg_pred = eeg_preprocessing(eeg_seed_mne.copy(), steps, verbose=True)
    
    # df = eeg_pred.to_data_frame()
    
    # # Decomposition
    # bands_def = DefinationEEGBands.retrieve("higher")
    # eeg_decomposed = eeg_decomposition(eeg_pred.copy(), bands_def, verbose=True)
    # alpha, beta, gamma = eeg_decomposed["alpha"], eeg_decomposed["beta"], eeg_decomposed["gamma"]
    
    # %% Test; DREAMER
    # # MNE data
    # eeg_dreamer, eeg_dreamer_mne, path_file = utils_eeg_loading.read_eeg_mne("dreamer", "sub1", verbose=False)
    
    # # Preprocessing
    # steps = StepsPreprocessing.retrieve("dreamer")
    # eeg_ori, eeg_pred = eeg_preprocessing(eeg_dreamer_mne.copy(), steps, verbose=True)
    
    # df = eeg_pred.to_data_frame()
    
    # # Decomposition
    # bands_def = DefinationEEGBands.retrieve("lower")
    # eeg_decomposed = eeg_decomposition(eeg_pred.copy(), bands_def, verbose=True)
    # alpha, beta, gamma = eeg_decomposed["alpha"], eeg_decomposed["beta"], eeg_decomposed["gamma"]
    
    # %% Test; DEAP
    # eeg_deap, eeg_deap_mne, path_file = utils_eeg_loading.read_eeg_mne("deap", "s02", verbose=False)
    
    # # Preprocessing
    # steps = StepsPreprocessing.retrieve("deap")
    # eeg_ori, eeg_pred = eeg_preprocessing(eeg_deap_mne.copy(), steps, verbose=True)
    
    # df = eeg_pred.to_data_frame()
    
    # # Decomposition
    # bands_def = DefinationEEGBands.retrieve("higher")
    # eeg_decomposed = eeg_decomposition(eeg_pred.copy(), bands_def, verbose=True)
    # alpha, beta, gamma = eeg_decomposed["alpha"], eeg_decomposed["beta"], eeg_decomposed["gamma"]