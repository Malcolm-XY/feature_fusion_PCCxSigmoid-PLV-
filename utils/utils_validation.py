# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 15:38:00 2026

@author: 18307
"""

import os
import re
import warnings

from collections.abc import Mapping

class Validation:
    DATASETS = ("seed", "deap", "dreamer")
    FEATURES = ("pcc", "plv", "mi", "pli", "wpli", "dpli", "sdpli")
    BANDS = ("joint", "theta", "delta", "alpha", "beta", "gamma")
    
    FILE_TYPES = ('pandas_dataframe', 'numpy_array', 'mne', 'fif')
    
    SAMPLING_RATES = {"SEED": 200, "DEAP": 128, "DREAMER": 128}
    
    IDENTIFIER_PATTERN = re.compile(r"^sub(?P<subject>\d+)ex(?P<experiment>\d+)$", re.IGNORECASE)
    IDENTIFIER_PATTERN_1 = re.compile(r"^s(?P<subject>\d+)$", re.IGNORECASE)
    
    @classmethod
    def report(cls):
        print("Validation configuration")
        print("-" * 40)
    
        print(f"Datasets      : {', '.join(cls.DATASETS)}")
        print(f"Features      : {', '.join(cls.FEATURES)}")
        print(f"Bands         : {', '.join(cls.BANDS)}")
        print(f"File types    : {', '.join(cls.FILE_TYPES)}")
    
        print("\nSampling rates:")
        for dataset, rate in cls.SAMPLING_RATES.items():
            print(f"  {dataset:<10}: {rate} Hz")
    
        print("\nIdentifier format:")
        print("  sub<number>ex<number>")
        print("  Example: sub1ex2")
        
        print("-" * 40, "\n")
        
    @staticmethod
    def _validate(value, reference, term):
        if value not in reference:
            raise ValueError(f"Invalid {term} '{value}'. "f"Supported {term}s: {', '.join(reference)}")
        return value
    
    @classmethod
    def validate_dataset(cls, value:str):
        value = value.lower()
        return cls._validate(value, cls.DATASETS, "DATASET")
        
    @classmethod
    def validate_feature(cls, value:str):
        value = value.lower()
        return cls._validate(value, cls.FEATURE, "FEATURE")
    
    @classmethod
    def validate_bands(cls, value:str):
        value = value.lower()
        return cls._validate(value, cls.BANDS, "BANDS")
    
    @classmethod
    def validate_file_type(cls, value:str):
        value = value.lower()
        return cls._validate(value, cls.FILE_TYPES, "FILE_TYPES")
    
    @classmethod
    def validate_identifier(cls, value: str or None):
        if value is None:
            warnings.warn("Nonetype 'Identifier' detected in 'Validation' procedure")
            return None
        
        _mark = 0
        for pattern in [cls.IDENTIFIER_PATTERN, cls.IDENTIFIER_PATTERN_1]:
            _match = pattern.fullmatch(value)
            if _match is not None:
                return value.lower()
            elif _match is None:
                _mark +=1
                
        if _mark > 0:
            raise ValueError(f"Invalid identifier '{value}'. " 
                             "Expected format: 'sub<number>ex<number>', " 
                             "for example 'sub1ex2'.")
    
    @classmethod
    def retrive_sampling_rate(cls, dataset:str):
        return cls.SAMPLING_RATES[dataset]

class PathDefinition:
    # global ROOT
    # global DATASET_ROOT
    # global ORIGINAL
    # global PREPROCESSED
    # global DECOMPOSED
    
    _BASE = os.path.dirname(os.path.abspath(__file__))
    
    ROOT = {
        "data": os.path.abspath(os.path.join(_BASE, "../../../Research_Data")),
        "code": os.path.abspath(os.path.join(_BASE, "../../../Research_Engineering")),
    }

    # Dataset=================================================
    DATASET_ROOT = {"seed": os.path.join(ROOT["data"], "SEED_test"),
                    "dreamer": os.path.join(ROOT["data"], "DREAMER_test"),
                    "deap": os.path.join(ROOT["data"], "DEAP_test"),}

    DATASET = {name: os.path.join(root, "DATASET") for name, root in DATASET_ROOT.items()}

    ORIGINAL = {name: os.path.join(root, "eeg_original") for name, root in DATASET_ROOT.items()}

    PREPROCESSED = {name: os.path.join(root, "eeg_preprocessed") for name, root in DATASET_ROOT.items()}

    DECOMPOSED = {name: os.path.join(root, "eeg_decomposed") for name, root in DATASET_ROOT.items()}
    
    # Raw dataset/Original dataset=============================
    _RAW_DATASET_CONFIG = {
        "seed": "sub{argument_1}ex{argument_2}.mat",
        "dreamer": "DREAMER.mat",
        "deap": "s{argument_1:02d}.bdf",
    }
    
    @classmethod
    def retrive_raw_dataset(cls, dataset, argument_1=None, argument_2=None):
        dataset_path = cls.retrive_path("dataset", dataset)
    
        filename = cls._RAW_DATASET_CONFIG[dataset].format(
            argument_1=argument_1,
            argument_2=argument_2
        )
    
        return os.path.join(dataset_path, filename)
        
    # Additional information===================================
    # global ELECTRODE_DISTRIBUTIONS
    
    ELECTRODE_DISTRIBUTIONS = {name: os.path.join(root, "electrode_distributions")
                               for name, root in DATASET_ROOT.items()}
    
    _ELECTRODE_CONFIG = {"seed": 62, "dreamer": 14, "deap": 32,}
    
    ELECTRODE_DISTRIBUTION_FILE = {}
    for name, channels in _ELECTRODE_CONFIG.items():
        ELECTRODE_DISTRIBUTION_FILE.update({
            name: os.path.join(ELECTRODE_DISTRIBUTIONS[name],
            f"biosemi64_{channels}_channels_original_distribution.txt")})

    ELECTRODE_DISTRIBUTION_FILE_MANUAL = {}
    for name, channels in _ELECTRODE_CONFIG.items():
        ELECTRODE_DISTRIBUTION_FILE_MANUAL.update({
            name: os.path.join(ELECTRODE_DISTRIBUTIONS[name],
            f"biosemi64_{channels}_channels_manual_distribution.txt")})

    # Features=================================================
    # global FEATURE_ROOT
    # global FEATURES
    # global FUNCTIONAL_CONNECTIVITY
    # global CHANNEL_FEATURE
    
    FEATURE_ROOT = {
        "functional_connectivity": {
            name: os.path.join(root, "functional_connectivity")
            for name, root in DATASET_ROOT.items()},

        "channel_features": {
            name: os.path.join(root, "channel_features")
            for name, root in DATASET_ROOT.items()},
    }
    
    # Funtional connectivity configuration
    _FUNCTIONAL_CONNECTIVITY_CONFIG = {"pcc": "pcc_h5", "plv": "plv_h5", 
                                       "pli": "pli_h5", "wpli": "wpli_h5",
                                       "dpli": "dpli_h5", "sdpli": "sdpli_h5",
                                       "mi":   "mi_h5",}
    
    _DATASET_LIST =  Validation.DATASETS # ["seed", "dreamer", "deap"]
    
    FUNCTIONAL_CONNECTIVITY = {}
    for feature, directoty in _FUNCTIONAL_CONNECTIVITY_CONFIG.items():
        for dataset in _DATASET_LIST:
            if FUNCTIONAL_CONNECTIVITY.get(feature) is None:
                FUNCTIONAL_CONNECTIVITY[feature] = {}
            
            FUNCTIONAL_CONNECTIVITY[feature].update({
                dataset: os.path.join(FEATURE_ROOT["functional_connectivity"][dataset], 
                                  directoty)})
    
    # Channel feature configuration
    _CHANNEL_FEATURE_CONFIG = {"psd": "psd_h5", 
                               "de": "de_h5",
                               "psd_lds": "psd_lds_h5", 
                               "de_lds": "de_lds_h5",}
    
    CHANNEL_FEATURES = {}
    for feature, directoty in _CHANNEL_FEATURE_CONFIG.items():
        for dataset in _DATASET_LIST:
            if CHANNEL_FEATURES.get(feature) is None:
                CHANNEL_FEATURES[feature] = {}
            
            CHANNEL_FEATURES[feature].update({
                dataset: os.path.join(FEATURE_ROOT["channel_features"][dataset], 
                                  directoty)})
            
    # Designated feature files=================================
    # global FUNCTIONAL_CONNECTIVITY_DESIGNATE
    # global CHANNEL_FEATURES_DESIGNATE
    
    _GLOBAL_AVERAGE_CONFIG = {"seed": 5, "dreamer": 8, "deap": 10,}

    FUNCTIONAL_CONNECTIVITY_DESIGNATE = {}
    for feature, directoty in _FUNCTIONAL_CONNECTIVITY_CONFIG.items():
        for dataset, number in _GLOBAL_AVERAGE_CONFIG.items():
            if FUNCTIONAL_CONNECTIVITY_DESIGNATE.get(feature) is None:
                FUNCTIONAL_CONNECTIVITY_DESIGNATE[feature] = {}
            
            FUNCTIONAL_CONNECTIVITY_DESIGNATE[feature].update({
                dataset: os.path.join(FEATURE_ROOT["functional_connectivity"][dataset], 
                                  directoty, f"globally_average_{number}.h5")})
            
    CHANNEL_FEATURES_DESIGNATE = {}
    for feature, directoty in _FUNCTIONAL_CONNECTIVITY_CONFIG.items():
        for dataset, number in _GLOBAL_AVERAGE_CONFIG.items():
            if CHANNEL_FEATURES_DESIGNATE.get(feature) is None:
                CHANNEL_FEATURES_DESIGNATE[feature] = {}
            
            CHANNEL_FEATURES_DESIGNATE[feature].update({
                dataset: os.path.join(FEATURE_ROOT["channel_features"][dataset], 
                                  directoty, f"globally_average_{number}.h5")})
    
    @staticmethod
    def _normalize_path(path):
        """
        Return a normalized absolute filesystem path.
    
        On Windows, ``os.path.normcase`` also normalizes capitalization,
        so differently-cased representations of the same path are handled
        consistently.
    
        Examples
        --------
        Windows:
            F:\\RnD_Repo\\Project
            f:\\rnd_repo\\project
    
        both normalize to the same representation.
        """
        if path is None:
            return None
    
        path = os.fspath(path)
    
        return os.path.normcase(
            os.path.normpath(
                os.path.abspath(path)
            )
        )
    
    @classmethod
    def _path_status(cls, path):
        """
        Return a human-readable status for a filesystem path.
    
        The path is normalized before checking so that capitalization and
        path formatting are handled consistently across the reports.
        """
        if path is None:
            return "MISSING"
    
        path = cls._normalize_path(path)
    
        if os.path.isfile(path):
            return "FILE"
    
        if os.path.isdir(path):
            return "DIR"
    
        if os.path.exists(path):
            return "EXISTS"
        
        return "MISSING"
      
    @staticmethod
    def _flatten_paths(mapping, prefix=""):
        """
        Recursively flatten nested mappings.
    
        Example
        -------
        {
            "pcc": {
                "seed": "/path/to/seed"
            }
        }
    
        becomes:
    
            ("pcc.seed", "/path/to/seed")
        """
        if not isinstance(mapping, Mapping):
            yield prefix, mapping
            return
    
        for key, value in mapping.items():
            name = f"{prefix}.{key}" if prefix else str(key)
    
            if isinstance(value, Mapping):
                yield from PathDefinition._flatten_paths(
                    value,
                    prefix=name,
                )
            else:
                yield name, value
    
    @classmethod
    def report(cls):
        """
        Print a compact overview of the main project/data paths.
    
        Paths are normalized before being displayed so that Windows path
        capitalization is consistent throughout the report.
        """
        sections = {
            "ROOT": cls.ROOT,
            "DATASET_ROOT": cls.DATASET_ROOT,
            "DATASET": cls.DATASET,
            "ORIGINAL": cls.ORIGINAL,
            "PREPROCESSED": cls.PREPROCESSED,
            "DECOMPOSED": cls.DECOMPOSED,
            "FUNCTIONAL_CONNECTIVITY": cls.FUNCTIONAL_CONNECTIVITY,
            "CHANNEL_FEATURES": cls.CHANNEL_FEATURES,
            "FUNCTIONAL_CONNECTIVITY_DESIGNATE": cls.FUNCTIONAL_CONNECTIVITY_DESIGNATE,
            "CHANNEL_FEATURES_DESIGNATE": cls.CHANNEL_FEATURES_DESIGNATE,
        }
    
        print("=" * 80)
        print("PATH DEFINITION REPORT")
        print("=" * 80)
    
        base = cls._normalize_path(cls._BASE)
    
        print(f"Base directory : {base}")
        print()
    
        total = 0
        existing = 0
    
        for section_name, mapping in sections.items():
            print(f"[{section_name}]")
    
            for name, path in cls._flatten_paths(mapping):
                normalized_path = cls._normalize_path(path)
                
                status = cls._path_status(normalized_path)
    
                total += 1
    
                if status != "MISSING":
                    existing += 1
    
                marker = "✓" if status != "MISSING" else "✗"
    
                print(
                    f"  {marker} " f"{name:<8} "
                    f"[{status}] " # f"[{status:<7}] "
                    f"{normalized_path}"
                )
    
            print()
    
        print("-" * 80)
        print(
            f"Summary: {existing}/{total} configured paths exist "
            f"({total - existing} missing)"
        )
        print("=" * 80)
    
    @classmethod
    def report_detailed(cls):
        """
        Print all configured paths, including nested feature paths.
    
        Paths are normalized before checking and displaying them so that
        different Windows capitalization does not produce inconsistent
        report output.
        """
        sections = {
            "ROOT": cls.ROOT,
            "DATASET_ROOT": cls.DATASET_ROOT,
            "DATASET": cls.DATASET,
            "ORIGINAL": cls.ORIGINAL,
            "PREPROCESSED": cls.PREPROCESSED,
            "DECOMPOSED": cls.DECOMPOSED,
            "ELECTRODE_DISTRIBUTIONS": cls.ELECTRODE_DISTRIBUTIONS,
            "ELECTRODE_DISTRIBUTION_FILE":
                cls.ELECTRODE_DISTRIBUTION_FILE,
            "ELECTRODE_DISTRIBUTION_FILE_MANUAL":
                cls.ELECTRODE_DISTRIBUTION_FILE_MANUAL,
            "FUNCTIONAL_CONNECTIVITY": cls.FUNCTIONAL_CONNECTIVITY,
            "CHANNEL_FEATURES": cls.CHANNEL_FEATURES,
            "FUNCTIONAL_CONNECTIVITY_DESIGNATE": cls.FUNCTIONAL_CONNECTIVITY_DESIGNATE,
            "CHANNEL_FEATURES_DESIGNATE": cls.CHANNEL_FEATURES_DESIGNATE,
        }
    
        print("=" * 100)
        print("DETAILED PATH DEFINITION REPORT")
        print("=" * 100)
        print("Python file base directory:")
        print(f"  {cls._normalize_path(cls._BASE)}")
        print()
    
        total = 0
        files = 0
        directories = 0
        other_existing = 0
        missing = 0
    
        for section_name, mapping in sections.items():
            print(f"[{section_name}]")
            print("-" * 100)
    
            for name, path in cls._flatten_paths(mapping):
                normalized_path = cls._normalize_path(path)
                status = cls._path_status(normalized_path)
    
                total += 1
    
                if status == "FILE":
                    files += 1
                    marker = "✓"
    
                elif status == "DIR":
                    directories += 1
                    marker = "✓"
    
                elif status == "EXISTS":
                    other_existing += 1
                    marker = "✓"
    
                else:
                    missing += 1
                    marker = "✗"
    
                print(
                    f"{marker} "
                    f"{name:<10} "
                    f"{status:<8} "
                    f"{normalized_path}"
                )
    
            print()
    
        print("=" * 100)
        print("SUMMARY")
        print("-" * 100)
        print(f"Configured paths : {total}")
        print(f"Directories      : {directories}")
        print(f"Files            : {files}")
    
        if other_existing:
            print(f"Other existing   : {other_existing}")
    
        print(f"Missing          : {missing}")
        print("=" * 100)
        
    @classmethod
    def retrive_path(cls, argument, dataset=None, feature=None):
        """
        Retrieve a configured path.
    
        Parameters
        ----------
        argument : str
            Path category. Supported values:
            - "dataset"
            - "original_eeg"
            - "preprocessed_eeg"
            - "decomposed"
            - "functional_connectivity"
            - "channel_features"
            - "functional_connectivity_average"
            - "channel_features_average"
    
        dataset : str, optional
            Dataset name, e.g. "seed", "dreamer", "deap".
    
        feature : str, optional
            Feature name, e.g. "pcc", "plv", "pli", "psd", "de".
    
        Returns
        -------
        str or dict
            The requested path. If dataset or feature is omitted where
            appropriate, the corresponding dictionary is returned.
        """
    
        aliases = {
            # Keep compatibility with the original spellings
            "decompossed": "decomposed",
            "channel_futures": "channel_features",
            "channel_futures_average": "channel_features_average",
        }
    
        argument = aliases.get(argument, argument)
    
        legal_arguments = {
            "dataset",
            "original_eeg",
            "preprocessed_eeg",
            "decomposed",
            "functional_connectivity",
            "channel_features",
            "functional_connectivity_average",
            "channel_features_average",
        }
    
        if argument not in legal_arguments:
            raise ValueError(
                f"Unknown path argument: {argument!r}. "
                f"Expected one of {sorted(legal_arguments)}."
            )
    
        # ------------------------------------------------------------
        # Dataset-level paths
        # ------------------------------------------------------------
        simple_paths = {
            "dataset": cls.DATASET,
            "original_eeg": cls.ORIGINAL,
            "preprocessed_eeg": cls.PREPROCESSED,
            "decomposed": cls.DECOMPOSED,
        }
    
        if argument in simple_paths:
            mapping = simple_paths[argument]
    
            if dataset is None:
                return mapping
    
            dataset = dataset.lower()
    
            if dataset not in mapping:
                raise ValueError(
                    f"Unknown dataset: {dataset!r}. "
                    f"Expected one of {sorted(mapping.keys())}."
                )
            
            return cls._normalize_path(mapping[dataset])
    
        # ------------------------------------------------------------
        # Feature-level paths
        # Structure:
        #     FEATURE[feature][dataset]
        # ------------------------------------------------------------
        feature_paths = {
            "functional_connectivity":
                cls.FUNCTIONAL_CONNECTIVITY,
    
            "channel_features":
                cls.CHANNEL_FEATURES,
    
            "functional_connectivity_average":
                cls.FUNCTIONAL_CONNECTIVITY_DESIGNATE,
    
            "channel_features_average":
                cls.CHANNEL_FEATURES_DESIGNATE,
        }
    
        mapping = feature_paths[argument]
    
        # Neither specified -> return the complete mapping
        if feature is None and dataset is None:
            return mapping
    
        # dataset alone is ambiguous because feature is the first level
        if feature is None:
            raise ValueError(
                f"'feature' is required for {argument!r}."
            )
    
        feature = feature.lower()
    
        if feature not in mapping:
            raise ValueError(
                f"Unknown feature: {feature!r}. "
                f"Expected one of {sorted(mapping.keys())}."
            )
    
        feature_mapping = mapping[feature]
    
        # Feature specified, dataset omitted -> return all datasets
        if dataset is None:
            return feature_mapping
    
        dataset = dataset.lower()
    
        if dataset not in feature_mapping:
            raise ValueError(
                f"Unknown dataset: {dataset!r}. "
                f"Expected one of {sorted(feature_mapping.keys())}."
            )
            
        return cls._normalize_path(feature_mapping[dataset])
        
Validation.report()
PathDefinition.report()
# PathDefinition.report_detailed()