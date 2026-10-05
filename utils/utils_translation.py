# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 16:53:05 2026

@author: usouu
"""
"""
*** This file needs to be proofreaded.
"""

import numpy as np
import pandas as pd
from scipy.stats import boxcox, yeojohnson

# %% Normalization
def normalize_matrix(matrix, method='minmax', epsilon=1e-8, param=None):
    """
    Normalize or transform a matrix or a batch of matrices.

    Supported methods:
        minmax, max, mean, z-score, boxcox, yeojohnson,
        sqrt, log, and none.

    The input can be either a single matrix with shape (H, W)
    or a batch of matrices with shape (N, H, W).

    Parameters
    ----------
    matrix : np.ndarray
        Input matrix or batch of matrices.

    method : str, optional
        Normalization or transformation method.
        Default is 'minmax'.

    epsilon : float, optional
        Small value used to prevent division by zero
        or numerical instability.
        Default is 1e-8.

    param : dict, optional
        Additional parameters for specific methods.

        Supported parameters include:
        - 'target_range': tuple (a, b)
            Target range for min-max normalization.
            Default is (0, 1).
        - 'lmbda': float or None
            Transformation parameter for Box-Cox or
            Yeo-Johnson transformation.
            If None, SciPy estimates the optimal value.

    Returns
    -------
    np.ndarray
        Normalized or transformed matrix/matrices with
        the same dimensional structure as the input.
    """
    if param is None:
        param = {}

    a, b = param.get('target_range', (0, 1))
    lmbda = param.get('lmbda', None)

    # Determine whether the input contains a batch of matrices
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
                raise ValueError(
                    "Box-Cox transformation requires all values to be > 0."
                )

            mat = boxcox(
                mat.flatten(),
                lmbda=lmbda
            )[0].reshape(mat.shape)

        elif method == 'yeojohnson':
            mat = yeojohnson(
                mat.flatten(),
                lmbda=lmbda
            )[0].reshape(mat.shape)

        elif method == 'sqrt':
            if np.any(mat < 0):
                raise ValueError(
                    "Square-root transformation requires non-negative values."
                )

            mat = np.sqrt(mat + epsilon)

        elif method == 'log':
            if np.any(mat <= 0):
                raise ValueError(
                    "Logarithmic transformation requires all values to be > 0."
                )

            mat = np.log(mat + epsilon)

        elif method == 'none':
            pass

        else:
            raise ValueError(
                f"Unsupported normalization method: {method}"
            )

        normalized.append(mat)

    result = np.stack(normalized) if is_batch else normalized[0]

    return 

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

# %% Label translation
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