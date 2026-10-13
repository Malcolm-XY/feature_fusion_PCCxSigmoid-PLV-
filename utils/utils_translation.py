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

    return result

def rectify_diagonal(matrix, value=0):
    """
    Replace diagonal elements of a matrix or a batch of matrices.

    Parameters
    ----------
    matrix : np.ndarray
        Input array with shape (..., N, N).
        For 1D arrays, all elements are replaced.
    value : scalar
        Value assigned to diagonal elements.

    Returns
    -------
    np.ndarray
        Modified input array (in-place).
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("Input must be a NumPy array.")

    if matrix.ndim == 1:
        matrix[:] = value

    elif matrix.ndim >= 2:
        if matrix.shape[-2] != matrix.shape[-1]:
            raise ValueError("The last two dimensions must be equal.")

        idx = np.arange(matrix.shape[-1])
        matrix[..., idx, idx] = value

    else:
        raise ValueError("Input must have at least one dimension.")

    return matrix

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

