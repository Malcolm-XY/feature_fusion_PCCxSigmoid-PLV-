# -*- coding: utf-8 -*-
"""
Created on Sun Feb 15 23:24:35 2026

This code is for parameter optimazation of k and tau for PG-AC model.

@author: 18307
"""
import numpy as np
import feature_fusion

# defination of index
def redundancy_4_matrix(matrix, absolute=True):
    matrix = np.array(matrix)
    width_1, width_2 = matrix.shape
    repeat = width_1

    corr = []
    for i in range(repeat):
        for j in range(repeat):
            if i != j:
                edge_1, edge_2 = matrix[i], matrix[j]
                _corr = np.corrcoef(edge_1, edge_2)[0, 1]
                
                if absolute:
                   _corr = np.abs(_corr)
                    
                corr.append(_corr)
                
    redundancy_ = (1/(width_1*(width_1 - 1))) * np.sum(corr)
    return redundancy_

def spectral_entropy_4_matrix(matrix, eps=1e-12):
    eigvals = np.linalg.eigvalsh(matrix)
    eigvals = np.maximum(eigvals, 0)
    p = eigvals / (np.sum(eigvals) + eps)
    H = -np.sum(p * np.log(p + eps))
    Hmax = np.log(len(p))
    redundancy = 1 - H / Hmax
    return H, redundancy

def spectral_energy_compaction_4_matrix(matrix, k_ratio=0.1, eps=1e-12):
    """
    Compute spectral energy compaction of a matrix.

    Parameters
    ----------
    A : ndarray (N, N)
        Input matrix (symmetric preferred).
    k_ratio : float
        Ratio of top eigenvalues to accumulate (0 < k_ratio <= 1).
    eps : float
        Numerical stability constant.

    Returns
    -------
    EC : float
        Energy compaction ratio in [0, 1].
    """
    eigvals = np.linalg.eigvalsh(matrix)
    eigvals = np.abs(eigvals)
    eigvals = np.sort(eigvals)[::-1]

    k = max(1, int(len(eigvals) * k_ratio))
    return np.sum(eigvals[:k]) / (np.sum(eigvals) + eps)

def mutual_information_4_matrices_labels(matrices, labels):
    X = np.asarray(matrices)
    y = np.asarray(labels).ravel()

    if X.ndim < 2:
        raise ValueError(f"`matrices` must have shape (n_samples, ...). Got {X.shape}")
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"Sample mismatch: matrices has {X.shape[0]} samples, labels has {y.shape[0]}.")

    n_samples = X.shape[0]
    feat_shape = X.shape[1:]
    n_feats = int(np.prod(feat_shape))

    # Flatten each sample matrix -> (n_samples, n_feats)
    Xf = X.reshape(n_samples, n_feats)

    # Clean invalid values
    Xf = np.nan_to_num(Xf, nan=0.0, posinf=0.0, neginf=0.0)

    # Preferred: sklearn's kNN MI estimator for continuous features + discrete labels
    try:
        from sklearn.feature_selection import mutual_info_classif

        mi = mutual_info_classif(
            Xf, y,
            discrete_features=False,
            random_state=0,
            n_neighbors=3
        )
        mi = np.asarray(mi, dtype=float)

    except Exception:
        # Fallback (no sklearn or error): discretize each feature and use mutual_info_score
        from sklearn.metrics import mutual_info_score

        def _discretize_1d(v, bins=16):
            v = np.asarray(v, dtype=float)
            v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
            if np.all(v == v[0]):
                return np.zeros_like(v, dtype=int)
            edges = np.histogram_bin_edges(v, bins=bins)
            # digitize -> [1..bins], convert to [0..bins-1]
            return np.clip(np.digitize(v, edges[1:-1], right=False), 0, bins - 1)

        mi = np.empty(n_feats, dtype=float)
        for j in range(n_feats):
            x_disc = _discretize_1d(Xf[:, j], bins=16)
            mi[j] = mutual_info_score(x_disc, y)

    return mi.reshape(feat_shape)

def cohens_d_4_matrices_labels(matrices, labels):
    X = np.asarray(matrices)
    y = np.asarray(labels).ravel()

    if X.ndim < 2:
        raise ValueError("`matrices` must have shape (n_samples, ...).")
    if X.shape[0] != y.shape[0]:
        raise ValueError("Sample mismatch.")

    # Flatten (n_samples, features)
    n_samples = X.shape[0]
    feat_shape = X.shape[1:]
    Xf = X.reshape(n_samples, -1)

    # Binary labels: group A / B
    g1 = Xf[y == y.min()]  # class 0
    g2 = Xf[y == y.max()]  # class 1

    mean1, mean2 = g1.mean(0), g2.mean(0)
    var1, var2 = g1.var(0, ddof=1), g2.var(0, ddof=1)

    # pooled SD
    n1, n2 = len(g1), len(g2)
    pooled_std = np.sqrt(( (n1-1)*var1 + (n2-1)*var2 ) / (n1+n2-2) + 1e-12)

    d = (mean2 - mean1) / pooled_std

    return d.reshape(feat_shape)

# index for implementation
def redundancy_4_fns(k, tau,
                    avg_pcc_alpha, avg_plv_alpha,
                    avg_pcc_beta,  avg_plv_beta,
                    avg_pcc_gamma, avg_plv_gamma):
    params = {'k': float(k), 'tau': float(tau)}

    fused_alpha = feature_fusion.feature_fusion_sigmoid_gating(avg_pcc_alpha, avg_plv_alpha, params=params)
    fused_beta  = feature_fusion.feature_fusion_sigmoid_gating(avg_pcc_beta,  avg_plv_beta,  params=params)
    fused_gamma = feature_fusion.feature_fusion_sigmoid_gating(avg_pcc_gamma, avg_plv_gamma, params=params)
    
    fused_alpha = np.mean(fused_alpha, axis=0)
    fused_beta = np.mean(fused_beta, axis=0)
    fused_gamma = np.mean(fused_gamma, axis=0)
    
    r_a = redundancy_4_matrix(fused_alpha)
    r_b = redundancy_4_matrix(fused_beta)
    r_g = redundancy_4_matrix(fused_gamma)

    r_mean = np.mean([r_a, r_b, r_g])

    return r_mean

def spectral_entropy_4_fns(k, tau,
                           avg_pcc_alpha, avg_plv_alpha,
                           avg_pcc_beta,  avg_plv_beta,
                           avg_pcc_gamma, avg_plv_gamma):
    params = {'k': float(k), 'tau': float(tau)}

    fused_alpha = feature_fusion.feature_fusion_sigmoid_gating(avg_pcc_alpha, avg_plv_alpha, params=params)
    fused_beta  = feature_fusion.feature_fusion_sigmoid_gating(avg_pcc_beta,  avg_plv_beta,  params=params)
    fused_gamma = feature_fusion.feature_fusion_sigmoid_gating(avg_pcc_gamma, avg_plv_gamma, params=params)
    
    h_a = spectral_entropy_4_matrix(fused_alpha)
    h_b = spectral_entropy_4_matrix(fused_beta)
    h_g = spectral_entropy_4_matrix(fused_gamma)

    h_mean = np.mean([h_a, h_b, h_g])

    return h_mean

def spectral_energy_compaction_4_fns(k, tau,
                           avg_pcc_alpha, avg_plv_alpha,
                           avg_pcc_beta,  avg_plv_beta,
                           avg_pcc_gamma, avg_plv_gamma):
    params = {'k': float(k), 'tau': float(tau)}

    fused_alpha = feature_fusion.feature_fusion_sigmoid_gating(avg_pcc_alpha, avg_plv_alpha, params=params)
    fused_beta  = feature_fusion.feature_fusion_sigmoid_gating(avg_pcc_beta,  avg_plv_beta,  params=params)
    fused_gamma = feature_fusion.feature_fusion_sigmoid_gating(avg_pcc_gamma, avg_plv_gamma, params=params)
    
    h_a = spectral_energy_compaction_4_matrix(fused_alpha)
    h_b = spectral_energy_compaction_4_matrix(fused_beta)
    h_g = spectral_energy_compaction_4_matrix(fused_gamma)

    h_mean = np.mean([h_a, h_b, h_g])

    return h_mean

def mutual_information_4_fns_labels(k, tau,
                                    avg_pcc_alpha, avg_plv_alpha,
                                    avg_pcc_beta,  avg_plv_beta,
                                    avg_pcc_gamma, avg_plv_gamma, labels):
    params = {'k': float(k), 'tau': float(tau)}

    # fused_alpha = feature_fusion.feature_fusion_sigmoid_gating(avg_pcc_alpha, avg_plv_alpha, params=params)
    # fused_beta  = feature_fusion.feature_fusion_sigmoid_gating(avg_pcc_beta,  avg_plv_beta,  params=params)
    fused_gamma = feature_fusion.feature_fusion_sigmoid_gating(avg_pcc_gamma, avg_plv_gamma, params=params)
    
    # h_a = mutual_information_4_matrices_labels(fused_alpha, labels)
    # h_b = mutual_information_4_matrices_labels(fused_beta, labels)
    h_g = mutual_information_4_matrices_labels(fused_gamma, labels)

    # h_mean = np.mean([h_a, h_b, h_g])
    h_mean = np.mean(h_g)

    return h_mean

def cohens_d_4_fns_labels(k, tau,
                          avg_pcc_alpha, avg_plv_alpha,
                          avg_pcc_beta,  avg_plv_beta,
                          avg_pcc_gamma, avg_plv_gamma, labels):
    params = {'k': float(k), 'tau': float(tau)}

    fused_alpha = feature_fusion.feature_fusion_sigmoid_gating(avg_pcc_alpha, avg_plv_alpha, params=params)
    fused_beta  = feature_fusion.feature_fusion_sigmoid_gating(avg_pcc_beta,  avg_plv_beta,  params=params)
    fused_gamma = feature_fusion.feature_fusion_sigmoid_gating(avg_pcc_gamma, avg_plv_gamma, params=params)
    
    h_a = cohens_d_4_matrices_labels(fused_alpha, labels)
    h_b = cohens_d_4_matrices_labels(fused_beta, labels)
    h_g = cohens_d_4_matrices_labels(fused_gamma, labels)

    h_mean = np.mean([h_a, h_b, h_g])

    return h_mean

# grid search (loss function embedded)
def grid_search_p1_p2(p1_list, p2_list, boundary, loss_func, *args):
    loss_function = loss_func
    best = {"p1": None, "p2": None, "loss": np.inf if boundary == "lower" else -np.inf}
    for p1 in p1_list:
        for p2 in p2_list:
            loss = loss_function(p1, p2, *args)
            if boundary == "lower":
                if loss < best["loss"]:
                    best.update(p1=float(p1), p2=float(p2), loss=float(loss))
            elif boundary == "upper":
                if loss > best["loss"]:
                    best.update(p1=float(p1), p2=float(p2), loss=float(loss))
            
    return best

if __name__ == '__main__':
    #%% Data Preparation
    from utils import utils_feature_loading
    pcc = utils_feature_loading.read_fcs('seed', 'sub1ex1', 'pcc')
    plv = utils_feature_loading.read_fcs('seed', 'sub1ex1', 'plv')
    
    pcc_alpha, pcc_beta, pcc_gamma = pcc['alpha'], pcc['beta'], pcc['gamma']
    plv_alpha, plv_beta, plv_gamma = plv['alpha'], plv['beta'], plv['gamma']
    
    labels = utils_feature_loading.read_labels('seed', header=True)
    
    mi_alpha = mutual_information_4_matrices_labels(pcc_alpha, labels)
    mi_beta = mutual_information_4_matrices_labels(pcc_beta, labels)
    mi_gamma = mutual_information_4_matrices_labels(pcc_gamma, labels)
    
    mi_avg_alpha, mi_avg_beta, mi_avg_gamma = np.mean(mi_alpha), np.mean(mi_beta), np.mean(mi_gamma)
    
    #%% Optimization
    # optimization parameters
    k_list   = np.linspace(1, 100, 10)     # 你按实际调整范围/步长
    tau_list = np.linspace(0.01, 1, 10)  # tau 通常希望 >0
    
    """
    Optimal Succussed
    Optimal Parameters: k, tau = [100, 0.12]
    """
    # optimization; redundancy
    best = grid_search_p1_p2(k_list, tau_list, "upper", redundancy_4_fns,
                             pcc_alpha, plv_alpha,
                             pcc_beta,  plv_beta,
                             pcc_gamma, plv_gamma)
    
    print(best["p1"], best["p2"], best["loss"]) # "upper": 100, 0.12; "lower": 100, 1.0, fail
    
    # optimization； mutual information
    best = grid_search_p1_p2(k_list, tau_list, "upper", mutual_information_4_fns_labels,
                             pcc_alpha, plv_alpha,
                             pcc_beta,  plv_beta,
                             pcc_gamma, plv_gamma, labels)
    
    print(best["p1"], best["p2"], best["loss"]) # "upper": 12.0, 0.12; "lower": fail
    
    # optimization; cohens d
    best = grid_search_p1_p2(k_list, tau_list, "lower", cohens_d_4_fns_labels,
                             pcc_alpha, plv_alpha,
                             pcc_beta,  plv_beta,
                             pcc_gamma, plv_gamma, labels)
    
    print(best["p1"], best["p2"], best["loss"]) # "upper": 100, 1.0, fail; "lower": 12.0, 0.34
    
    # optimization; spectral energy
    best = grid_search_p1_p2(k_list, tau_list, "upper", spectral_energy_compaction_4_fns,
                             pcc_alpha, plv_alpha,
                             pcc_beta,  plv_beta,
                             pcc_gamma, plv_gamma)
    
    print(best["p1"], best["p2"], best["loss"]) # "upper": 100, 0.12; "lower": 100, 0.56
    
    """
    Optimal Failed
    """
    # optimization; spectral entropy
    # best = grid_search_p1_p2(k_list, tau_list, "lower", spectral_entropy_4_fns,
    #                          pcc_alpha, plv_alpha,
    #                          pcc_beta,  plv_beta,
    #                          pcc_gamma, plv_gamma)
    
    # print(best["p1"], best["p2"], best["loss"]) # "upper": 100, 1.0, fail; "lower": 100, 0.01
