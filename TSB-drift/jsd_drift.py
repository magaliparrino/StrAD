from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from sklearn.preprocessing import StandardScaler

ArrayLike = Union[np.ndarray, pd.Series]

def normalize_df_zscore(df: pd.DataFrame, label_col: int | str = -1) -> tuple[pd.DataFrame, StandardScaler]:
    """Z-score normalization on features only."""
    if isinstance(label_col, int):
        label_name = df.columns[label_col]
    else:
        label_name = label_col

    feature_cols = [c for c in df.columns if c != label_name]
    X = df[feature_cols].astype(float).values
    y = df[label_name].values

    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X)

    # Handle constant features (std=0)
    if hasattr(scaler, "scale_"):
        const_idx = np.where(scaler.scale_ == 0)[0]
        if const_idx.size:
            X_norm[:, const_idx] = 0.0

    df_norm = pd.DataFrame(X_norm, index=df.index, columns=feature_cols)
    df_norm[label_name] = y

    return df_norm, scaler

def optimal_bins_fd(x, min_bins=10, max_bins=30):
    """Freedman-Diaconis rule for bin calculation."""
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return min_bins

    iqr = np.percentile(x, 75) - np.percentile(x, 25)
    if iqr == 0:
        return min_bins

    h = 2 * iqr * (x.size ** (-1/3))
    if h <= 0:
        return min_bins

    nbins = int(np.ceil((x.max() - x.min()) / h))
    return max(min(nbins, max_bins), min_bins)

def make_batches_index(n_samples: int, batch_len: int, drop_incomplete: bool = True) -> List[slice]:
    """Divide indices into contiguous batches."""
    if batch_len <= 0:
        raise ValueError("batch_len must be > 0")
    slices = []
    for start in range(0, n_samples, batch_len):
        end = start + batch_len
        if end > n_samples and drop_incomplete:
            break
        slices.append(slice(start, min(end, n_samples)))
    return slices

def batch_labels_from_index(index: pd.Index, batches: List[slice]) -> List[str]:
    """Generate string labels for time or range intervals."""
    labels = []
    for s in batches:
        i0, i1 = s.start, s.stop - 1
        if isinstance(index, pd.DatetimeIndex):
            labels.append(f"{index[i0].strftime('%Y-%m-%d %H:%M:%S')} -> {index[i1].strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            labels.append(f"{i0}->{i1}")
    return labels

def _bin_edges_for_feature(values: ArrayLike, bins: Union[int, str] = "fd",
                           clip_quantiles: Optional[Tuple[float, float]] = (0.001, 0.999)) -> np.ndarray:
    """Calculate stable bin edges for a feature."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.array([-1.0, 1.0], dtype=float)
    if clip_quantiles is not None:
        lo, hi = np.quantile(v, [clip_quantiles[0], clip_quantiles[1]])
        if not np.isclose(lo, hi):
            v = v[(v >= lo) & (v <= hi)]
    
    edges = np.histogram_bin_edges(v, bins=bins)
    edges = np.unique(edges)
    if edges.size < 2:
        m = np.nanmean(v) if v.size else 0.0
        edges = np.array([m - 1e-6, m + 1e-6], dtype=float)
    return edges

def _hist_probs(x: np.ndarray, edges: np.ndarray, alpha: float = 1e-3) -> np.ndarray:
    """Discrete probability distribution with Laplace smoothing."""
    x = np.clip(x, edges[0], edges[-1])
    counts, _ = np.histogram(x, bins=edges)
    counts = counts.astype(float)
    counts += alpha
    return counts / counts.sum()

def _stack_batch_probs(values: ArrayLike, batches: List[slice], edges: np.ndarray,
                       alpha: float = 1e-3) -> np.ndarray:
    """Stack distributions per batch into [n_batches, n_bins] matrix."""
    v = np.asarray(values, dtype=float)
    P = np.zeros((len(batches), len(edges) - 1), dtype=float)
    for i, s in enumerate(batches):
        x = v[s]
        x = x[np.isfinite(x)]
        if x.size == 0:
            P[i] = np.ones(len(edges) - 1, dtype=float)
        else:
            P[i] = _hist_probs(x, edges=edges, alpha=alpha)
    
    row_sums = P.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    P /= row_sums
    return P

def kl_matrix_from_probs(P: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Compute pairwise KL divergence matrix KL(i||j)."""
    P_ = np.clip(P, eps, 1.0)
    P_ = P_ / P_.sum(axis=1, keepdims=True)
    logP = np.log(P_)
    Pi = P_[:, None, :]
    logPi = logP[:, None, :]
    logPj = logP[None, :, :]
    return np.sum(Pi * (logPi - logPj), axis=2)

def jsd_matrix_from_probs(P: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Compute symmetric Jensen-Shannon Distance matrix."""
    P_ = np.clip(P, eps, 1.0)
    P_ = P_ / P_.sum(axis=1, keepdims=True)
    Pi = P_[:, None, :]
    Pj = P_[None, :, :]
    M = 0.5 * (Pi + Pj)
    logPi, logPj, logM = np.log(Pi), np.log(Pj), np.log(M)
    KL_iM = np.sum(Pi * (logPi - logM), axis=2)
    KL_jM = np.sum(Pj * (logPj - logM), axis=2)
    return 0.5 * (KL_iM + KL_jM)

def smooth_gaussian(p: np.ndarray, sigma_bins: float = 1.0) -> np.ndarray:
    """Apply Gaussian smoothing to a distribution."""
    radius = int(3 * sigma_bins)
    xs = np.arange(-radius, radius + 1)
    k = np.exp(-0.5 * (xs / sigma_bins) ** 2)
    k /= k.sum()
    return np.convolve(p, k, mode="same")

def wasserstein_matrix_from_probs(P: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Compute pairwise 1-Wasserstein distance matrix."""
    n, b = P.shape
    widths = np.diff(edges)
    cdfs = np.cumsum(P, axis=1)
    W = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            W[i, j] = np.sum(np.abs(cdfs[i] - cdfs[j]) * widths)
    return W

def compute_kl_matrices(
    df: pd.DataFrame,
    batch_len: int,
    features: Optional[List[str]] = None,
    bins: Union[int, str] = "fd",
    alpha: float = 1e-3,
    drop_incomplete: bool = True,
    metric: str = "kl",
    clip_quantiles: Optional[Tuple[float, float]] = (0.001, 0.999),
    smooth_sigma: Optional[float] = None,
    auto_min_bins: int = 10,
    auto_max_bins: int = 30
) -> Tuple[Dict[str, np.ndarray], List[str], Dict[str, np.ndarray]]:
    """Calculates divergence matrices per feature between batches."""
    if features is None:
        features = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]

    n_samples = len(df)
    batches = make_batches_index(n_samples, batch_len=batch_len, drop_incomplete=drop_incomplete)
    if len(batches) < 2:
        raise ValueError("Insufficient batches for comparison.")

    batch_labels = batch_labels_from_index(df.index, batches)
    mats: Dict[str, np.ndarray] = {}
    edges_dict: Dict[str, np.ndarray] = {}

    for feat in features:
        v = df[feat].values.astype(float)
        v = v[np.isfinite(v)]
        
        if v.size == 0 or np.ptp(v) <= 1e-12:
            n_b = len(batches)
            mats[feat] = np.zeros((n_b, n_b), dtype=float)
            edges_dict[feat] = np.array([0.0, 1.0])
            continue

        v_used = v
        if clip_quantiles is not None:
            lo_q, hi_q = np.quantile(v, [clip_quantiles[0], clip_quantiles[1]])
            if lo_q < hi_q:
                v_clipped = v[(v >= lo_q) & (v <= hi_q)]
                if v_clipped.size >= max(10, auto_min_bins):
                    v_used = v_clipped

        if isinstance(bins, int):
            edges = np.histogram_bin_edges(v_used, bins=max(bins, 2))
        elif isinstance(bins, str) and bins.lower() == "auto":
            nb = optimal_bins_fd(v_used, min_bins=auto_min_bins, max_bins=auto_max_bins)
            edges = np.histogram_bin_edges(v_used, bins=nb)
        else:
            edges = np.histogram_bin_edges(v_used, bins=bins)

        edges = np.unique(edges)
        P = _stack_batch_probs(df[feat].values.astype(float), batches, edges, alpha=alpha)

        if smooth_sigma:
            P = np.apply_along_axis(lambda p: smooth_gaussian(p, sigma_bins=smooth_sigma), axis=1, arr=P)
            P /= P.sum(axis=1, keepdims=True)

        if metric == "kl":
            M = kl_matrix_from_probs(P)
        elif metric == "jsd":
            M = jsd_matrix_from_probs(P)
        elif metric == "wasserstein":
            M = wasserstein_matrix_from_probs(P, edges)
        else:
            raise ValueError("Unsupported metric.")

        mats[feat], edges_dict[feat] = M, edges

    return mats, batch_labels, edges_dict

def aggregate_feature_matrices(mats: Dict[str, np.ndarray], how: str = "mean") -> np.ndarray:
    """Aggregates multiple feature matrices into one."""
    if not mats:
        raise ValueError("No matrices to aggregate.")
    stack = np.stack(list(mats.values()), axis=0)
    if how == "mean": return np.nanmean(stack, axis=0)
    if how == "median": return np.nanmedian(stack, axis=0)
    if how == "max": return np.nanmax(stack, axis=0)
    raise ValueError("Invalid aggregation method.")