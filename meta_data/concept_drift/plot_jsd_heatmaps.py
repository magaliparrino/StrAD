from __future__ import annotations
import math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, List, Optional

def _tick_positions_labels(n: int, labels: List[str], max_ticks: int = 12):
    """Select a subset of indices for readable axis ticks."""
    if n <= max_ticks:
        idx = np.arange(n)
    else:
        idx = np.linspace(0, n - 1, max_ticks, dtype=int)
        idx = np.unique(idx)
    return idx, [labels[i] for i in idx]

def plot_single_heatmap(
    M: np.ndarray,
    batch_labels: List[str],
    title: str = "",
    ax: Optional[plt.Axes] = None,
    cmap: str = "mako",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cbar: bool = True,
    annotate_diag: bool = False,
    highlight_ref: bool = True,
    ref_index: int = 0
):
    """Plot a heatmap for a single [n, n] matrix."""
    n = M.shape[0]
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    # Reduce ticks for clarity
    tidx, tlab = _tick_positions_labels(n, batch_labels, max_ticks=10)

    sns.heatmap(
        M, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax,
        cbar=cbar, square=True,
        xticklabels=False, yticklabels=False
    )
    
    ax.set_title(title)
    ax.set_xlabel("batch j")
    ax.set_ylabel("batch i")

    # Center ticks on heatmap cells
    ax.set_xticks(tidx + 0.5)
    ax.set_yticks(tidx + 0.5)
    ax.set_xticklabels(tlab, rotation=45, ha="right")
    ax.set_yticklabels(tlab, rotation=0)

    if annotate_diag:
        ax.plot([0, n], [0, n], color="white", lw=1, alpha=0.6)

    if highlight_ref and 0 <= ref_index < n:
        # Highlight the reference batch row and column
        ax.axhline(ref_index, color="yellow", lw=1.2, alpha=0.8)
        ax.axhline(ref_index + 1, color="yellow", lw=1.2, alpha=0.8)
        ax.axvline(ref_index, color="yellow", lw=1.2, alpha=0.8)
        ax.axvline(ref_index + 1, color="yellow", lw=1.2, alpha=0.8)

def plot_feature_heatmaps_grid(
    mats: Dict[str, np.ndarray],
    batch_labels: List[str],
    ncols: int = 3,
    figsize_per_cell: tuple = (4.5, 4.2),
    cmap: str = "mako",
    share_color_scale: bool = True,
    percentile_clip: Optional[float] = 99.0,
    suptitle: Optional[str] = None,
    use_jsd_range: bool = True 
):
    """Display a grid of heatmaps, one per feature."""
    feats = list(mats.keys())
    n = len(feats)
    ncols = max(1, ncols)
    nrows = math.ceil(n / ncols)

    # Use JSD theoretical range [0, ln(2)] if requested
    if use_jsd_range:
        vmin, vmax = 0.0, float(np.log(2.0))
    else:
        if share_color_scale:
            all_vals = np.concatenate([m.flatten() for m in mats.values()])
            vmin, vmax = np.nanmin(all_vals), np.nanmax(all_vals)
            if percentile_clip is not None:
                vmax = np.nanpercentile(all_vals, percentile_clip)
        else:
            vmin, vmax = None, None

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_cell[0]*ncols, figsize_per_cell[1]*nrows)
    )

    # Ensure axes is always a 2D array for consistent indexing
    if nrows == 1 and ncols == 1: axes = np.array([[axes]])
    elif nrows == 1: axes = np.array([axes])
    elif ncols == 1: axes = axes.reshape(-1, 1)

    k = 0
    for r in range(nrows):
        for c in range(ncols):
            ax = axes[r, c]
            if k < n:
                feat = feats[k]
                plot_single_heatmap(
                    mats[feat], batch_labels,
                    title=str(feat), ax=ax, cmap=cmap,
                    vmin=vmin, vmax=vmax, cbar=True,
                    annotate_diag=False, highlight_ref=True, ref_index=0
                )
                k += 1
            else:
                ax.axis("off")

    if suptitle:
        fig.suptitle(suptitle, y=0.995, fontsize=12)

    fig.tight_layout()
    return fig, axes