import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
import pandas as pd
from matplotlib.gridspec import GridSpec
from os.path import expanduser
from typing import Dict, Optional, Tuple, List

try:
    # Used for cat-vs-mean correlations; script should still run without SciPy.
    from scipy import stats
except Exception:  # pragma: no cover
    stats = None

def load_matrix_from_file(file_path):
    """Load matrix data from file (supports .npy, .csv, .txt)"""
    file_path = Path(file_path)
    
    if not file_path.exists():
        return None
    
    if file_path.suffix == '.npy':
        return np.load(file_path)
    elif file_path.suffix == '.csv':
        return pd.read_csv(file_path, header=None).values
    else:
        return np.loadtxt(file_path)

def extract_frequency_from_filename(filename):
    """Extract frequency from filename (e.g., 'msc_cat_10Hz' -> '10Hz')"""
    parts = filename.split('_')
    for part in parts:
        if 'Hz' in part:
            return part
    return 'unknown'

def find_wilcoxon_folder(base_path):
    """Find the Wilcoxon folder (e.g., 'Wilcoxon_11022025-1703')"""
    dataset_path = base_path.parent
    wilcoxon_folders = list(dataset_path.glob('Wilcoxon_*'))
    if wilcoxon_folders:
        return sorted(wilcoxon_folders)[-1]  # Return the most recent one
    return None

def get_lower_triangle_vector(matrix: np.ndarray, include_diagonal: bool = False) -> np.ndarray:
    """
    Return the vectorized lower-triangular part of a square matrix.

    Parameters
    ----------
    matrix : np.ndarray
        Square matrix.
    include_diagonal : bool, default False
        If True, include the diagonal; otherwise use strictly lower triangle.

    Returns
    -------
    np.ndarray
        1D array of lower-triangular entries.
    """
    if matrix is None:
        return None
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("get_lower_triangle_vector expects a square 2D matrix.")
    n = matrix.shape[0]
    # k = -1 → strictly below diagonal; use k=0 if you want to include diagonal
    k = 0 if include_diagonal else -1
    idx = np.tril_indices(n, k=k)
    return matrix[idx]


def run_pca_and_plot(vectors, labels, title, save_path):
    """
    Run PCA on a list of 1D vectors and save 3D scatter plots (PC1, PC2, PC3).

    Parameters
    ----------
    vectors : list of np.ndarray
        Each entry is a 1D feature vector (same length).
    labels : list of str
        Text labels for each sample.
    title : str
        Figure title.
    save_path : Path or str
        Where to save the PCA figure.
    """
    if len(vectors) < 2:
        print("  [PCA] Not enough samples to run PCA (need >= 2). Skipping.")
        return

    # Stack into matrix (n_samples, n_features)
    X = np.vstack(vectors)

    # Center features
    X_mean = X.mean(axis=0, keepdims=True)
    Xc = X - X_mean

    # SVD-based PCA (no sklearn dependency)
    try:
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    except np.linalg.LinAlgError as e:
        print(f"  [PCA] SVD failed: {e}. Skipping PCA plot.")
        return

    # Scores: projection onto PCs
    scores = U * S  # (n_samples, n_components)

    # Calculate explained variance
    total_variance = np.sum(S**2)
    explained_variance_ratio = (S**2) / total_variance
    
    # Print PCA results
    print(f"  [PCA] Singular values (S): {S[:3]}")
    print(f"  [PCA] Explained variance:")
    print(f"    PC1: {explained_variance_ratio[0]:.4f} ({explained_variance_ratio[0]*100:.2f}%)")
    print(f"    PC2: {explained_variance_ratio[1]:.4f} ({explained_variance_ratio[1]*100:.2f}%)")
    print(f"    PC3: {explained_variance_ratio[2]:.4f} ({explained_variance_ratio[2]*100:.2f}%)")
    print(f"  [PCA] Total variance explained (PC1-3): {explained_variance_ratio[:3].sum():.4f} ({explained_variance_ratio[:3].sum()*100:.2f}%)")

    # Use first three PCs
    pc1 = scores[:, 0]
    pc2 = scores[:, 1] if scores.shape[1] > 1 else np.zeros_like(pc1)
    pc3 = scores[:, 2] if scores.shape[1] > 2 else np.zeros_like(pc1)

    # Create figure with multiple subplots
    fig = plt.figure(figsize=(18, 5))

    # Plot 1: Singular values (S) - descending order
    ax1 = fig.add_subplot(141)
    n_components_to_show = min(10, len(S))
    ax1.bar(range(1, n_components_to_show + 1), S[:n_components_to_show])
    ax1.set_xlabel('Component')
    ax1.set_ylabel('Singular Value (S)')
    ax1.set_title('Singular Values (Descending Order)')
    ax1.set_xticks(range(1, n_components_to_show + 1))

    # Plot 2: Variance explained
    ax2 = fig.add_subplot(142)
    ax2.bar(range(1, 4), explained_variance_ratio[:3])
    ax2.set_xlabel('Principal Component')
    ax2.set_ylabel('Variance Explained')
    ax2.set_title('Variance Explained by Each PC')
    ax2.set_xticks(range(1, 4))

    # Plot 3: 3D scatter plot (PC1, PC2, PC3)
    ax3 = fig.add_subplot(143, projection='3d')
    ax3.scatter(pc1, pc2, pc3, alpha=0.8, s=30)

    # Label each point with participant + condition (optional - can be cluttered)
    # Uncomment the next two lines if you want labels on the 3D plot
    # for x, y, z, lab in zip(pc1, pc2, pc3, labels):
    #     ax3.text(x, y, z, lab, fontsize=5, ha='center', va='center')

    ax3.set_xlabel(f'PC1 ({explained_variance_ratio[0]*100:.1f}%)')
    ax3.set_ylabel(f'PC2 ({explained_variance_ratio[1]*100:.1f}%)')
    ax3.set_zlabel(f'PC3 ({explained_variance_ratio[2]*100:.1f}%)')
    ax3.set_title('3D PCA Space')

    # Plot 4: 2D scatter plot (PC1 vs PC2)
    ax4 = fig.add_subplot(144)
    ax4.scatter(pc1, pc2, alpha=0.8)

    # Label each point with participant + condition
    for x, y, lab in zip(pc1, pc2, labels):
        ax4.text(x, y, lab, fontsize=7, ha='center', va='center')

    ax4.set_xlabel(f'PC1 ({explained_variance_ratio[0]*100:.1f}%)')
    ax4.set_ylabel(f'PC2 ({explained_variance_ratio[1]*100:.1f}%)')
    ax4.set_title('2D View (PC1 vs PC2)')
    ax4.axhline(0, color='grey', linewidth=0.5, linestyle='--', alpha=0.5)
    ax4.axvline(0, color='grey', linewidth=0.5, linestyle='--', alpha=0.5)
    
    fig.tight_layout()

    save_path = Path(save_path)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  [PCA] Saved PCA plot to: {save_path}")


def _spearmanr_safe(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Compute Spearman correlation with graceful fallback if SciPy is unavailable."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = min(x.size, y.size)
    x = x[:m]
    y = y[:m]
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.size < 3:
        return float('nan'), float('nan')
    if stats is None:
        # Fallback: Pearson correlation of rank-transformed data.
        rx = pd.Series(x).rank(method='average').to_numpy()
        ry = pd.Series(y).rank(method='average').to_numpy()
        r = np.corrcoef(rx, ry)[0, 1]
        return float(r), float('nan')
    r, p = stats.spearmanr(x, y, nan_policy='omit')
    return float(r), float(p)


def create_cat_mean_scatterplot(mean_vec: np.ndarray,
                                cat_vec: np.ndarray,
                                rho: float,
                                p_value: float,
                                out_path: Path,
                                title: str) -> None:
    """Scatterplot of mean vs cat lower-triangle edge values with correlation stats."""
    mean_vec = np.asarray(mean_vec, dtype=float)
    cat_vec = np.asarray(cat_vec, dtype=float)
    m = min(mean_vec.size, cat_vec.size)
    mean_vec = mean_vec[:m]
    cat_vec = cat_vec[:m]
    valid = np.isfinite(mean_vec) & np.isfinite(cat_vec)
    mean_vec = mean_vec[valid]
    cat_vec = cat_vec[valid]
    if mean_vec.size < 3:
        return

    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.scatter(mean_vec, cat_vec, alpha=0.6, s=25, edgecolors='black', linewidth=0.4)
    ax.set_xlabel('MSC mean (lower triangle edges)')
    ax.set_ylabel('MSC cat (lower triangle edges)')
    ax.set_title(title, fontsize=12, fontweight='bold')

    if np.isfinite(p_value):
        stats_text = f"Spearman ρ = {rho:.4f}\np = {p_value:.3e}\nn = {mean_vec.size}"
    else:
        stats_text = f"Spearman ρ = {rho:.4f}\nn = {mean_vec.size}"
    ax.text(
        0.05, 0.95, stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85)
    )
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def correlate_cat_vs_mean_for_frequency(
    *,
    dataset: str,
    movement: str,
    freq_name_cat: str,
    left_path: Path,
    right_path: Path,
    output_dir: Path,
) -> Optional[pd.DataFrame]:
    """Correlate cat vs mean matrices (lower triangles) per subject and condition.

    This uses the *exact* corresponding edges by vectorizing the strictly lower
    triangle of each matrix (same region ordering), then correlates cat vs mean.

    Expected directory naming:
      - Left_path/<freq_name_cat>/<subj>/... (e.g., msc_cat_10Hz)
      - Left_path/<freq_name_mean>/<subj>/... (e.g., msc_mean_10Hz)
    and similarly for right_path.

    Returns
    -------
    pd.DataFrame or None
        Long-form summary with one row per (subject, condition).
    """
    if 'msc_cat' not in freq_name_cat:
        return None

    freq_name_mean = freq_name_cat.replace('msc_cat', 'msc_mean')
    left_cat_dir = left_path / freq_name_cat
    right_cat_dir = right_path / freq_name_cat
    left_mean_dir = left_path / freq_name_mean
    right_mean_dir = right_path / freq_name_mean

    # Only proceed if mean exists; otherwise skip quietly.
    if not left_mean_dir.exists() and not right_mean_dir.exists():
        print(f"  [cat-mean] No matching mean directory for {freq_name_cat}. Skipping correlations.")
        return None

    # Determine participant set across cat/mean + left/right
    participants: List[str] = []
    for d in [left_cat_dir, right_cat_dir, left_mean_dir, right_mean_dir]:
        if d.exists():
            participants.extend([p.name for p in d.iterdir() if p.is_dir()])
    participants = sorted(set(participants))
    if not participants:
        print(f"  [cat-mean] No participant folders found for {freq_name_cat} / {freq_name_mean}.")
        return None

    hz_label = extract_frequency_from_filename(freq_name_cat)
    corr_out_dir = output_dir / 'cat_vs_mean_correlations' / freq_name_cat
    rows: List[Dict[str, object]] = []

    # File patterns per movement type
    # For mov: {movement}_{freq_name}_{participant}_{Pro|Anti}_average*.csv
    # For cue: {movement}_{L|R}_*_{participant}_{Pro|Anti}_average*.csv
    conditions = [
        ('ProL', 'left', 'Pro'),
        ('ProR', 'right', 'Pro'),
        ('AntiL', 'left', 'Anti'),
        ('AntiR', 'right', 'Anti'),
    ]

    for participant in participants:
        for cond_label, side, proanti in conditions:
            # Select base dirs
            if side == 'left':
                cat_dir = left_cat_dir / participant
                mean_dir = left_mean_dir / participant
            else:
                cat_dir = right_cat_dir / participant
                mean_dir = right_mean_dir / participant

            cat_mat = None
            mean_mat = None

            if cat_dir.exists():
                if movement == 'cue':
                    side_token = 'L' if side == 'left' else 'R'
                    cat_files = list(cat_dir.glob(f"{movement}_{side_token}_*_{participant}_{proanti}_average*.csv"))
                else:
                    cat_files = list(cat_dir.glob(f"{movement}_{freq_name_cat}_{participant}_{proanti}_average*.csv"))
                if cat_files:
                    cat_mat = load_matrix_from_file(cat_files[0])

            if mean_dir.exists():
                if movement == 'cue':
                    side_token = 'L' if side == 'left' else 'R'
                    mean_files = list(mean_dir.glob(f"{movement}_{side_token}_*_{participant}_{proanti}_average*.csv"))
                else:
                    mean_files = list(mean_dir.glob(f"{movement}_{freq_name_mean}_{participant}_{proanti}_average*.csv"))
                if mean_files:
                    mean_mat = load_matrix_from_file(mean_files[0])

            if cat_mat is None or mean_mat is None:
                continue

            try:
                cat_vec = get_lower_triangle_vector(cat_mat, include_diagonal=False)
                mean_vec = get_lower_triangle_vector(mean_mat, include_diagonal=False)
            except Exception as e:
                print(f"  [cat-mean] Skipping {participant} {cond_label}: {e}")
                continue

            rho, p = _spearmanr_safe(mean_vec, cat_vec)
            # effective n after finite filtering
            m = min(mean_vec.size, cat_vec.size)
            valid = np.isfinite(mean_vec[:m]) & np.isfinite(cat_vec[:m])
            n_edges = int(valid.sum())

            title = f"{dataset} | {hz_label} | {participant} | {cond_label}: cat vs mean"
            out_path = corr_out_dir / f"{participant}_{cond_label}_cat_vs_mean.png"
            create_cat_mean_scatterplot(mean_vec, cat_vec, rho, p, out_path, title)

            rows.append({
                'dataset': dataset,
                'movement': movement,
                'frequency_folder_cat': freq_name_cat,
                'frequency_folder_mean': freq_name_mean,
                'hz': hz_label,
                'participant': participant,
                'condition': cond_label,
                'n_edges': n_edges,
                'spearman_rho': rho,
                'spearman_p': p,
                'cat_file_dir': str(cat_dir),
                'mean_file_dir': str(mean_dir),
            })

    if not rows:
        print(f"  [cat-mean] No cat/mean pairs found for {freq_name_cat}.")
        return None

    df = pd.DataFrame(rows)
    summary_csv = corr_out_dir / f"cat_vs_mean_correlations_{freq_name_cat}.csv"
    df.to_csv(summary_csv, index=False)
    print(f"  [cat-mean] Saved correlation summary to: {summary_csv}")
    return df


def create_heatmap_subplot(matrix: np.ndarray, ax, vmin: float, vmax: float, 
                           show_labels: bool = True):
    """
    Create a heatmap subplot with symmetric colour scale around 0.
    """
    region_labels = [
        'V1-L', 'V3-L', 'SPOC-L', 'AG-L', 'POJ-L', 'SPL-L', 'mIPS-L', 'VIP-L', 'IPL-L', 'STS-L',
        'S1-L', 'M1-L', 'SMA-L', 'PMd-L', 'FEF-L', 'PMv-L',
        'V1-R', 'V3-R', 'SPOC-R', 'AG-R', 'POJ-R', 'SPL-R', 'mIPS-R', 'VIP-R', 'IPL-R', 'STS-R',
        'S1-R', 'M1-R', 'SMA-R', 'PMd-R', 'FEF-R', 'PMv-R'
    ]
    
    xticklabels = region_labels if show_labels else False
    yticklabels = region_labels if show_labels else False
    
    abs_max = max(abs(vmin), abs(vmax))
    symmetric_vmin = -abs_max
    symmetric_vmax = abs_max
    
    sns.heatmap(
        matrix, 
        ax=ax,
        xticklabels=xticklabels, 
        yticklabels=yticklabels,
        cmap='RdBu_r', 
        center=0,
        square=True,
        vmin=symmetric_vmin,
        vmax=symmetric_vmax,
        cbar=False
    )
    
    ax.grid(True, which='both', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    
    if show_labels:
        ax.set_xlabel('Source Region', fontsize=6)
        plt.setp(ax.get_xticklabels(), rotation=90, ha='right', fontsize=4)
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=4)


def lower_triangle_nan(matrix: np.ndarray, include_diagonal: bool = True) -> np.ndarray:
    """Return a copy of `matrix` where everything above the diagonal is NaN.

    Parameters
    ----------
    include_diagonal : bool
        If True, keep diagonal values; if False, set diagonal to NaN as well.
    """
    if matrix is None:
        return None
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("lower_triangle_nan expects a square 2D matrix.")
    n = matrix.shape[0]
    out = matrix.astype(float, copy=True)
    k = 0 if include_diagonal else -1
    # mask positions above allowed triangle
    mask = np.triu(np.ones((n, n), dtype=bool), k=1 if include_diagonal else 0)
    out[mask] = np.nan
    if not include_diagonal:
        out[np.diag_indices(n)] = np.nan
    return out


def build_proanti_diamond(proL: np.ndarray,
                          proR: np.ndarray,
                          antiL: np.ndarray,
                          antiR: np.ndarray,
                          include_diagonal: bool = True) -> np.ndarray:
    """Build a diamond-shaped heatmap from the lower triangles of Pro/Anti matrices.

    Steps
    -----
    1) Compute participant average across the four condition matrices.
    2) Compute deviation matrices as: condition_matrix - avg.
    3) Take only the lower triangle (NaN-mask the upper triangle) of each deviation.
    4) Arrange as a 2N x 2N composite (four right-triangles; right angles at the center).
       Quadrant assignment (as requested):
       - Top-right: ProL
       - Top-left:  AntiL
       - Bottom-right: ProR
       - Bottom-left:  AntiR

       Each condition contributes only its lower triangle. We mirror that lower-triangular image
       so that the triangle's right angle sits at the *center* corner of the quadrant; the
       triangle's hypotenuse then forms one outer side of the diamond.

    Returns
    -------
    np.ndarray
        Composite (2N x 2N) array with NaNs outside the diamond.
    """
    if any(m is None for m in (proL, proR, antiL, antiR)):
        return None
    # ensure float
    mats = [np.asarray(m, dtype=float) for m in (proL, proR, antiL, antiR)]
    n = mats[0].shape[0]
    if any(m.shape != (n, n) for m in mats):
        raise ValueError('All Pro/Anti matrices must have the same square shape.')

    avg = sum(mats) / 4.0
    # Deviation is condition - average ("z-score-like" centering without scaling)
    dev_proL = mats[0] - avg
    dev_proR = mats[1] - avg
    dev_antiL = mats[2] - avg
    dev_antiR = mats[3] - avg

    tri_proL = lower_triangle_nan(dev_proL, include_diagonal=include_diagonal)
    tri_proR = lower_triangle_nan(dev_proR, include_diagonal=include_diagonal)
    tri_antiL = lower_triangle_nan(dev_antiL, include_diagonal=include_diagonal)
    tri_antiR = lower_triangle_nan(dev_antiR, include_diagonal=include_diagonal)

    # To make the hypotenuse of each triangle form the *outer* diamond boundary,
    # we place each triangle with its right angle at the quadrant's inner (center) corner.
    # The base lower-triangle has its right angle at bottom-left.
    # - Top-right quadrant inner corner is bottom-left  -> no flip
    # - Top-left quadrant inner corner is bottom-right  -> horizontal flip
    # - Bottom-right quadrant inner corner is top-left  -> vertical flip
    # - Bottom-left quadrant inner corner is top-right  -> vertical + horizontal flip
    tri_proL_TR = tri_proL
    tri_antiL_TL = np.fliplr(tri_antiL)
    tri_proR_BR = np.flipud(tri_proR)
    tri_antiR_BL = np.flipud(np.fliplr(tri_antiR))

    comp = np.full((2 * n, 2 * n), np.nan, dtype=float)

    # Quadrant placement (requested)
    # Top-left:  AntiL
    comp[0:n, 0:n] = tri_antiL_TL
    # Top-right: ProL
    comp[0:n, n:2*n] = tri_proL_TR
    # Bottom-left: AntiR
    comp[n:2*n, 0:n] = tri_antiR_BL
    # Bottom-right: ProR
    comp[n:2*n, n:2*n] = tri_proR_BR

    return comp

def create_heatmap_facet(dataset='Dataset31_Align_mov', 
                         movement='mov',
                         output_file='combined_heatmaps.png'):
    """
    Create faceted heatmap visualization.

    For each frequency/method combination:

    1) L / R / Subtraction plots (rows = participants, cols = 3)
       - L/R share one scale.
       - Subtraction uses its own scale based only on Sub matrices.

    2) Pro/Anti plots (rows = participants, cols = 4: ProL, ProR, AntiL, AntiR)
       - For movement='cue', filenames are like: cue_L_*_BG_Pro_average*.csv
       - For movement='mov', filenames are like:
         mov_msc_cat_20Hz_BG_Pro_average_NO-LT.csv
         mov_msc_cat_20Hz_BG_Anti_average_NO-LT.csv
         with side encoded in Left_movement / Right_movement folders.
    """
    
    # Get Desktop path
    desktop_path = Path(expanduser("~")) / "Desktop"
    
    # Workspace is at /Users/andy/Desktop/MEG_shared
    base_path = desktop_path / "MEG_shared" / "Data" / dataset / movement
    
    # Handle different folder structures for 'mov' vs 'cue'
    if movement == 'mov':
        left_path = base_path / "Left_movement"
        right_path = base_path / "Right_movement"
        wilcoxon_folder_prefix = "mov_"
        wilcoxon_file_pattern = "{freq_name}_Left-Right_{participant}.csv"
    elif movement == 'cue':
        left_path = base_path / "L"
        right_path = base_path / "R"
        wilcoxon_folder_prefix = "Cue_"
        wilcoxon_file_pattern = "{freq_name}_L-R_{participant}.csv"
    else:
        print(f"Error: Unknown movement type '{movement}'. Expected 'mov' or 'cue'.")
        return
    
    # Find Wilcoxon folder
    wilcoxon_base = find_wilcoxon_folder(base_path)
    if wilcoxon_base is None:
        print(f"Warning: Could not find Wilcoxon folder in {base_path.parent}")
        return
    
    # Create faceted_outputs directory inside Wilcoxon folder
    output_dir = wilcoxon_base / "faceted_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Looking in: {base_path}")
    print(f"Left path: {left_path}")
    print(f"Right path: {right_path}")
    print(f"Wilcoxon folder: {wilcoxon_base}")
    print(f"Output directory: {output_dir}")
    
    # Get all frequency/method folders from left path (e.g., 'msc_cat_10Hz')
    freq_folders = sorted([f for f in left_path.iterdir() if f.is_dir()])
    
    # Process each frequency/method combination
    for freq_folder in freq_folders:
        freq_name = freq_folder.name  # e.g., 'msc_cat_10Hz'
        hz_label = extract_frequency_from_filename(freq_name)
        
        print(f"\nProcessing frequency/method: {freq_name} ({hz_label})")
        
        # Get all participant folders from left side
        participant_folders_left = sorted([p for p in freq_folder.iterdir() if p.is_dir()])
        # Also get participants from right side (for safety)
        right_freq_dir = right_path / freq_name
        if right_freq_dir.exists():
            participant_folders_right = sorted([p for p in right_freq_dir.iterdir() if p.is_dir()])
        else:
            participant_folders_right = []
        
        participants_all = sorted(
            set([p.name for p in participant_folders_left] +
                [p.name for p in participant_folders_right])
        )
        
        # ---------------------------------------------------------------------
        # Part 1: L / R / Subtraction
        # ---------------------------------------------------------------------
        data_structure = {}
        
        for participant in participants_all:
            left_part_dir = left_path / freq_name / participant
            right_part_dir = right_path / freq_name / participant
            
            # Left average file (take first *average*.csv)
            left_files = list(left_part_dir.glob('*average*.csv')) if left_part_dir.exists() else []
            if not left_files:
                print(f"  Warning: No Left average file found for participant {participant}")
                continue
            
            left_file = left_files[0]
            
            # Right file name
            if movement == 'cue':
                right_file_name = left_file.name.replace('cue_L_', 'cue_R_')
                right_file = right_part_dir / right_file_name
            else:
                # For mov, filenames are identical between Left_movement and Right_movement
                right_file = right_part_dir / left_file.name
            
            # Wilcoxon file
            wilcoxon_folder_name = f"{wilcoxon_folder_prefix}{freq_name}"
            wilcoxon_file_name = wilcoxon_file_pattern.format(freq_name=freq_name, participant=participant)
            wilcoxon_file = wilcoxon_base / wilcoxon_folder_name / wilcoxon_file_name
            
            left_matrix = load_matrix_from_file(left_file)
            right_matrix = load_matrix_from_file(right_file)
            sub_matrix = load_matrix_from_file(wilcoxon_file)
            
            if left_matrix is None:
                print(f"  Warning: Could not load Left file for {participant}: {left_file}")
            if right_matrix is None:
                print(f"  Warning: Could not load Right file for {participant}: {right_file}")
            if sub_matrix is None:
                print(f"  Warning: Could not load Wilcoxon file for {participant}: {wilcoxon_file}")
            
            data_structure[participant] = {
                'L': left_matrix,
                'R': right_matrix,
                'Sub': sub_matrix
            }
        
        # Skip if no data found for L/R/Sub
        if not data_structure:
            print(f"  Skipping {freq_name} - no data found for L/R/Sub plots")
        else:
            n_participants = len(data_structure)
            n_cols = 3
            
            left_margin = 0.08
            colorbar_left = 0.88
            colorbar_width = 0.02
            
            fig = plt.figure(figsize=(24, n_participants * 3))
            
            gs = GridSpec(
                n_participants, n_cols, figure=fig,
                hspace=0.12, wspace=0.03,
                left=left_margin, right=colorbar_left,
                top=0.96, bottom=0.04
            )
            
            # Separate scales for L/R vs Sub
            all_values_lr = []
            all_values_sub = []
            for participant in data_structure:
                for matrix_type in ['L', 'R']:
                    matrix = data_structure[participant][matrix_type]
                    if matrix is not None:
                        all_values_lr.append(matrix.flatten())
                matrix_sub = data_structure[participant]['Sub']
                if matrix_sub is not None:
                    all_values_sub.append(matrix_sub.flatten())
            
            if all_values_lr:
                all_values_lr = np.concatenate(all_values_lr)
                vmin_lr, vmax_lr = np.min(all_values_lr), np.max(all_values_lr)
                print(f"  [L/R] Color scale: vmin={vmin_lr:.4f}, vmax={vmax_lr:.4f}")
            else:
                vmin_lr, vmax_lr = -1, 1
            
            if all_values_sub:
                all_values_sub = np.concatenate(all_values_sub)
                vmin_sub, vmax_sub = np.min(all_values_sub), np.max(all_values_sub)
                print(f"  [Sub] Color scale: vmin={vmin_sub:.4f}, vmax={vmax_sub:.4f}")
            else:
                vmin_sub, vmax_sub = vmin_lr, vmax_lr
                print(f"  [Sub] No Sub matrices found, falling back to L/R scale.")
            
            participants = sorted(data_structure.keys())
            
            # Collect lower-triangle vectors for PCA across all matrices in this figure
            pca_vectors_lrsub = []
            pca_labels_lrsub = []
            
            for row_idx, participant in enumerate(participants):
                show_labels = True
                
                # L
                ax_l = fig.add_subplot(gs[row_idx, 0])
                matrix_l = data_structure[participant]['L']
                if matrix_l is not None:
                    create_heatmap_subplot(matrix_l, ax_l, vmin_lr, vmax_lr, show_labels=show_labels)
                    try:
                        vec_l = get_lower_triangle_vector(matrix_l, include_diagonal=False)
                        pca_vectors_lrsub.append(vec_l)
                        pca_labels_lrsub.append(f"{participant}_L")
                    except ValueError as e:
                        print(f"  [PCA] Skipping L matrix for {participant}: {e}")
                ax_l.set_ylabel(participant, fontsize=8, fontweight='bold')
                if row_idx == 0:
                    ax_l.set_title('LEFT MOVEMENT (proL + antiR)', fontsize=10, fontweight='bold')
                
                # R
                ax_r = fig.add_subplot(gs[row_idx, 1])
                matrix_r = data_structure[participant]['R']
                if matrix_r is not None:
                    create_heatmap_subplot(matrix_r, ax_r, vmin_lr, vmax_lr, show_labels=show_labels)
                    try:
                        vec_r = get_lower_triangle_vector(matrix_r, include_diagonal=False)
                        pca_vectors_lrsub.append(vec_r)
                        pca_labels_lrsub.append(f"{participant}_R")
                    except ValueError as e:
                        print(f"  [PCA] Skipping R matrix for {participant}: {e}")
                if row_idx == 0:
                    ax_r.set_title('RIGHT MOVEMENT (proR + antiL)', fontsize=10, fontweight='bold')
                
                # Sub
                ax_sub = fig.add_subplot(gs[row_idx, 2])
                matrix_sub = data_structure[participant]['Sub']
                if matrix_sub is not None:
                    create_heatmap_subplot(matrix_sub, ax_sub, vmin_sub, vmax_sub, show_labels=show_labels)
                    try:
                        vec_sub = get_lower_triangle_vector(matrix_sub, include_diagonal=False)
                        pca_vectors_lrsub.append(vec_sub)
                        pca_labels_lrsub.append(f"{participant}_Sub")
                    except ValueError as e:
                        print(f"  [PCA] Skipping Sub matrix for {participant}: {e}")
                if row_idx == 0:
                    ax_sub.set_title('SUBTRACTION ((proL + antiR) - (proR + antiL))', fontsize=10, fontweight='bold')
            
            from matplotlib.colors import TwoSlopeNorm
            
            abs_max_lr = max(abs(vmin_lr), abs(vmax_lr))
            norm_lr = TwoSlopeNorm(vmin=-abs_max_lr, vcenter=0, vmax=abs_max_lr)
            cbar_ax_lr = fig.add_axes([colorbar_left + 0.005, 0.55, colorbar_width, 0.3])
            sm_lr = plt.cm.ScalarMappable(cmap='RdBu_r', norm=norm_lr)
            sm_lr.set_array([])
            cbar_lr = fig.colorbar(sm_lr, cax=cbar_ax_lr, label='Value (L/R)')
            cbar_lr.ax.tick_params(labelsize=8)
            print(f"  [L/R] Colorbar: -{abs_max_lr:.4f} to {abs_max_lr:.4f}")
            
            abs_max_sub = max(abs(vmin_sub), abs(vmax_sub))
            norm_sub = TwoSlopeNorm(vmin=-abs_max_sub, vcenter=0, vmax=abs_max_sub)
            cbar_ax_sub = fig.add_axes([colorbar_left + 0.005, 0.15, colorbar_width, 0.3])
            sm_sub = plt.cm.ScalarMappable(cmap='RdBu_r', norm=norm_sub)
            sm_sub.set_array([])
            cbar_sub = fig.colorbar(sm_sub, cax=cbar_ax_sub, label='Value (Subtraction)')
            cbar_sub.ax.tick_params(labelsize=8)
            print(f"  [Sub] Colorbar: -{abs_max_sub:.4f} to {abs_max_sub:.4f}")
            
            # Run PCA on all lower-triangle vectors for this L/R/Sub figure
            try:
                pca_title = f'{dataset} - {freq_name}: PCA of L/R/Sub matrices'
                pca_output_filename = output_file.replace('.png', f'_{freq_name}_LRSub_PCA.png')
                pca_output_path = output_dir / pca_output_filename

                if pca_vectors_lrsub:
                    base_len = len(pca_vectors_lrsub[0])
                    valid_vectors = []
                    valid_labels = []
                    for vec, lab in zip(pca_vectors_lrsub, pca_labels_lrsub):
                        if len(vec) == base_len:
                            valid_vectors.append(vec)
                            valid_labels.append(lab)
                        else:
                            print(f"  [PCA] Skipping {lab}: inconsistent vector length.")
                    if len(valid_vectors) >= 2:
                        run_pca_and_plot(valid_vectors, valid_labels, pca_title, pca_output_path)
                    else:
                        print("  [PCA] Not enough valid vectors for L/R/Sub PCA after length check.")
                else:
                    print("  [PCA] No matrices collected for L/R/Sub PCA.")
            except Exception as e:
                print(f"  [PCA] Error while computing L/R/Sub PCA: {e}")
            
            output_filename = output_file.replace('.png', f'_{freq_name}_LRSub.png')
            output_path = output_dir / output_filename
            
            plt.suptitle(
                f'{dataset} - {freq_name}: L/R/Subtraction Heatmaps by Participant',
                fontsize=14, fontweight='bold', y=0.998
            )
            
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"  Saved L/R/Sub heatmap to: {output_path}")
            plt.close()

        
        # ---------------------------------------------------------------------
        # Part 2: Pro/Anti -> participant-normalized deviations -> diamond plot
        # ---------------------------------------------------------------------
        # For each participant we:
        #   1) Load ProL, ProR, AntiL, AntiR matrices
        #   2) Compute participant average across these 4
        #   3) Compute deviations as (avg - condition)
        #   4) Keep only lower triangles, mirror as requested, and combine into
        #      a single 2N x 2N "diamond" heatmap per participant.

        cond_data = {}

        # Recompute participants for this freq
        participants_left = []
        if (left_path / freq_name).exists():
            participants_left = [p.name for p in (left_path / freq_name).iterdir() if p.is_dir()]
        participants_right = []
        if (right_path / freq_name).exists():
            participants_right = [p.name for p in (right_path / freq_name).iterdir() if p.is_dir()]

        participants_all_cond = sorted(set(participants_left + participants_right))

        for participant in participants_all_cond:
            left_part_dir = left_path / freq_name / participant
            right_part_dir = right_path / freq_name / participant

            proL_matrix = None
            antiL_matrix = None
            proR_matrix = None
            antiR_matrix = None

            # Left side
            if left_part_dir.exists():
                if movement == 'cue':
                    proL_files = list(left_part_dir.glob(f"{movement}_L_*_{participant}_Pro_average*.csv"))
                    antiL_files = list(left_part_dir.glob(f"{movement}_L_*_{participant}_Anti_average*.csv"))
                else:  # mov
                    proL_files = list(left_part_dir.glob(f"{movement}_{freq_name}_{participant}_Pro_average*.csv"))
                    antiL_files = list(left_part_dir.glob(f"{movement}_{freq_name}_{participant}_Anti_average*.csv"))

                if proL_files:
                    proL_matrix = load_matrix_from_file(proL_files[0])
                if antiL_files:
                    antiL_matrix = load_matrix_from_file(antiL_files[0])

            # Right side
            if right_part_dir.exists():
                if movement == 'cue':
                    proR_files = list(right_part_dir.glob(f"{movement}_R_*_{participant}_Pro_average*.csv"))
                    antiR_files = list(right_part_dir.glob(f"{movement}_R_*_{participant}_Anti_average*.csv"))
                else:  # mov
                    proR_files = list(right_part_dir.glob(f"{movement}_{freq_name}_{participant}_Pro_average*.csv"))
                    antiR_files = list(right_part_dir.glob(f"{movement}_{freq_name}_{participant}_Anti_average*.csv"))

                if proR_files:
                    proR_matrix = load_matrix_from_file(proR_files[0])
                if antiR_files:
                    antiR_matrix = load_matrix_from_file(antiR_files[0])

            cond_data[participant] = {
                'ProL': proL_matrix,
                'ProR': proR_matrix,
                'AntiL': antiL_matrix,
                'AntiR': antiR_matrix
            }

        # Build diamond composites
        diamond_data = {}
        for participant, mats in cond_data.items():
            try:
                comp = build_proanti_diamond(mats['ProL'], mats['ProR'], mats['AntiL'], mats['AntiR'], include_diagonal=True)
            except Exception as e:
                print(f"  [Pro/Anti diamond] Skipping {participant}: {e}")
                comp = None
            if comp is not None:
                diamond_data[participant] = comp

        if not diamond_data:
            print(f"  Skipping {freq_name} Pro/Anti diamond - no complete Pro/Anti sets found")
        else:
            # Determine a global symmetric color scale across all participants' composites
            all_vals = []
            for comp in diamond_data.values():
                finite = comp[np.isfinite(comp)]
                if finite.size:
                    all_vals.append(finite)
            if all_vals:
                all_vals = np.concatenate(all_vals)
                vmin_d, vmax_d = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
            else:
                vmin_d, vmax_d = -1.0, 1.0

            abs_max_d = max(abs(vmin_d), abs(vmax_d))
            print(f"  [Pro/Anti diamond] Colorbar: -{abs_max_d:.4f} to {abs_max_d:.4f}")

            participants_cond = sorted(diamond_data.keys())
            n_participants_cond = len(participants_cond)

            # One column per participant (diamond)
            fig2 = plt.figure(figsize=(10, n_participants_cond * 3.2))
            gs2 = GridSpec(
                n_participants_cond, 1, figure=fig2,
                hspace=0.18, wspace=0.0,
                left=0.10, right=0.86,
                top=0.96, bottom=0.04
            )

            for row_idx, participant in enumerate(participants_cond):
                ax = fig2.add_subplot(gs2[row_idx, 0])
                comp = diamond_data[participant]

                sns.heatmap(
                    comp,
                    ax=ax,
                    cmap='RdBu_r',
                    center=0,
                    square=True,
                    vmin=-abs_max_d,
                    vmax=abs_max_d,
                    cbar=False,
                    xticklabels=False,
                    yticklabels=False,
                    mask=~np.isfinite(comp)
                )

                # Draw axes to split the diamond into its four quadrants
                # (center lines between pixels)
                n2 = comp.shape[0]
                mid = n2 // 2
                ax.axhline(mid, color='k', linewidth=0.25)
                ax.axvline(mid, color='k', linewidth=0.25)


                ax.set_ylabel(participant, fontsize=9, fontweight='bold', rotation=0, labelpad=35)
                ax.set_xlabel('')
                ax.set_title('' if row_idx else 'Participant Pro/Anti deviations (diamond)', fontsize=10, fontweight='bold')

            from matplotlib.colors import TwoSlopeNorm
            norm_d = TwoSlopeNorm(vmin=-abs_max_d, vcenter=0, vmax=abs_max_d)
            cbar_ax2 = fig2.add_axes([0.88, 0.15, 0.03, 0.7])
            sm2 = plt.cm.ScalarMappable(cmap='RdBu_r', norm=norm_d)
            sm2.set_array([])
            cbar2 = fig2.colorbar(sm2, cax=cbar_ax2, label='Deviation (condition - avg)')
            cbar2.ax.tick_params(labelsize=8)

            output_filename_cond = output_file.replace('.png', f'_{freq_name}_ProAnti_Diamond.png')
            output_path_cond = output_dir / output_filename_cond

            plt.suptitle(
                f'{dataset} - {freq_name}: Pro/Anti participant-normalized diamond heatmaps',
                fontsize=14, fontweight='bold', y=0.998
            )

            plt.savefig(output_path_cond, dpi=300, bbox_inches='tight')
            print(f"  Saved Pro/Anti diamond heatmap to: {output_path_cond}")
            plt.close()

        # -----------------------------------------------------------------
        # Part 3: cat vs mean correlations (per subject, per condition)
        # -----------------------------------------------------------------
        try:
            _ = correlate_cat_vs_mean_for_frequency(
                dataset=dataset,
                movement=movement,
                freq_name_cat=freq_name,
                left_path=left_path,
                right_path=right_path,
                output_dir=output_dir,
            )
        except Exception as e:
            print(f"  [cat-mean] Error while computing correlations for {freq_name}: {e}")

# Example usage:
create_heatmap_facet(dataset='DataSet31_Align_mov', movement='mov', output_file='dataset31_mov_heatmaps.png')
