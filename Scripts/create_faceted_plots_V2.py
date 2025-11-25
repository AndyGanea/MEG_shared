import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
import pandas as pd
from matplotlib.gridspec import GridSpec
from os.path import expanduser

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
    Run PCA on a list of 1D vectors and save a 2D scatter plot (PC1 vs PC2).

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

    # Use first two PCs
    pc1 = scores[:, 0]
    pc2 = scores[:, 1] if scores.shape[1] > 1 else np.zeros_like(pc1)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(pc1, pc2, alpha=0.8)

    # Label each point with participant + condition
    for x, y, lab in zip(pc1, pc2, labels):
        ax.text(x, y, lab, fontsize=7, ha='center', va='center')

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title(title)
    ax.axhline(0, color='grey', linewidth=0.5, linestyle='--', alpha=0.5)
    ax.axvline(0, color='grey', linewidth=0.5, linestyle='--', alpha=0.5)
    fig.tight_layout()

    save_path = Path(save_path)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  [PCA] Saved PCA plot to: {save_path}")


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
        # Part 2: ProL / ProR / AntiL / AntiR (now supports mov + cue)
        # ---------------------------------------------------------------------
        cond_data = {}

        # Recompute participants for this freq (in case L/RSub was skipped)
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
                else:  # mov: filenames like mov_msc_cat_20Hz_BG_Pro_average_NO-LT.csv
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

        # Check if at least one Pro/Anti matrix exists
        any_matrix = any(
            m is not None
            for participant_mats in cond_data.values()
            for m in participant_mats.values()
        )
        if not any_matrix:
            print(f"  Skipping {freq_name} Pro/Anti - no Pro/Anti averaged files found")
            continue

        # Global min/max for Pro/Anti
        all_values_cond = []
        for participant, mats in cond_data.items():
            for key in ['ProL', 'ProR', 'AntiL', 'AntiR']:
                mat = mats[key]
                if mat is not None:
                    all_values_cond.append(mat.flatten())
        all_values_cond = np.concatenate(all_values_cond)
        vmin_c, vmax_c = np.min(all_values_cond), np.max(all_values_cond)
        print(f"  [Pro/Anti] Color scale: vmin={vmin_c:.4f}, vmax={vmax_c:.4f}")

        # Figure: rows = participants, cols = 4 (ProL, ProR, AntiL, AntiR)
        participants_cond = sorted(cond_data.keys())
        n_participants_cond = len(participants_cond)
        n_cols_cond = 4

        left_margin = 0.08
        colorbar_left = 0.88
        colorbar_width = 0.02

        fig2 = plt.figure(figsize=(26, n_participants_cond * 3))

        gs2 = GridSpec(
            n_participants_cond, n_cols_cond, figure=fig2,
            hspace=0.12, wspace=0.03,
            left=left_margin, right=colorbar_left,
            top=0.96, bottom=0.04
        )

        # Collect lower-triangle vectors for PCA across all Pro/Anti matrices in this figure
        pca_vectors_proanti = []
        pca_labels_proanti = []

        for row_idx, participant in enumerate(participants_cond):
            mats = cond_data[participant]
            show_labels = True

            # ProL
            ax_pl = fig2.add_subplot(gs2[row_idx, 0])
            if mats['ProL'] is not None:
                create_heatmap_subplot(mats['ProL'], ax_pl, vmin_c, vmax_c, show_labels=show_labels)
                try:
                    vec_pl = get_lower_triangle_vector(mats['ProL'], include_diagonal=False)
                    pca_vectors_proanti.append(vec_pl)
                    pca_labels_proanti.append(f"{participant}_ProL")
                except ValueError as e:
                    print(f"  [PCA] Skipping ProL for {participant}: {e}")
            else:
                ax_pl.axis('off')
            ax_pl.set_ylabel(participant, fontsize=8, fontweight='bold')
            if row_idx == 0:
                ax_pl.set_title('Pro Left (ProL)', fontsize=10, fontweight='bold')

            # ProR
            ax_pr = fig2.add_subplot(gs2[row_idx, 1])
            if mats['ProR'] is not None:
                create_heatmap_subplot(mats['ProR'], ax_pr, vmin_c, vmax_c, show_labels=show_labels)
                try:
                    vec_pr = get_lower_triangle_vector(mats['ProR'], include_diagonal=False)
                    pca_vectors_proanti.append(vec_pr)
                    pca_labels_proanti.append(f"{participant}_ProR")
                except ValueError as e:
                    print(f"  [PCA] Skipping ProR for {participant}: {e}")
            else:
                ax_pr.axis('off')
            if row_idx == 0:
                ax_pr.set_title('Pro Right (ProR)', fontsize=10, fontweight='bold')

            # AntiL
            ax_al = fig2.add_subplot(gs2[row_idx, 2])
            if mats['AntiL'] is not None:
                create_heatmap_subplot(mats['AntiL'], ax_al, vmin_c, vmax_c, show_labels=show_labels)
                try:
                    vec_al = get_lower_triangle_vector(mats['AntiL'], include_diagonal=False)
                    pca_vectors_proanti.append(vec_al)
                    pca_labels_proanti.append(f"{participant}_AntiL")
                except ValueError as e:
                    print(f"  [PCA] Skipping AntiL for {participant}: {e}")
            else:
                ax_al.axis('off')
            if row_idx == 0:
                ax_al.set_title('Anti Left (AntiL)', fontsize=10, fontweight='bold')

            # AntiR
            ax_ar = fig2.add_subplot(gs2[row_idx, 3])
            if mats['AntiR'] is not None:
                create_heatmap_subplot(mats['AntiR'], ax_ar, vmin_c, vmax_c, show_labels=show_labels)
                try:
                    vec_ar = get_lower_triangle_vector(mats['AntiR'], include_diagonal=False)
                    pca_vectors_proanti.append(vec_ar)
                    pca_labels_proanti.append(f"{participant}_AntiR")
                except ValueError as e:
                    print(f"  [PCA] Skipping AntiR for {participant}: {e}")
            else:
                ax_ar.axis('off')
            if row_idx == 0:
                ax_ar.set_title('Anti Right (AntiR)', fontsize=10, fontweight='bold')

        from matplotlib.colors import TwoSlopeNorm
        abs_max_c = max(abs(vmin_c), abs(vmax_c))
        norm_c = TwoSlopeNorm(vmin=-abs_max_c, vcenter=0, vmax=abs_max_c)

        cbar_ax2 = fig2.add_axes([colorbar_left + 0.005, 0.15, colorbar_width, 0.7])
        sm2 = plt.cm.ScalarMappable(cmap='RdBu_r', norm=norm_c)
        sm2.set_array([])
        cbar2 = fig2.colorbar(sm2, cax=cbar_ax2, label='Value')
        cbar2.ax.tick_params(labelsize=8)

        print(f"  [Pro/Anti] Colorbar: -{abs_max_c:.4f} to {abs_max_c:.4f}")

        # Run PCA on all lower-triangle vectors for this Pro/Anti figure
        try:
            pca_title_cond = f'{dataset} - {freq_name}: PCA of Pro/Anti matrices'
            pca_output_filename_cond = output_file.replace('.png', f'_{freq_name}_ProAnti_PCA.png')
            pca_output_path_cond = output_dir / pca_output_filename_cond

            if pca_vectors_proanti:
                base_len_cond = len(pca_vectors_proanti[0])
                valid_vectors_cond = []
                valid_labels_cond = []
                for vec, lab in zip(pca_vectors_proanti, pca_labels_proanti):
                    if len(vec) == base_len_cond:
                        valid_vectors_cond.append(vec)
                        valid_labels_cond.append(lab)
                    else:
                        print(f"  [PCA] Skipping {lab}: inconsistent vector length.")
                if len(valid_vectors_cond) >= 2:
                    run_pca_and_plot(valid_vectors_cond, valid_labels_cond, pca_title_cond, pca_output_path_cond)
                else:
                    print("  [PCA] Not enough valid vectors for Pro/Anti PCA after length check.")
            else:
                print("  [PCA] No matrices collected for Pro/Anti PCA.")
        except Exception as e:
            print(f"  [PCA] Error while computing Pro/Anti PCA: {e}")


        output_filename_cond = output_file.replace('.png', f'_{freq_name}_ProAnti.png')
        output_path_cond = output_dir / output_filename_cond



        plt.suptitle(
            f'{dataset} - {freq_name}: ProL/ProR/AntiL/AntiR Heatmaps by Participant',
            fontsize=14, fontweight='bold', y=0.998
        )

        plt.savefig(output_path_cond, dpi=300, bbox_inches='tight')
        print(f"  Saved Pro/Anti heatmap to: {output_path_cond}")
        plt.close()

# Example usage:
create_heatmap_facet(dataset='Dataset31_Align_mov', movement='mov', output_file='dataset31_mov_heatmaps.png')
