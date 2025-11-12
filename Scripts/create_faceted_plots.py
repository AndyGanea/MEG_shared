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
    """Extract frequency from filename (e.g., 'mov_msc_cat_10Hz_BG_...' -> '10Hz')"""
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

def create_heatmap_subplot(matrix: np.ndarray, ax, vmin: float, vmax: float, 
                           show_labels: bool = True):
    """
    Create a heatmap subplot using the same style as meg_wilcoxon_analysis_V3.py
    Uses raw values with consistent vmin/vmax across all plots (no normalization)
    
    Parameters:
    -----------
    matrix : np.ndarray
        Matrix data to plot
    ax : matplotlib.axes.Axes
        Axes object to plot on
    vmin : float
        Minimum value for color scale (for consistency across all heatmaps)
    vmax : float
        Maximum value for color scale (for consistency across all heatmaps)
    show_labels : bool
        Whether to show region labels (now shown on all graphs)
    """
    # Create region labels as provided - updated to include all 32 regions
    region_labels = [
        'V1-L', 'V3-L', 'SPOC-L', 'AG-L', 'POJ-L', 'SPL-L', 'mIPS-L', 'VIP-L', 'IPL-L', 'STS-L',
        'S1-L', 'M1-L', 'SMA-L', 'PMd-L', 'FEF-L', 'PMv-L',
        'V1-R', 'V3-R', 'SPOC-R', 'AG-R', 'POJ-R', 'SPL-R', 'mIPS-R', 'VIP-R', 'IPL-R', 'STS-R',
        'S1-R', 'M1-R', 'SMA-R', 'PMd-R', 'FEF-R', 'PMv-R'
    ]
    
    # Show labels on all graphs with smaller font to avoid overlap
    xticklabels = region_labels if show_labels else False
    yticklabels = region_labels if show_labels else False
    
    # Create heatmap with raw values, using consistent vmin/vmax
    # Make symmetric around 0 to properly center the colormap
    # This ensures 0 values appear as white (center of RdBu_r colormap)
    abs_max = max(abs(vmin), abs(vmax))
    symmetric_vmin = -abs_max
    symmetric_vmax = abs_max
    
    # This matches the style from meg_wilcoxon_analysis_V3.py
    sns.heatmap(matrix, 
               ax=ax,
               xticklabels=xticklabels, 
               yticklabels=yticklabels,
               cmap='RdBu_r', 
               center=0,
               square=True,
               vmin=symmetric_vmin,
               vmax=symmetric_vmax,
               cbar=False)
    
    # Add gridlines to all heatmaps (matching original style)
    ax.grid(True, which='both', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    
    if show_labels:
        ax.set_xlabel('Target Region', fontsize=6)
        # Note: ylabel (Source Region) is not set here to allow participant name to be set separately
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=4)
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=4)

def create_heatmap_facet(dataset='Dataset31_Align_mov', 
                         movement='mov',
                         output_file='combined_heatmaps.png'):
    """
    Create faceted heatmap visualization: L, R, Subtraction for each participant and frequency/method
    
    For each frequency/method combination:
    - Creates a figure with 10 rows (participants) × 3 columns (Left, Right, Subtraction)
    - Each row shows 3 heatmaps for one participant
    
    Parameters:
    -----------
    dataset : str
        Dataset folder name (default: 'Dataset31_Align_mov')
    movement : str
        Movement type (default: 'mov')
    output_file : str
        Output filename pattern (will be modified per frequency/method)
    """
    
    # Get Desktop path
    desktop_path = Path(expanduser("~")) / "Desktop"
    
    # Define paths based on your directory structure
    # Workspace is at /Users/andy/Desktop/MEG_shared
    base_path = desktop_path / "MEG_shared" / "Data" / dataset / movement
    left_path = base_path / "Left_movement"
    right_path = base_path / "Right_movement"
    
    # Find Wilcoxon folder
    wilcoxon_base = find_wilcoxon_folder(base_path)
    if wilcoxon_base is None:
        print(f"Warning: Could not find Wilcoxon folder in {base_path.parent}")
        return
    
    # Create faceted_outputs directory inside Wilcoxon folder
    output_dir = wilcoxon_base / "faceted_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Looking in: {base_path}")
    print(f"Wilcoxon folder: {wilcoxon_base}")
    print(f"Output directory: {output_dir}")
    
    # Get all frequency/method folders from Left_movement (e.g., 'msc_cat_10Hz')
    freq_folders = sorted([f for f in left_path.iterdir() if f.is_dir()])
    
    # Process each frequency/method combination
    for freq_folder in freq_folders:
        freq_name = freq_folder.name  # e.g., 'msc_cat_10Hz'
        hz_label = extract_frequency_from_filename(freq_name)
        
        print(f"\nProcessing frequency/method: {freq_name} ({hz_label})")
        
        # Get all participant folders (BG, DOC, GB, JDC, JFXD, JZ, LT, NvA, RR, SJB)
        participant_folders = sorted([p for p in freq_folder.iterdir() if p.is_dir()])
        
        # Collect data for this frequency/method: {participant: {'L': matrix, 'R': matrix, 'Sub': matrix}}
        data_structure = {}
        
        for participant_folder in participant_folders:
            participant = participant_folder.name  # e.g., 'BG'
            
            # Find the average file in Left_movement
            # Pattern: mov_msc_cat_10Hz_BG_average_NO-LT.csv
            left_files = list(participant_folder.glob('*average*.csv'))
            if not left_files:
                print(f"  Warning: No average file found for participant {participant}")
                continue
            
            left_file = left_files[0]  # Take the first average file found
            
            # Construct corresponding Right path
            right_file = right_path / freq_name / participant / left_file.name
            
            # Construct corresponding Wilcoxon path
            # Pattern: msc_cat_10Hz_Left-Right_BG.csv
            wilcoxon_file = wilcoxon_base / f"mov_{freq_name}" / f"{freq_name}_Left-Right_{participant}.csv"
            
            # Load matrices
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
        
        # Skip if no data found
        if not data_structure:
            print(f"  Skipping {freq_name} - no data found")
            continue
        
        # Create figure for this frequency/method: 10 rows × 3 columns
        n_participants = len(data_structure)
        n_cols = 3
        # Increase figure size and reduce spacing to make heatmaps occupy more space
        fig = plt.figure(figsize=(20, n_participants * 3))
        gs = GridSpec(n_participants, n_cols, figure=fig, hspace=0.15, wspace=0.15, 
                     left=0.05, right=0.88, top=0.95, bottom=0.05)
        
        # Find global min/max for consistent color scaling across all 3 types
        # Calculate from all values (all participants, all matrix types)
        all_values = []
        for participant in data_structure:
            for matrix_type in ['L', 'R', 'Sub']:
                matrix = data_structure[participant][matrix_type]
                if matrix is not None:
                    all_values.append(matrix.flatten())
        
        if all_values:
            all_values = np.concatenate(all_values)
            # Use min/max of all values for consistent scaling
            vmin, vmax = np.min(all_values), np.max(all_values)
            print(f"  Color scale: vmin={vmin:.4f}, vmax={vmax:.4f}")
        else:
            vmin, vmax = -1, 1
        
        # Plot all heatmaps
        participants = sorted(data_structure.keys())
        
        for row_idx, participant in enumerate(participants):
            # Show labels on all graphs now
            show_labels = True
            
            # Plot L (Left)
            ax_l = fig.add_subplot(gs[row_idx, 0])
            matrix_l = data_structure[participant]['L']
            if matrix_l is not None:
                create_heatmap_subplot(matrix_l, ax_l, vmin, vmax, show_labels=show_labels)
            # Show participant name as ylabel on all rows
            ax_l.set_ylabel(participant, fontsize=8, fontweight='bold')
            if row_idx == 0:
                ax_l.set_title('LEFT', fontsize=10, fontweight='bold')
            
            # Plot R (Right)
            ax_r = fig.add_subplot(gs[row_idx, 1])
            matrix_r = data_structure[participant]['R']
            if matrix_r is not None:
                create_heatmap_subplot(matrix_r, ax_r, vmin, vmax, show_labels=show_labels)
            if row_idx == 0:
                ax_r.set_title('RIGHT', fontsize=10, fontweight='bold')
            
            # Plot Subtraction
            ax_sub = fig.add_subplot(gs[row_idx, 2])
            matrix_sub = data_structure[participant]['Sub']
            if matrix_sub is not None:
                create_heatmap_subplot(matrix_sub, ax_sub, vmin, vmax, show_labels=show_labels)
            if row_idx == 0:
                ax_sub.set_title('SUBTRACTION', fontsize=10, fontweight='bold')
        
        # Add colorbar with raw values (no normalization)
        # Position colorbar to the right of the plots
        # Use TwoSlopeNorm to properly center at 0, matching the center=0 in heatmap
        from matplotlib.colors import TwoSlopeNorm
        abs_max = max(abs(vmin), abs(vmax))
        norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
        
        cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.7])
        sm = plt.cm.ScalarMappable(cmap='RdBu_r', norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax, label='Value')
        cbar.ax.tick_params(labelsize=8)
        
        # Verify the colorbar range matches vmin/vmax
        print(f"  Color scale: vmin={vmin:.4f}, vmax={vmax:.4f}")
        print(f"  Colorbar range: {cbar.vmin:.4f} to {cbar.vmax:.4f}")
        print(f"  Using symmetric range: -{abs_max:.4f} to {abs_max:.4f} (centered at 0)")
        
        # Create output filename with frequency/method name
        output_filename = output_file.replace('.png', f'_{freq_name}.png')
        output_path = output_dir / output_filename
        
        plt.suptitle(f'{dataset} - {freq_name}: L-R-Subtraction Heatmaps by Participant',
                     fontsize=14, fontweight='bold', y=0.998)
        
        # Save
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved heatmap to: {output_path}")
        plt.close()

# Example usage:
create_heatmap_facet(dataset='Dataset31_Align_mov', movement='mov', output_file='dataset31_heatmaps.png')
