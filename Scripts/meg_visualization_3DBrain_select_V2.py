#!/usr/bin/env python3

"""
MEG 3D Brain Visualization with File Selection
---------------------------------------------
This script allows the user to select a file from the Visual folder
and create 3D brain visualizations.
"""

import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import logging
from mpl_toolkits.mplot3d import Axes3D

# Import nilearn for brain visualization
try:
    from nilearn import datasets, plotting, surface
    import nibabel as nib
    NILEARN_AVAILABLE = True
except ImportError:
    print("Installing required libraries...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "nilearn"])
    from nilearn import datasets, plotting, surface
    import nibabel as nib
    NILEARN_AVAILABLE = True

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logging.info("Starting MEG 3D Brain Visualization with File Selection")

# Define the Visual folder path
visual_folder = Path("Data/Visual")
visual_folder.mkdir(exist_ok=True, parents=True)

# Function to select a file from the Visual folder
def select_file():
    """Let the user select a file from the Visual folder"""
    # Find all CSV files in the Visual folder
    csv_files = list(visual_folder.glob("*.csv"))
    
    if not csv_files:
        logging.error(f"No CSV files found in {visual_folder}")
        sys.exit(1)
    
    # Categorize files into regular and NO-LT files
    regular_files = []
    no_lt_files = []
    
    for file_path in csv_files:
        if "_NO-LT" in file_path.name:
            no_lt_files.append(file_path)
        else:
            regular_files.append(file_path)
    
    print("\nAvailable files:")
    print("================")
    
    file_counter = 1
    all_files = []  # Keep track of all files in order
    
    # Display regular files first
    if regular_files:
        print("\nRegular Files:")
        print("--------------")
        for file_path in regular_files:
            print(f"{file_counter}. {file_path.name}")
            all_files.append(file_path)
            file_counter += 1
    
    # Display NO-LT files
    if no_lt_files:
        print("\nNO-LT Files:")
        print("------------")
        for file_path in no_lt_files:
            print(f"{file_counter}. {file_path.name}")
            all_files.append(file_path)
            file_counter += 1
    
    # Log file categorization
    logging.info(f"Found {len(regular_files)} regular files and {len(no_lt_files)} NO-LT files")
    
    while True:
        try:
            choice = int(input(f"\nSelect file number (1-{len(all_files)}): "))
            if 1 <= choice <= len(all_files):
                selected_file = all_files[choice-1]
                is_no_lt = "_NO-LT" in selected_file.name
                logging.info(f"Selected {'NO-LT' if is_no_lt else 'regular'} file: {selected_file.name}")
                return selected_file
            else:
                print(f"Please enter a number between 1 and {len(all_files)}")
        except ValueError:
            print("Please enter a valid number")

# Function to select connection display mode
def select_connection_display():
    """Let user select how to display connections"""
    print("\nHow would you like to display connections?")
    print("====================================")
    print("1. All connections in one visualization")
    print("2. Separate positive and negative connections")
    
    while True:
        try:
            choice = int(input("\nSelect option (1 or 2): "))
            if choice == 1:
                return "all"
            elif choice == 2:
                return "separate"
            else:
                print("Please enter either 1 or 2")
        except ValueError:
            print("Please enter a valid number")

# Define region labels in the exact order they appear in the CSV file
region_labels = [
    'V1-L', 'V3-L', 'SPOC-L', 'AG-L', 'POJ-L', 'SPL-L', 'mIPS-L', 'VIP-L', 'IPL-L', 'STS-L',
    'S1-L', 'M1-L', 'SMA-L', 'PMd-L', 'FEF-L', 'PMv-L',
    'V1-R', 'V3-R', 'SPOC-R', 'AG-R', 'POJ-R', 'SPL-R', 'mIPS-R', 'VIP-R', 'IPL-R', 'STS-R',
    'S1-R', 'M1-R', 'SMA-R', 'PMd-R', 'FEF-R', 'PMv-R'
]

# Define region categories
region_categories = {
    'Visual': ['V1-L', 'V3-L', 'VIP-L', 'V1-R', 'V3-R', 'VIP-R'],
    'Parietal': ['SPOC-L', 'AG-L', 'POJ-L', 'SPL-L', 'mIPS-L', 'IPL-L',
                 'SPOC-R', 'AG-R', 'POJ-R', 'SPL-R', 'mIPS-R', 'IPL-R'],
    'Temporal': ['STS-L', 'STS-R'],
    'Sensorimotor': ['S1-L', 'M1-L', 'S1-R', 'M1-R'],
    'Frontal': ['SMA-L', 'PMd-L', 'FEF-L', 'PMv-L', 'SMA-R', 'PMd-R', 'FEF-R', 'PMv-R']
}

# Define colors for region categories
category_colors = {
    'Visual': '#1f77b4',      # Blue
    'Parietal': '#ff7f0e',    # Orange
    'Temporal': '#2ca02c',    # Green
    'Sensorimotor': '#d62728', # Red
    'Frontal': '#9467bd'      # Purple
}

# Add Talairach coordinates
talairach_coordinates = {
    # Left hemisphere
    'V1-L': [-8, -91, 0],
    'V3-L': [-21, -85, 16],
    'VIP-L': [-37, -40, 44],
    'SPOC-L': [-9, -71, 37],
    'AG-L': [-35, -61, 35],
    'POJ-L': [-18, -79, 43],
    'SPL-L': [-23, -54, 46],
    'mIPS-L': [-22, -61, 40],
    'IPL-L': [-43, -35, 49],
    'STS-L': [-45, -57, 15],
    'S1-L': [-40, -26, 48],
    'M1-L': [-35, -23, 54],
    'SMA-L': [-4, -9, 52],
    'PMd-L': [-27, -14, 61],
    'FEF-L': [-28, -1, 43],
    'PMv-L': [-50, 5, 21],
    
    # Right hemisphere
    'V1-R': [7, -89, 1],
    'V3-R': [20, -87, 15],
    'VIP-R': [37, -44, 47],
    'SPOC-R': [10, -77, 34],
    'AG-R': [32, -70, 35],
    'POJ-R': [16, -79, 43],
    'SPL-R': [27, -55, 49],
    'mIPS-R': [23, -62, 40],
    'IPL-R': [41, -41, 39],
    'STS-R': [49, -41, 12],
    'S1-R': [39, -26, 40],
    'M1-R': [37, -23, 52],
    'SMA-R': [3, -7, 49],
    'PMd-R': [21, -14, 61],
    'FEF-R': [31, -2, 45],
    'PMv-R': [48, 8, 21]
}

# Define the standard views - reduced to just 3 key views
views = [
    {'title': 'Top View', 'view_angle': (90, -90)},
    {'title': 'Side View (Left)', 'view_angle': (0, 180)},
    {'title': 'Side View (Right)', 'view_angle': (0, 0)}
]

def get_region_category(region):
    """Get the category of a brain region based on its name"""
    for category, regions in region_categories.items():
        if region in regions:
            return category
    return 'Other'

def get_region_color(region):
    """Get the color for a brain region based on its category"""
    category = get_region_category(region)
    return category_colors.get(category, 'gray')

def read_csv_file(file_path):
    """Read the CSV file and create a matrix"""
    logging.info(f"Reading file: {file_path}")
    
    # Read the raw content of the file as text
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Create a matrix to store the connections
    matrix = np.zeros((32, 32))
    connection_count = 0
    
    # Parse the CSV file line by line
    for i, line in enumerate(lines):
        if i >= 32:  # Ensure we only read 32 rows
            break
            
        values = line.strip().split(',')
        for j, val in enumerate(values):
            if j >= 32:  # Ensure we only read 32 columns
                break
                
            # If the value is exactly "0.0", it's no connection
            # Otherwise, it's a connection with the given strength
            if val != "0.0" and val != "0":
                # CRITICAL FIX: The CSV file has source nodes in columns and target nodes in rows
                # So matrix[i, j] represents a connection FROM column j TO row i
                matrix[i, j] = float(val)
                connection_count += 1
    
    logging.info(f"Matrix shape: {matrix.shape}")
    logging.info(f"Found {connection_count} connections out of {32*32} possible connections")
    
    return matrix

def create_3d_visualization(matrix, output_path, title, connection_type="all"):
    """Create 3D brain visualization with connections - matching v1 implementation"""
    # Fetch the fsaverage surface
    fsaverage = datasets.fetch_surf_fsaverage()
    
    # Create figure with black background - adjusted for 3 views instead of 6
    fig = plt.figure(figsize=(36, 12), facecolor='black')  # Height reduced from 30 to 12
    
    # Calculate maximum weight for scaling
    if connection_type == "positive":
        # Only consider positive values
        max_weight = np.max(np.maximum(matrix, 0))
    elif connection_type == "negative":
        # Only consider absolute values of negative connections
        max_weight = np.max(np.abs(np.minimum(matrix, 0)))
    else:  # "all"
        # Consider all values (absolute for negatives)
        max_weight = np.max(np.abs(matrix))
    
    # Process each view
    for i, view_info in enumerate(views):
        # Create subplot - now using 1x3 grid instead of 2x3
        ax = plt.subplot(1, 3, i+1, projection='3d', facecolor='black')
        
        # Load both hemispheres
        mesh_left = surface.load_surf_mesh(fsaverage.pial_left)
        mesh_right = surface.load_surf_mesh(fsaverage.pial_right)
        
        # Plot hemispheres
        x, y, z = mesh_left[0].T
        ax.plot_trisurf(x, y, z, triangles=mesh_left[1], 
                      alpha=0.2, color='gray', linewidth=0, edgecolor='none')
        
        x, y, z = mesh_right[0].T
        ax.plot_trisurf(x, y, z, triangles=mesh_right[1], 
                      alpha=0.2, color='gray', linewidth=0, edgecolor='none')
        
        # Draw connections
        for source_idx, source in enumerate(region_labels):
            for target_idx, target in enumerate(region_labels):
                if source_idx != target_idx:
                    weight = matrix[target_idx, source_idx]  # Source is column, target is row
                    
                    # Skip based on connection type
                    if connection_type == "positive" and weight <= 0:
                        continue
                    if connection_type == "negative" and weight >= 0:
                        continue
                    if weight == 0:
                        continue
                    
                    # Get coordinates
                    source_coords = talairach_coordinates[source]
                    target_coords = talairach_coordinates[target]
                    
                    # Set color based on connection weight
                    if weight > 0:
                        color = plt.cm.Reds(min(abs(weight) / max_weight, 1.0))
                    else:
                        color = plt.cm.Blues(min(abs(weight) / max_weight, 1.0))
                    
                    # Draw connection line
                    ax.plot([source_coords[0], target_coords[0]],
                           [source_coords[1], target_coords[1]],
                           [source_coords[2], target_coords[2]],
                            color=color,
                            alpha=0.6,
                            linewidth=1 + 2 * abs(weight) / max_weight,
                            zorder=1)
                    
                    # Add arrow at target, but position it slightly before the target node
                    arrow_ratio = 0.7  # Position arrow further from target (was 0.8)
                    arrow_pos = [
                        source_coords[0] + arrow_ratio * (target_coords[0] - source_coords[0]),
                        source_coords[1] + arrow_ratio * (target_coords[1] - source_coords[1]),
                        source_coords[2] + arrow_ratio * (target_coords[2] - source_coords[2])
                    ]
                    
                    # Calculate the direction vector from source to target
                    direction = [
                        target_coords[0] - arrow_pos[0],
                        target_coords[1] - arrow_pos[1],
                        target_coords[2] - arrow_pos[2]
                    ]
                    
                    # Draw arrow with improved visibility
                    ax.quiver(arrow_pos[0], arrow_pos[1], arrow_pos[2],
                            direction[0], direction[1], direction[2],
                            color=color,
                            alpha=1.0,  # Full opacity
                            length=0.8,  # Even larger arrow length
                            linewidth=3.0,  # Thicker lines
                            arrow_length_ratio=1.0,  # Maximum arrowhead size
                            normalize=True)
        
        # Plot regions as spheres
        for region in region_labels:
            coords = talairach_coordinates[region]
            color = get_region_color(region)
            
            ax.scatter(coords[0], coords[1], coords[2],
                      color=color,
                      s=50,
                      alpha=0.9,
                      edgecolors='white',
                      linewidth=0.5,
                      zorder=2)
        
        # Set view angle
        elev, azim = view_info['view_angle']
        ax.view_init(elev=elev, azim=azim)
        
        # Adjust view distance and limits
        ax.dist = 4
        ax.set_xlim(-70, 70)
        ax.set_ylim(-70, 70)
        ax.set_zlim(-70, 70)
        
        # Set title for this view - moved down to avoid overlap
        ax.set_title(view_info['title'], color='white', pad=40, fontsize=16, y=-0.1)
        
        # Remove axes
        ax.set_axis_off()
    
    # Add main title
    plt.suptitle(title, color='white', fontsize=22, y=0.98)
    
    # Adjust layout
    plt.subplots_adjust(
        left=0.01, right=0.85,
        top=0.95, bottom=0.02,
        wspace=0.0, hspace=0.0
    )
    
    # Add colorbar
    if connection_type == "positive":
        cmap = plt.cm.Reds
        vmin = 0
        vmax = max_weight
    elif connection_type == "negative":
        cmap = plt.cm.Blues
        vmin = 0
        vmax = max_weight
    else:
        # For "all" connections, use a diverging colormap
        cmap = plt.cm.RdBu_r
        vmin = -max_weight
        vmax = max_weight
    
    # Create a ScalarMappable for the colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.88, 0.25, 0.02, 0.5])  # [left, bottom, width, height]
    cbar = plt.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Positive Connectivity' if connection_type == "positive" else 
                  'Negative Connectivity' if connection_type == "negative" else 
                  'Connectivity', color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    
    # Add max value at the top of the colorbar
    plt.text(0.90, 0.76, f'Max: {max_weight:.3f}', 
             color='white', 
             fontsize=10, 
             ha='left',
             va='top',
             transform=fig.transFigure)  # Use figure coordinates
    
    # Add legend for region categories
    legend_elements = []
    for region_type, color in category_colors.items():
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                         markerfacecolor=color, markersize=10, 
                                         label=region_type, linewidth=0))
    
    # Position the legend below the title - moved down to avoid overlap
    fig.legend(handles=legend_elements, loc='upper center', 
             bbox_to_anchor=(0.5, 0.92), ncol=5, frameon=False,  # Changed from 0.95 to 0.92
             fontsize=10, labelcolor='white')
    
    return fig

def run():
    """Run the visualization with file selection"""
    # Select file
    file_path = select_file()
    logging.info(f"Selected file: {file_path}")
    
    # Select connection display mode
    connection_display = select_connection_display()
    logging.info(f"Connection display mode: {connection_display}")
    
    # Read the file
    matrix = read_csv_file(file_path)
    
    # Generate timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Extract method and frequency from filename
    filename = file_path.stem
    
    # Check if this is a NO-LT file
    is_no_lt_file = "_NO-LT" in filename
    
    # Extract method and frequency (handle NO-LT suffix)
    if is_no_lt_file:
        # Remove _NO-LT suffix for parsing
        filename_parts = filename.replace("_NO-LT", "").split('_')
    else:
        filename_parts = filename.split('_')
    
    # Method_freq is the first two parts (e.g., "gDTF_10Hz")
    method_freq = filename_parts[0] + "_" + filename_parts[1]
    # Analysis type is the last part (e.g., "wilcoxon")
    analysis_type = filename_parts[-1]
    
    # Add NO-LT indicator to analysis type if applicable
    if is_no_lt_file:
        analysis_type += "_NO-LT"
    
    logging.info(f"Detected method_freq: {method_freq}")
    logging.info(f"Detected analysis_type: {analysis_type}")
    logging.info(f"File type: {'NO-LT' if is_no_lt_file else 'Regular'}")
    
    # Output folder is the Visual folder
    output_folder = visual_folder
    
    if connection_display == "all":
        # Create single visualization with all connections
        title = f"{method_freq} {analysis_type}\nConnectivity"
        output_path = output_folder / f"{method_freq}_{analysis_type}_3D_brain_all_{timestamp}.png"
        svg_path = output_path.with_suffix('.svg')
        
        fig = create_3d_visualization(matrix, output_path, title, connection_type="all")
        
        # Save the PNG version with black background
        fig.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='black')
        
        # Save the SVG version
        fig.savefig(svg_path, format='svg', bbox_inches='tight', facecolor='black')
        
        plt.close(fig)
        logging.info(f"Created visualization: {output_path}")
        
    else:  # "separate"
        # Create positive connections visualization
        pos_title = f"{method_freq} {analysis_type}\nPositive Connectivity"
        pos_output_path = output_folder / f"{method_freq}_{analysis_type}_3D_brain_positive_{timestamp}.png"
        pos_svg_path = pos_output_path.with_suffix('.svg')
        
        pos_fig = create_3d_visualization(matrix, pos_output_path, pos_title, connection_type="positive")
        
        # Save the PNG version with black background
        pos_fig.savefig(pos_output_path, dpi=600, bbox_inches='tight', facecolor='black')
        
        # Save the SVG version
        pos_fig.savefig(pos_svg_path, format='svg', bbox_inches='tight', facecolor='black')
        
        plt.close(pos_fig)
        logging.info(f"Created positive connections 3D brain visualization: {pos_output_path}")
        
        # Create negative connections visualization
        neg_title = f"{method_freq} {analysis_type}\nNegative Connectivity"
        neg_output_path = output_folder / f"{method_freq}_{analysis_type}_3D_brain_negative_{timestamp}.png"
        neg_svg_path = neg_output_path.with_suffix('.svg')
        
        neg_fig = create_3d_visualization(matrix, neg_output_path, neg_title, connection_type="negative")
        
        # Save the PNG version with black background
        neg_fig.savefig(neg_output_path, dpi=600, bbox_inches='tight', facecolor='black')
        
        # Save the SVG version
        neg_fig.savefig(neg_svg_path, format='svg', bbox_inches='tight', facecolor='black')
        
        plt.close(neg_fig)
        logging.info(f"Created negative connections 3D brain visualization: {neg_output_path}")
    
    logging.info("Visualization completed successfully!")

if __name__ == "__main__":
    run()