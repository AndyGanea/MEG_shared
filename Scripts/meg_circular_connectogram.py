#!/usr/bin/env python3

"""
MEG Circular Connectogram Visualization
------------------
This script creates professional Circos-style connectogram visualizations from MEG connectivity matrices.
It reads the same CSV files used for heatmaps and generates circular diagrams showing:
1. Brain regions arranged in a circle with improved label placement
2. Connectivity between regions shown as curved lines with arrows indicating direction
3. Line colors and thickness representing connection strength with transparency
4. Positive connections in red, negative in blue
5. Outer rings showing additional metrics for each region
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
from matplotlib.path import Path
from pathlib import Path as FilePath
import logging
from datetime import datetime
import sys
import os
import math
import networkx as nx
import matplotlib.colors as mcolors

class MEGConnectogramVisualizer:
    def __init__(self):
        """Initialize the connectogram visualizer"""
        self.data_dir = FilePath("Data")
        self.logs_dir = FilePath("Logs")
        
        # Get user selection for dataset folder
        self.dataset_dir = self.select_dataset_folder()
        
        # Get user selection for analysis type
        self.analysis_type = self.select_analysis_type()
        
        # Get user selection for specific analysis folder
        self.analysis_dir = self.select_analysis_folder()
        
        # Initialize counters
        self.folders_processed = 0
        self.visualizations_created = 0
        self.errors = 0
        
        # Define region labels
        self.region_labels = [
            'V1-L', 'V3-L', 'SPOC-L', 'AG-L', 'POJ-L', 'SPL-L', 'mIPS-L', 'VIP-L', 
            'IPL-L', 'STS-L', 'S1-L', 'M1-L', 'SMA-L', 'PMd-L', 'FEF-L', 'PMv-L',
            'V1-R', 'V3-R', 'SPOC-R', 'AG-R', 'POJ-R', 'SPL-R', 'mIPS-R', 'VIP-R', 
            'IPL-R', 'STS-R', 'S1-R', 'M1-R', 'SMA-R', 'PMd-R', 'FEF-R', 'PMv-R'
        ]
        
        # Define region categories for coloring the outer rings
        self.region_categories = {
            'Visual': ['V1-L', 'V3-L', 'V1-R', 'V3-R'],
            'Parietal': ['SPOC-L', 'AG-L', 'POJ-L', 'SPL-L', 'mIPS-L', 'VIP-L', 'IPL-L',
                         'SPOC-R', 'AG-R', 'POJ-R', 'SPL-R', 'mIPS-R', 'VIP-R', 'IPL-R'],
            'Temporal': ['STS-L', 'STS-R'],
            'Sensorimotor': ['S1-L', 'M1-L', 'S1-R', 'M1-R'],
            'Frontal': ['SMA-L', 'PMd-L', 'FEF-L', 'PMv-L', 'SMA-R', 'PMd-R', 'FEF-R', 'PMv-R']
        }
        
        # Define colors for region categories
        self.category_colors = {
            'Visual': '#1f77b4',      # Blue
            'Parietal': '#ff7f0e',    # Orange
            'Temporal': '#2ca02c',    # Green
            'Sensorimotor': '#d62728', # Red
            'Frontal': '#9467bd'      # Purple
        }
        
        # Setup logging
        self.setup_logging()

    def select_dataset_folder(self) -> FilePath:
        """Present available folders and let user select one"""
        print("\nAvailable datasets:")
        print("================")
        
        available_dirs = [d for d in self.data_dir.iterdir() 
                         if d.is_dir() and d.name.startswith("DataSet")]
        
        for idx, dir_path in enumerate(available_dirs, 1):
            print(f"{idx}. {dir_path.name}")
        
        while True:
            try:
                choice = int(input("\nSelect dataset number: "))
                if 1 <= choice <= len(available_dirs):
                    selected_dir = available_dirs[choice - 1]
                    print(f"\nSelected: {selected_dir.name}")
                    return selected_dir
                else:
                    print("Invalid selection. Please try again.")
            except ValueError:
                print("Please enter a valid number.")

    def select_analysis_type(self) -> str:
        """Ask user to select analysis type (T_Test or Wilcoxon)"""
        print("\nSelect analysis type:")
        print("1. T_Test")
        print("2. Wilcoxon")
        
        while True:
            try:
                choice = int(input("\nEnter your choice (1 or 2): "))
                if choice == 1:
                    return "T_Test"
                elif choice == 2:
                    return "Wilcoxon"
                else:
                    print("Invalid selection. Please enter 1 or 2.")
            except ValueError:
                print("Please enter a valid number.")

    def select_analysis_folder(self) -> FilePath:
        """Present available analysis folders and let user select one"""
        print(f"\nAvailable {self.analysis_type} folders:")
        print("=========================")
        
        available_dirs = [d for d in self.dataset_dir.iterdir() 
                         if d.is_dir() and d.name.startswith(self.analysis_type)]
        
        if not available_dirs:
            print(f"No {self.analysis_type} folders found in {self.dataset_dir.name}")
            sys.exit(1)
        
        for idx, dir_path in enumerate(available_dirs, 1):
            print(f"{idx}. {dir_path.name}")
        
        while True:
            try:
                choice = int(input("\nSelect folder number: "))
                if 1 <= choice <= len(available_dirs):
                    selected_dir = available_dirs[choice - 1]
                    print(f"\nSelected: {selected_dir.name}")
                    return selected_dir
                else:
                    print("Invalid selection. Please try again.")
            except ValueError:
                print("Please enter a valid number.")

    def setup_logging(self):
        """Setup logging to both file and console"""
        timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        log_file = self.logs_dir / f"meg_connectogram_{timestamp}.txt"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        logging.info("MEG Circular Connectogram Visualization")
        logging.info("=====================================")
        logging.info(f"Selected dataset: {self.dataset_dir.name}")
        logging.info(f"Analysis type: {self.analysis_type}")
        logging.info(f"Selected analysis folder: {self.analysis_dir.name}")
        logging.info(f"Process started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info("---------------------\n")

    def get_region_category(self, region):
        """Get the category of a brain region"""
        for category, regions in self.region_categories.items():
            if region in regions:
                return category
        return "Other"

    def get_region_color(self, region):
        """Get the color for a brain region based on its category"""
        category = self.get_region_category(region)
        return self.category_colors.get(category, '#7f7f7f')  # Default to gray

    def calculate_node_metrics(self, matrix):
        """Calculate metrics for each node/region based on the connectivity matrix"""
        n_regions = matrix.shape[0]  # Get actual size from matrix
        
        # Calculate total connectivity (sum of absolute values)
        total_connectivity = np.zeros(n_regions)
        for i in range(n_regions):
            # Sum of absolute values in row i (incoming) and column i (outgoing)
            total_connectivity[i] = np.sum(np.abs(matrix[i, :])) + np.sum(np.abs(matrix[:, i]))
        
        # Calculate outgoing connectivity (sum of column i)
        outgoing_connectivity = np.zeros(n_regions)
        for i in range(n_regions):
            outgoing_connectivity[i] = np.sum(np.abs(matrix[:, i]))
        
        # Calculate incoming connectivity (sum of row i)
        incoming_connectivity = np.zeros(n_regions)
        for i in range(n_regions):
            incoming_connectivity[i] = np.sum(np.abs(matrix[i, :]))
        
        # Calculate net connectivity (outgoing - incoming)
        net_connectivity = outgoing_connectivity - incoming_connectivity
        
        # Create a new dictionary with the results
        return {
            'total': total_connectivity,
            'outgoing': outgoing_connectivity,
            'incoming': incoming_connectivity,
            'net': net_connectivity
        }

    def create_connectogram(self, matrix, output_path, title):
        """
        Create a circular connectogram visualization of brain connectivity.
        Returns the figure for saving externally.
        """
        # Verify matrix is 32x32
        assert matrix.shape == (32, 32), f"Matrix must be 32x32, got {matrix.shape}"
        
        # CRITICAL FIX: Use a small threshold to filter out numerical noise
        # This is much smaller than any meaningful connection but will catch floating point issues
        epsilon = 1e-6
        
        # Apply the threshold
        filtered_matrix = np.where(np.abs(matrix) < epsilon, 0.0, matrix)
        
        # Count zeros vs non-zeros after filtering
        zeros = (filtered_matrix == 0.0).sum()
        non_zeros = filtered_matrix.size - zeros
        print(f"After filtering: Matrix contains {zeros} zeros and {non_zeros} non-zeros")
        
        # If we still have too many non-zeros, use a percentile-based threshold
        if non_zeros > 100:  # If we have more than 100 connections, use a percentile threshold
            print("Too many non-zero connections, using percentile threshold")
            flat_matrix = np.abs(filtered_matrix.flatten())
            flat_matrix = flat_matrix[flat_matrix > 0]  # Only consider non-zero values
            
            # Use 75th percentile (top 25% of non-zero connections)
            threshold = np.percentile(flat_matrix, 75)
            print(f"Using significance threshold: {threshold}")
            
            # Apply the threshold
            filtered_matrix = np.where(np.abs(filtered_matrix) < threshold, 0.0, filtered_matrix)
            
            # Count zeros vs non-zeros after thresholding
            zeros = (filtered_matrix == 0.0).sum()
            non_zeros = filtered_matrix.size - zeros
            print(f"After thresholding: Matrix contains {zeros} zeros and {non_zeros} non-zeros")
        
        n_regions = 32  # We know we have 32 regions
        
        # Calculate node metrics
        node_metrics = self.calculate_node_metrics(filtered_matrix)
        
        # Create figure with black background
        fig = plt.figure(figsize=(14, 14), facecolor='black')
        ax = fig.add_subplot(111)
        ax.set_facecolor('black')
        
        # Set the axis limits explicitly to ensure the circle is centered
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        
        # Create a graph
        G = nx.DiGraph()  # Changed to DiGraph to represent directed connections
        
        # Reorganize region labels by hemisphere
        left_regions = [label for label in self.region_labels if label.endswith('-L')]
        right_regions = [label for label in self.region_labels if label.endswith('-R')]
        
        # Sort regions by their base name (without -L or -R) to ensure matching regions are aligned
        left_regions.sort(key=lambda x: x.split('-')[0])
        right_regions.sort(key=lambda x: x.split('-')[0])
        
        # Add nodes with explicit positioning
        # Apply a small rotation to ensure proper hemisphere separation
        rotation_angle = np.pi/32  # Small rotation angle (about 5.6 degrees)
        
        # Add nodes with explicit positioning
        # Left hemisphere on the RIGHT side (from π/2 to 3π/2) - SWAPPED
        # Right hemisphere on the LEFT side (from -π/2 to π/2) - SWAPPED
        pos = {}
        region_to_node = {}  # Direct mapping from region name to node index
        
        node_index = 0
        
        # Add right hemisphere nodes (on the left side) - SWAPPED
        for i, region in enumerate(right_regions):
            angle = np.pi * (i / len(right_regions) - 0.5) + rotation_angle  # Add rotation
            x = np.cos(angle)
            y = np.sin(angle)
            pos[node_index] = (x, y)
            region_to_node[region] = node_index
            G.add_node(node_index, label=region, category=self.get_region_category(region))
            node_index += 1
        
        # Add left hemisphere nodes (on the right side) - SWAPPED
        for i, region in enumerate(left_regions):
            angle = np.pi * (i / len(left_regions) + 0.5) + rotation_angle  # Add rotation
            x = np.cos(angle)
            y = np.sin(angle)
            pos[node_index] = (x, y)
            region_to_node[region] = node_index
            G.add_node(node_index, label=region, category=self.get_region_category(region))
            node_index += 1
        
        # Create a mapping from region name to matrix index
        region_to_matrix_idx = {region: idx for idx, region in enumerate(self.region_labels)}
        
        # Find all non-zero connections
        edges_to_draw = []
        non_zero_connections = []
        
        # In the matrix, [row, col] means FROM col TO row
        for target_region in self.region_labels:
            for source_region in self.region_labels:
                if target_region != source_region:  # Skip self-connections
                    target_idx = region_to_matrix_idx[target_region]
                    source_idx = region_to_matrix_idx[source_region]
                    
                    # The connection is FROM source TO target
                    # So we look at matrix[target_idx, source_idx]
                    strength = filtered_matrix[target_idx, source_idx]
                    
                    # Use exact comparison with 0.0
                    if strength != 0.0:
                        non_zero_connections.append((source_region, target_region, strength))
        
        # Verify we have a reasonable number of connections
        print(f"Found {len(non_zero_connections)} non-zero connections")
        
        # Check if the S1-R to VIP-L connection is non-zero
        s1r_to_vipl = next((conn for conn in non_zero_connections if conn[0] == "S1-R" and conn[1] == "VIP-L"), None)
        if s1r_to_vipl:
            print(f"Connection FROM S1-R TO VIP-L has strength {s1r_to_vipl[2]}")
        else:
            # Find the actual strength of this connection
            vipl_idx = region_to_matrix_idx["VIP-L"]
            s1r_idx = region_to_matrix_idx["S1-R"]
            strength = matrix[vipl_idx, s1r_idx]
            print(f"Connection FROM S1-R TO VIP-L has strength {strength} but is not significant")
        
        # Now create edges for the graph
        for source_region, target_region, strength in non_zero_connections:
            source_node = region_to_node[source_region]
            target_node = region_to_node[target_region]
            
            G.add_edge(source_node, target_node, weight=strength)
            edges_to_draw.append((source_node, target_node, strength))
        
        # Sort edges by absolute weight to draw stronger connections on top
        edges_to_draw.sort(key=lambda x: abs(x[2]))
        
        # Find min and max weights for normalization (separately for positive and negative)
        pos_weights = [w for _, _, w in edges_to_draw if w > 0]
        neg_weights = [w for _, _, w in edges_to_draw if w < 0]
        
        max_pos_weight = max(pos_weights) if pos_weights else 0.001
        min_neg_weight = min(neg_weights) if neg_weights else -0.001
        
        # Create colormaps for positive and negative values
        pos_cmap = plt.cm.Reds  # Red colormap for positive values
        neg_cmap = plt.cm.Blues  # Blue colormap for negative values (NOT reversed)
        
        # Draw edges with increased curvature and arrows
        for source, target, weight in edges_to_draw:
            source_x, source_y = pos[source]
            target_x, target_y = pos[target]
            
            # Calculate the angle between the two nodes
            angle = np.arctan2(target_y - source_y, target_x - source_x)
            
            # Calculate the distance between the two nodes
            distance = np.sqrt((target_x - source_x)**2 + (target_y - source_y)**2)
            
            # Calculate the arrow length
            arrow_length = 0.05 * distance
            
            # Calculate the arrow position
            arrow_x = target_x - arrow_length * np.cos(angle)
            arrow_y = target_y - arrow_length * np.sin(angle)
            
            # Draw the arrow
            arrow = patches.FancyArrowPatch(
                (source_x, source_y), (arrow_x, arrow_y),
                arrowstyle='-|>',
                mutation_scale=20,
                color='white',
                linewidth=2,
                alpha=0.9
            )
            ax.add_patch(arrow)
            
            # Draw the line between the two nodes
            ax.plot([source_x, target_x], [source_y, target_y], color='white', linewidth=2)
        
        # Draw nodes as colored dots
        for i, node in enumerate(G.nodes()):
            x, y = pos[node]
            category = G.nodes[node]['category']
            color = self.category_colors[category]
            
            # Draw a colored dot
            circle = plt.Circle((x, y), 0.025, facecolor=color, edgecolor='white', alpha=0.9)
            ax.add_patch(circle)
        
        # Add node labels with better positioning
        for i, node in enumerate(G.nodes()):
            x, y = pos[node]
            
            # Calculate angle for this node
            angle = np.arctan2(y, x)
            
            # Position labels slightly outside the circle
            label_radius = 1.15
            label_x = label_radius * np.cos(angle)
            label_y = label_radius * np.sin(angle)
            
            # Adjust text alignment based on position
            ha = 'left' if x < 0 else 'right'
            va = 'center'
            
            # Rotate labels for better readability
            rotation = angle * 180 / np.pi
            if x < 0:
                rotation += 180
            
            # Add the label with slightly larger font
            ax.text(label_x, label_y, G.nodes[node]['label'], 
                   color='white', 
                   fontsize=9,
                   ha=ha, va=va, 
                   rotation=rotation, 
                   rotation_mode='anchor',
                   fontweight='bold')
        
        # Add title
        plt.title(title, color='white', size=14, y=1.05)
        
        # Set equal aspect ratio and remove axes
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Create a custom colorbar with both positive and negative values
        # Create a custom colormap that goes from dark blue to light blue to light red to dark red
        colors_neg = plt.cm.Blues(np.linspace(0.2, 1, 128))  # Light blue to dark blue
        colors_pos = plt.cm.Reds(np.linspace(0.2, 1, 128))  # Light red to dark red
        
        # Combine the colormaps
        colors = np.vstack((colors_neg[::-1], colors_pos))  # Reverse the negative colors
        custom_cmap = mcolors.LinearSegmentedColormap.from_list('BlueRed', colors)
        
        # Create a custom norm that maps the data range to the colormap range
        min_val = -0.001
        max_val = 0.001
        
        # Add colorbar
        cax = fig.add_axes([0.80, 0.05, 0.015, 0.20])  # [left, bottom, width, height]
        norm = mcolors.TwoSlopeNorm(vmin=min_val, vcenter=0, vmax=max_val)
        sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm)
        sm.set_array([])
        cb = plt.colorbar(sm, cax=cax)
        cb.set_label('Sensory Coding Index', color='white', fontsize=8, labelpad=5)
        cb.ax.yaxis.set_tick_params(color='white', labelsize=6)
        plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color='white')
        
        # Add a legend for the arrows
        legend_ax = fig.add_axes([0.15, 0.05, 0.20, 0.05])  # [left, bottom, width, height]
        legend_ax.axis('off')
        legend_ax.set_facecolor('black')
        
        # Add arrow explanation
        arrow = patches.FancyArrowPatch(
            (0.1, 0.5), (0.4, 0.5),
            arrowstyle='-|>',
            mutation_scale=20,
            color='white',
            linewidth=2,
            alpha=0.9
        )
        legend_ax.add_patch(arrow)
        legend_ax.text(0.5, 0.5, 'Direction of connection', 
                      color='white', fontsize=8, ha='left', va='center')
        
        return fig

    def process_folder(self, folder_path):
        """Process a single method/frequency folder"""
        try:
            # Get the method and frequency from the folder name
            method_freq = folder_path.name.split('_', 1)[1]  # Skip the "Cue_" prefix
            
            # Find the CSV file
            csv_files = list(folder_path.glob("*.csv"))
            
            if not csv_files:
                logging.warning(f"No CSV files found in {folder_path}")
                return
            
            # Use the first CSV file found
            csv_file = csv_files[0]
            logging.info(f"Processing {csv_file}")
            
            # Read the CSV file
            matrix = pd.read_csv(csv_file, header=None).values
            
            # Print matrix shape for debugging
            print(f"Matrix shape: {matrix.shape}")
            
            # Ensure matrix is 32x32
            if matrix.shape != (32, 32):
                logging.error(f"Matrix is not 32x32: {matrix.shape}. Skipping.")
                return
            
            # Generate timestamp for the filename
            timestamp = datetime.now().strftime("%Y%m%d-%H%M")
            
            # Create the output path in the same folder as the input file with timestamp
            output_path = folder_path / f"{method_freq}_L-R_{self.analysis_type.lower()}_circular_connectogram_{timestamp}.png"
            title = f"{method_freq} {self.analysis_type.replace('_', ' ')} Connectivity"
            
            # Create the visualization
            fig = self.create_connectogram(matrix, output_path, title)
            
            # Save the PNG version with black background
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='black')
            
            # Create SVG version with transparent background
            svg_path = output_path.with_suffix('.svg')
            plt.savefig(svg_path, format='svg', dpi=300, bbox_inches='tight', transparent=True)
            
            plt.close(fig)
            
            self.visualizations_created += 1
            logging.info(f"Created connectogram for {method_freq} in {folder_path.name}")
            logging.info(f"Saved PNG: {output_path}")
            logging.info(f"Saved SVG: {svg_path}")
            
        except Exception as e:
            self.errors += 1
            logging.error(f"Error processing {folder_path}: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())

    def process_all_folders(self):
        """Process all method/frequency folders in the selected analysis folder"""
        logging.info(f"Processing all folders in {self.analysis_dir.name}...")
        
        # Get all method folders
        method_folders = [d for d in self.analysis_dir.iterdir() if d.is_dir()]
        
        if not method_folders:
            logging.warning(f"No method folders found in {self.analysis_dir.name}")
            return
        
        for folder in method_folders:
            logging.info(f"Processing {folder.name}...")
            self.process_folder(folder)
            self.folders_processed += 1
        
        logging.info("\n---------------------")
        logging.info("Processing complete!")
        logging.info(f"Folders processed: {self.folders_processed}")
        logging.info(f"Visualizations created: {self.visualizations_created}")
        logging.info(f"Errors encountered: {self.errors}")
        logging.info(f"Process completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def main():
    """Main function to run the MEG connectogram visualizer"""
    visualizer = MEGConnectogramVisualizer()
    visualizer.process_all_folders()

if __name__ == "__main__":
    main() 