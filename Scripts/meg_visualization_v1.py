#!/usr/bin/env python3

"""
MEG Visualization Tool V1
------------------------
Combines circular connectogram, 3D brain surface, and glass brain visualizations
for MEG connectivity analysis.

Features:
1. Circular connectogram with directional connections
2. 3D brain surface visualization with Talairach coordinates
3. Glass brain visualization
4. Support for positive/negative connection separation
5. Automated file handling and visualization generation

Requirements:
    - nilearn
    - matplotlib
    - numpy
    - scipy
    - networkx
    - datetime

MEGVisualizer (Coordinator)
├── MEGDataHandler
│   ├── Data selection
│   ├── File reading
│   └── Region management
├── CircularConnectogramVisualizer
│   └── Circular visualization
├── BrainSurfaceVisualizer
│   └── 3D brain visualization
└── GlassBrainVisualizer
    └── Glass brain visualization    
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch, Circle
from matplotlib.path import Path
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
from pathlib import Path as FilePath
from datetime import datetime
import logging
import sys
import os
from scipy.special import comb
from collections import defaultdict
from matplotlib.lines import Line2D
import matplotlib.patheffects as path_effects

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

class MEGDataHandler:
    """Handles data loading, CSV interpretation, and user interactions"""
    
    def __init__(self):
        """
        Initialize the MEG data handler
        """
        self.data_dir = FilePath("Data")
        self.logs_dir = FilePath("Logs")
        
        # Initialize paths
        self.dataset_dir = None
        self.analysis_type = None
        self.analysis_dir = None
        self.connection_display = None
        self.visualization_type = None
        
        # Initialize region information - match v3's order exactly
        self.region_labels = [
            'V1-L', 'V3-L', 'SPOC-L', 'AG-L', 'POJ-L', 'SPL-L', 'mIPS-L', 'VIP-L', 'IPL-L', 'STS-L',
            'S1-L', 'M1-L', 'SMA-L', 'PMd-L', 'FEF-L', 'PMv-L',
            'V1-R', 'V3-R', 'SPOC-R', 'AG-R', 'POJ-R', 'SPL-R', 'mIPS-R', 'VIP-R', 'IPL-R', 'STS-R',
            'S1-R', 'M1-R', 'SMA-R', 'PMd-R', 'FEF-R', 'PMv-R'
        ]
        
        # Add Talairach coordinates here
        self.talairach_coordinates = {
            # Left hemisphere
            'V1-L': [-8, -91, 0],
            'V3-L': [-21, -85, 16],
            'VIP-L': [-37, -40, 44],  # Moved to Visual category but kept coordinates
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
            'VIP-R': [37, -44, 47],  # Moved to Visual category but kept coordinates
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
        
        # Define region categories
        self.region_categories = {
            'Visual': ['V1-L', 'V3-L', 'VIP-L', 'V1-R', 'V3-R', 'VIP-R'],  # Added VIP-L and VIP-R to Visual
            'Parietal': ['SPOC-L', 'AG-L', 'POJ-L', 'SPL-L', 'mIPS-L', 'IPL-L',
                         'SPOC-R', 'AG-R', 'POJ-R', 'SPL-R', 'mIPS-R', 'IPL-R'],  # Removed VIP-L and VIP-R
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
        
        # Get user selections
        self.initialize_user_selections()

    def setup_logging(self):
        """Setup logging to file and console"""
        self.logs_dir.mkdir(exist_ok=True)
        timestamp = self.generate_timestamp()
        log_file = self.logs_dir / f"meg_visualization_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        logging.info("Starting MEG Visualization V1")

    @staticmethod
    def generate_timestamp():
        """Generate a timestamp string for file naming"""
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def add_timestamp_to_filename(self, output_path, timestamp=None):
        """Add timestamp to filename while avoiding double timestamps"""
        if timestamp is None:
            timestamp = self.generate_timestamp()
        
        # Check if filename already has a timestamp pattern (YYYYMMDD_HHMMSS)
        if '_20' in output_path and any(char.isdigit() for char in output_path.split('_20')[1][:6]):
            return output_path
        
        name, ext = os.path.splitext(output_path)
        return f"{name}_{timestamp}{ext}"

    def initialize_user_selections(self):
        """Get all necessary user inputs"""
        self.dataset_dir = self.select_dataset_folder()
        self.analysis_type = self.select_analysis_type()
        self.analysis_dir = self.select_analysis_folder()
        self.visualization_type = self.select_visualization_type()
        self.connection_display = self.select_connection_display()
        
        logging.info(f"Selected dataset: {self.dataset_dir.name}")
        logging.info(f"Selected analysis type: {self.analysis_type}")
        logging.info(f"Selected analysis folder: {self.analysis_dir.name}")
        logging.info(f"Selected visualization type: {self.visualization_type}")
        logging.info(f"Connection display mode: {self.connection_display}")

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
                    return available_dirs[choice-1]
                else:
                    print(f"Please enter a number between 1 and {len(available_dirs)}")
            except ValueError:
                print("Please enter a valid number")

    def select_analysis_type(self) -> str:
        """Let user select the analysis type"""
        print("\nSelect analysis type:")
        print("==================")
        
        analysis_types = ["t-test", "Wilcoxon"]
        
        for idx, analysis in enumerate(analysis_types, 1):
            print(f"{idx}. {analysis}")
        
        while True:
            try:
                choice = int(input("\nSelect analysis type number: "))
                if 1 <= choice <= len(analysis_types):
                    return analysis_types[choice-1]
                else:
                    print(f"Please enter a number between 1 and {len(analysis_types)}")
            except ValueError:
                print("Please enter a valid number")

    def select_analysis_folder(self) -> FilePath:
        """Let user select a specific analysis folder"""
        print("\nAvailable analysis folders:")
        print("========================")
        
        # Find folders matching the analysis type
        available_dirs = [d for d in self.dataset_dir.glob(f"*{self.analysis_type}*") 
                         if d.is_dir()]
        
        if not available_dirs:
            logging.error(f"No folders found for {self.analysis_type} analysis.")
            sys.exit(1)
        
        for idx, dir_path in enumerate(available_dirs, 1):
            print(f"{idx}. {dir_path.name}")
        
        while True:
            try:
                choice = int(input("\nSelect analysis folder number: "))
                if 1 <= choice <= len(available_dirs):
                    return available_dirs[choice-1]
                else:
                    print(f"Please enter a number between 1 and {len(available_dirs)}")
            except ValueError:
                print("Please enter a valid number")

    def select_visualization_type(self) -> str:
        """Let user select which visualization type to generate"""
        print("\nSelect visualization type:")
        print("=======================")
        print("1. Circular Connectogram")
        print("2. 3D Brain Visualization")
        print("3. Glass Brain Visualization")
        print("4. All Visualizations")
        
        while True:
            try:
                choice = int(input("\nSelect visualization type (1-4): "))
                if choice == 1:
                    return "circular"
                elif choice == 2:
                    return "brain_3d"
                elif choice == 3:
                    return "glass_brain"
                elif choice == 4:
                    return "all"
                else:
                    print("Please enter a number between 1 and 4")
            except ValueError:
                print("Please enter a valid number")

    def select_connection_display(self) -> str:
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

    def read_csv_file(self, file_path):
        """Read CSV file and create a connectivity matrix"""
        logging.info(f"Reading CSV file: {file_path}")
        
        # Read the raw content of the file as text
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Create a matrix to store the connections
        matrix = np.zeros((len(self.region_labels), len(self.region_labels)))
        connection_count = 0
        
        # Parse the CSV file line by line
        for i, line in enumerate(lines):
            if i >= len(self.region_labels):  # Ensure we only read expected number of rows
                break
            
            values = line.strip().split(',')
            for j, val in enumerate(values):
                if j >= len(self.region_labels):  # Ensure we only read expected number of columns
                    break
                
                # If the value is exactly "0.0" or "0", it's no connection
                # Otherwise, it's a connection with the given strength
                if val != "0.0" and val != "0":
                    # CRITICAL FIX: The CSV file has source nodes in columns and target nodes in rows
                    # So matrix[i, j] represents a connection FROM column j TO row i
                    matrix[i, j] = float(val)
                    connection_count += 1
        
        logging.info(f"Matrix shape: {matrix.shape}")
        logging.info(f"Found {connection_count} connections out of {len(self.region_labels)**2} possible connections")
        
        return matrix

    def process_folder(self, folder_path: FilePath) -> tuple:
        """
        Process a single method/frequency folder
        
        Returns:
        --------
        tuple: (matrix, method_freq, output_base_path)
        """
        try:
            # Get the method and frequency from the folder name
            method_freq = folder_path.name.split('_', 1)[1]  # Skip the "Cue_" prefix
            
            # Find the specific CSV file based on analysis type
            csv_pattern = f"*L-R_{self.analysis_type.lower()}.csv"
            csv_files = list(folder_path.glob(csv_pattern))
            
            if not csv_files:
                logging.warning(f"No {csv_pattern} file found in {folder_path}")
                return None, None, None
            
            # Use the first matching CSV file found
            csv_file = csv_files[0]
            matrix = self.read_csv_file(csv_file)
            
            # Create base output path
            output_base_path = folder_path / f"{method_freq}_L-R_{self.analysis_type.lower()}"
            
            return matrix, method_freq, output_base_path
            
        except Exception as e:
            logging.error(f"Error processing folder {folder_path}: {str(e)}")
            return None, None, None

    def get_region_category(self, region: str) -> str:
        """Get the category of a brain region based on its name"""
        for category, regions in self.region_categories.items():
            if region in regions:
                return category
        return 'Other'

    def get_region_color(self, region: str) -> str:
        """Get the color for a brain region based on its category"""
        category = self.get_region_category(region)
        return self.category_colors.get(category, 'gray')

    def generate_connectivity_report(self, matrix, method_freq, analysis_type, output_path):
        """Generate a detailed connectivity report"""
        
        # Create report content
        report = [
            f"CONNECTIVITY REPORT: {method_freq} {analysis_type}",
            "=" * 80,
            f"\nTotal connections: {np.count_nonzero(matrix)}\n",
            "\nSUMMARY OF OUTGOING CONNECTIONS BY SOURCE NODE",
            "=" * 80 + "\n"
        ]
        
        # Track statistics
        left_to_left = 0
        right_to_right = 0
        left_to_right = 0
        right_to_left = 0
        positive_connections = 0
        negative_connections = 0
        outgoing_counts = defaultdict(int)
        incoming_counts = defaultdict(int)
        
        # Process nodes in the same order as v3
        for j, source in enumerate(self.region_labels):  # j is column (source)
            outgoing = []
            for i, target in enumerate(self.region_labels):  # i is row (target)
                if i != j:  # Skip self-connections
                    weight = matrix[i, j]  # Connection FROM source (column j) TO target (row i)
                    if weight != 0:
                        connection_str = f"{source} {'-->' if weight > 0 else '--|'} {target}: {weight:.6f}"
                        outgoing.append((abs(weight), connection_str))
                    
                    # Update statistics
                    if weight > 0:
                        positive_connections += 1
                    elif weight < 0:  # Only count actual negative values
                        negative_connections += 1
                    
                    if weight != 0:  # Only count non-zero connections
                        outgoing_counts[source] += 1
                        incoming_counts[target] += 1
                    
                    # Update hemisphere statistics
                    if weight != 0:  # Only count non-zero connections
                        source_hem = 'L' if source.endswith('-L') else 'R'
                        target_hem = 'L' if target.endswith('-L') else 'R'
                        if source_hem == 'L' and target_hem == 'L':
                            left_to_left += 1
                        elif source_hem == 'R' and target_hem == 'R':
                            right_to_right += 1
                        elif source_hem == 'L' and target_hem == 'R':
                            left_to_right += 1
                        else:
                            right_to_left += 1
            
            # Add node's connections to report
            if outgoing:
                report.append(f"\n{source}: {len(outgoing)} outgoing connections")
                report.append("-" * 50)
                # Sort by absolute weight
                for _, conn in sorted(outgoing, reverse=True):
                    report.append(f"  {conn}")  # Add two spaces for indentation
            else:
                report.append(f"\n{source}: No outgoing connections")
        
        # Add overall statistics
        report.extend([
            "\n" + "=" * 80,
            "OVERALL CONNECTIVITY STATISTICS",
            "=" * 80,
            f"\nLeft --> Left connections: {left_to_left}",
            f"Right --> Right connections: {right_to_right}",
            f"Left --> Right connections: {left_to_right}",
            f"Right --> Left connections: {right_to_left}\n",
            f"Positive connections: {positive_connections}",
            f"Negative connections: {negative_connections}\n",
            "Top 5 regions with most outgoing connections:"
        ])
        
        # Add top outgoing connections
        for node, count in sorted(outgoing_counts.items(), key=lambda x: (-x[1], x[0]))[:5]:
            report.append(f"  {node}: {count} connections")
        
        report.append("\nTop 5 regions with most incoming connections:")
        for node, count in sorted(incoming_counts.items(), key=lambda x: (-x[1], x[0]))[:5]:
            report.append(f"  {node}: {count} connections")
        
        report.append("\nAnalysis complete.")
        
        # Write report to file
        try:
            with open(output_path, 'w') as f:
                f.write('\n'.join(report))
        except Exception as e:
            logging.error(f"Error writing report to {output_path}: {str(e)}")
            raise
        
        return output_path

class CircularConnectogramVisualizer:
    """Creates circular connectogram visualizations"""
    
    def __init__(self, data_handler):
        """Initialize with data handler reference"""
        self.data_handler = data_handler
    
    def create_connectogram(self, matrix, output_path, title, connection_type="all"):
        """Create circular connectogram visualization"""
        
        # Create figure with black background and proper margins
        fig, ax = plt.subplots(figsize=(15, 15), facecolor='black')
        ax.set_facecolor('black')
        
        # Calculate max weight based on connection type
        if connection_type == "positive":
            # Only consider positive values
            max_weight = np.max(np.maximum(matrix, 0))
        elif connection_type == "negative":
            # Only consider absolute values of negative connections
            max_weight = np.max(np.abs(np.minimum(matrix, 0)))
        else:  # "all"
            # Consider all values (absolute for negatives)
            max_weight = np.max(np.abs(matrix))
        
        # Create graph
        G = nx.DiGraph()
        
        # Add nodes with colors based on region type
        for i, label in enumerate(self.data_handler.region_labels):
            region_type = next(region for region, nodes in self.data_handler.region_categories.items() 
                             if label in nodes)
            G.add_node(i, label=label, color=self.data_handler.category_colors[region_type])
        
        # CRITICAL FIX: Add edges with correct direction based on the matrix structure
        # The matrix has source nodes in columns and target nodes in rows
        # So matrix[i, j] represents a connection FROM column j TO row i
        for i in range(len(matrix)):
            for j in range(len(matrix)):
                if i != j and matrix[i, j] != 0:
                    weight = matrix[i, j]
                    # The connection is FROM j TO i (column to row)
                    # Only add edges that match the connection_type
                    if connection_type == "positive" and weight > 0:
                        G.add_edge(j, i, weight=weight)  # j to i: column to row
                    elif connection_type == "negative" and weight < 0:
                        G.add_edge(j, i, weight=weight)  # j to i: column to row
                    elif connection_type == "all":
                        G.add_edge(j, i, weight=weight)  # j to i: column to row
        
        # Define node positions with region grouping
        pos = {}
        
        # Group nodes by region category - ensure proper ordering within each group
        region_groups = {
            'right': {
                'Frontal': ['FEF-R', 'PMd-R', 'PMv-R', 'SMA-R'],
                'Sensorimotor': ['S1-R', 'M1-R'],  # Keep S1 and M1 together
                'Parietal': ['AG-R', 'IPL-R', 'POJ-R', 'SPL-R', 'SPOC-R', 'mIPS-R'],
                'Temporal': ['STS-R'],  # FIXED: STS-R should be on the right side
                'Visual': ['V1-R', 'V3-R', 'VIP-R']
            },
            'left': {
                'Frontal': ['FEF-L', 'PMd-L', 'PMv-L', 'SMA-L'],
                'Sensorimotor': ['S1-L', 'M1-L'],  # Keep S1 and M1 together
                'Parietal': ['AG-L', 'IPL-L', 'POJ-L', 'SPL-L', 'SPOC-L', 'mIPS-L'],
                'Temporal': ['STS-L'],  # FIXED: STS-L should be on the left side
                'Visual': ['V1-L', 'V3-L', 'VIP-L']
            }
        }
        
        # Define the order of region categories (from top to bottom)
        region_order = ['Frontal', 'Sensorimotor', 'Parietal', 'Temporal', 'Visual']
        
        # Map region labels to their indices in the data_handler.region_labels list
        label_to_idx = {label: i for i, label in enumerate(self.data_handler.region_labels)}
        
        # Apply scaling factor to make circle smaller to avoid overlap with colorbar
        circle_scale = 0.75  # Reduced from 0.85 to 0.75
        
        # Calculate positions with region grouping and gaps between regions
        # Right hemisphere (right side of vertical diameter)
        angle_start_right = np.pi/2  # Start from top
        
        # Track the ending angle of Parietal region and starting angle of Visual region
        parietal_end_angle_right = 0
        visual_start_angle_right = 0

        for region_type in region_order:
            nodes_in_region = region_groups['right'][region_type]
            num_nodes = len(nodes_in_region)
            
            # Calculate angle span for this region (smaller than even distribution)
            # This creates tighter groups with gaps between regions
            region_span = np.pi * 0.8 / len(region_order)  # 80% of even distribution
            
            # Special handling for different region types
            if region_type == 'Sensorimotor':
                # Make Sensorimotor nodes (red) very close together
                node_span = region_span * 0.3  # Very tight grouping
                group_center = angle_start_right - (region_span * 0.5)  # Center of the region
                
                # Position nodes very close to each other
                for i, label in enumerate(nodes_in_region):
                    # Calculate angle with very small offset from center
                    angle = group_center - (i * node_span / max(1, num_nodes - 1) * 0.5)
                    
                    node_idx = label_to_idx[label]
                    # Position on right side (positive x)
                    pos[node_idx] = (circle_scale * np.cos(angle), circle_scale * np.sin(angle))
            
            elif region_type == 'Visual':
                # Make Visual nodes (blue) more spread out
                node_span = region_span * 0.6  # Increased from 0.4 to 0.6 for more spacing
                group_center = angle_start_right - (region_span * 0.5)  # Center of the region
                visual_start_angle_right = angle_start_right  # Track the starting angle of Visual region
                
                # Position nodes with more space between them
                for i, label in enumerate(nodes_in_region):
                    # Calculate angle with larger offset from center
                    angle = group_center - (i * node_span / max(1, num_nodes - 1) * 0.8)  # Increased from 0.6 to 0.8
                    
                    node_idx = label_to_idx[label]
                    # Position on right side (positive x)
                    pos[node_idx] = (circle_scale * np.cos(angle), circle_scale * np.sin(angle))
            
            elif region_type == 'Parietal':
                # Make Parietal nodes (orange) more spread out
                node_span = region_span * 0.85  # Increased from 0.7 to 0.85 for more spacing
                
                # Calculate angles for nodes in this region
                for i, label in enumerate(nodes_in_region):
                    if num_nodes > 1:
                        # Use a larger fraction of the region span for wider grouping
                        angle = angle_start_right - (region_span * 0.5) - (i * node_span / (num_nodes - 0.5))  # Added -0.5 to increase spacing
                    else:
                        angle = angle_start_right - (region_span * 0.5)  # Center single nodes
                        
                    node_idx = label_to_idx[label]
                    # Position on right side (positive x)
                    pos[node_idx] = (circle_scale * np.cos(angle), circle_scale * np.sin(angle))
            
            elif region_type == 'Temporal':
                # Position STS-R on the RIGHT side
                for i, label in enumerate(nodes_in_region):
                    if label == 'STS-R':  # Only process STS-R here
                        node_idx = label_to_idx[label]
                        # Position between Visual and Parietal on the RIGHT side
                        temporal_angle = 4.5  # Adjust this value to position between Visual and Parietal
                        pos[node_idx] = (circle_scale * np.cos(temporal_angle), circle_scale * np.sin(temporal_angle))
            
            else:
                # Standard positioning for other regions
                node_span = region_span * 0.7  # 70% of region span for standard groups
                
                # Calculate angles for nodes in this region
                for i, label in enumerate(nodes_in_region):
                    if num_nodes > 1:
                        # Use a smaller fraction of the region span for tighter grouping
                        angle = angle_start_right - (region_span * 0.5) - (i * node_span / (num_nodes))
                    else:
                        angle = angle_start_right - (region_span * 0.5)  # Center single nodes
                        
                    node_idx = label_to_idx[label]
                    # Position on right side (positive x)
                    pos[node_idx] = (circle_scale * np.cos(angle), circle_scale * np.sin(angle))
            
            # Move to next region with a gap
            angle_start_right -= region_span
        
        # Left hemisphere (left side of vertical diameter)
        angle_start_left = np.pi/2  # Start from top
        
        # Track the ending angle of Parietal region and starting angle of Visual region
        parietal_end_angle_left = 0
        visual_start_angle_left = 0

        for region_type in region_order:
            nodes_in_region = region_groups['left'][region_type]
            num_nodes = len(nodes_in_region)
            
            # Calculate angle span for this region
            region_span = np.pi * 0.8 / len(region_order)
            
            # Special handling for different region types
            if region_type == 'Sensorimotor':
                # Make Sensorimotor nodes (red) very close together
                node_span = region_span * 0.3  # Very tight grouping
                group_center = angle_start_left - (region_span * 0.5)  # Center of the region
                
                # Position nodes very close to each other
                for i, label in enumerate(nodes_in_region):
                    # Calculate angle with very small offset from center
                    angle = group_center - (i * node_span / max(1, num_nodes - 1) * 0.5)
                    
                    node_idx = label_to_idx[label]
                    # Position on left side (negative x)
                    pos[node_idx] = (-circle_scale * np.cos(angle), circle_scale * np.sin(angle))
            
            elif region_type == 'Visual':
                # Make Visual nodes (blue) more spread out
                node_span = region_span * 0.6  # Increased from 0.4 to 0.6 for more spacing
                group_center = angle_start_left - (region_span * 0.5)  # Center of the region
                visual_start_angle_left = angle_start_left
                
                # Position nodes with more space between them
                for i, label in enumerate(nodes_in_region):
                    # Calculate angle with larger offset from center
                    angle = group_center - (i * node_span / max(1, num_nodes - 1) * 0.8)  # Increased from 0.6 to 0.8
                    
                    node_idx = label_to_idx[label]
                    # Position on left side (negative x)
                    pos[node_idx] = (-circle_scale * np.cos(angle), circle_scale * np.sin(angle))
            
            elif region_type == 'Parietal':
                # Make Parietal nodes (orange) more spread out
                node_span = region_span * 0.85  # Increased from 0.7 to 0.85 for more spacing
                
                # Calculate angles for nodes in this region
                for i, label in enumerate(nodes_in_region):
                    if num_nodes > 1:
                        # Use a larger fraction of the region span for wider grouping
                        angle = angle_start_left - (region_span * 0.5) - (i * node_span / (num_nodes - 0.5))  # Added -0.5 to increase spacing
                    else:
                        angle = angle_start_left - (region_span * 0.5)  # Center single nodes
                        
                    node_idx = label_to_idx[label]
                    # Position on left side (negative x)
                    pos[node_idx] = (-circle_scale * np.cos(angle), circle_scale * np.sin(angle))
            
            elif region_type == 'Temporal':
                # Position STS-L on the LEFT side
                for i, label in enumerate(nodes_in_region):
                    if label == 'STS-L':  # Only process STS-L here
                        node_idx = label_to_idx[label]
                        # Position between Visual and Parietal on the LEFT side
                        temporal_angle = 4.5  # Adjust this value to position between Visual and Parietal
                        pos[node_idx] = (-circle_scale * np.cos(temporal_angle), circle_scale * np.sin(temporal_angle))
            
            else:
                # Standard positioning for other regions
                node_span = region_span * 0.7  # 70% of region span for standard groups
                
                # Calculate angles for nodes in this region
                for i, label in enumerate(nodes_in_region):
                    if num_nodes > 1:
                        # Use a smaller fraction of the region span for tighter grouping
                        angle = angle_start_left - (region_span * 0.5) - (i * node_span / (num_nodes))
                    else:
                        angle = angle_start_left - (region_span * 0.5)  # Center single nodes
                        
                    node_idx = label_to_idx[label]
                    # Position on left side (negative x)
                    pos[node_idx] = (-circle_scale * np.cos(angle), circle_scale * np.sin(angle))
            
            # Move to next region with a gap
            angle_start_left -= region_span
        
        # Manually position STS-L and STS-R at the correct locations
        sts_l_idx = label_to_idx['STS-L']
        sts_r_idx = label_to_idx['STS-R']
        mips_l_idx = label_to_idx['mIPS-L']
        mips_r_idx = label_to_idx['mIPS-R']
        v1_l_idx = label_to_idx['V1-L']
        v1_r_idx = label_to_idx['V1-R']

        # Get positions of reference nodes
        mips_l_pos = pos[mips_l_idx]
        mips_r_pos = pos[mips_r_idx]
        v1_l_pos = pos[v1_l_idx]
        v1_r_pos = pos[v1_r_idx]

        # Position STS-L halfway between mIPS-L and V1-L
        pos[sts_l_idx] = (
            (mips_l_pos[0] + v1_l_pos[0]) / 2,
            (mips_l_pos[1] + v1_l_pos[1]) / 2
        )

        # Position STS-R halfway between mIPS-R and V1-R
        pos[sts_r_idx] = (
            (mips_r_pos[0] + v1_r_pos[0]) / 2,
            (mips_r_pos[1] + v1_r_pos[1]) / 2
        )
        
        # Draw connections with arrows - no need to filter here as we already filtered when adding edges
        for source, target, data in G.edges(data=True):
            weight = data['weight']
            
            # Set color based on weight
            if weight > 0:
                # For positive connections: white (low) to red (high)
                color = plt.cm.Reds(min(0.3 + 0.7 * abs(weight) / max_weight, 1.0))
            else:
                # For negative connections: white (low) to blue (high)
                color = plt.cm.Blues(min(0.3 + 0.7 * abs(weight) / max_weight, 1.0))
            
            # Get source and target positions
            source_x, source_y = pos[source]
            target_x, target_y = pos[target]
            
            # Calculate the node radius (used to adjust arrow endpoints)
            node_radius = 0.02  # Keep the original node radius

            # Calculate the direction vector from source to target
            dx = target_x - source_x
            dy = target_y - source_y
            dist = np.sqrt(dx**2 + dy**2)

            # Normalize the direction vector
            if dist > 0:
                dx, dy = dx/dist, dy/dist

            # Keep arrow endpoints close to nodes but increase curvature
            source_pos = (source_x + node_radius * dx, source_y + node_radius * dy)
            target_pos = (target_x - node_radius * dx, target_y - node_radius * dy)

            # Calculate appropriate curvature based on node positions
            # For nodes that are close to each other in the circle
            angle_diff = abs(np.arctan2(source_y, source_x) - np.arctan2(target_y, target_x))
            if angle_diff > np.pi:
                angle_diff = 2 * np.pi - angle_diff

            # Base curvature on the angle difference
            if angle_diff < 0.5:  # Nodes are very close in angle
                rad = 0.9  # Very high curvature
            elif angle_diff < 1.0:  # Nodes are somewhat close
                rad = 0.7  # High curvature
            elif angle_diff < np.pi/2:  # Nodes are moderately separated
                rad = 0.5  # Medium curvature
            else:  # Nodes are far apart
                rad = 0.3  # Lower curvature

            # Adjust direction based on relative position
            # This ensures curves go toward the inside of the circle
            if source_x * target_y > source_y * target_x:
                rad = -rad

            # Use a more visible arrow style with larger arrow head
            arrow = FancyArrowPatch(
                source_pos,
                target_pos,
                arrowstyle='-|>',
                connectionstyle=f'arc3,rad={rad}',
                mutation_scale=25,
                linewidth=1 + 2 * abs(weight) / max_weight,
                color=color,
                alpha=min(0.7 + 0.3 * abs(weight) / max_weight, 0.95),
                zorder=1
            )
            ax.add_patch(arrow)
        
        # Draw nodes with exact colors from category_colors
        for node in G.nodes():
            x, y = pos[node]
            label = G.nodes[node]['label']
            
            # Get the exact color from the category_colors dictionary
            region_type = next(region for region, nodes in self.data_handler.region_categories.items() 
                             if label in nodes)
            color = self.data_handler.category_colors[region_type]
            
            # Create node circle with smaller radius
            circle = Circle((x, y), 
                           radius=0.02,  # Reduced from 0.04 to 0.02 (half the size)
                           facecolor=color,
                           edgecolor='white',
                           linewidth=0.5,
                           alpha=1.0,
                           zorder=2)
            ax.add_patch(circle)
            
            # Calculate the angle for positioning the label
            angle = np.arctan2(y, x)
            
            # Improved label positioning - always radially outward from center
            label_distance = 0.07  # Increased from 0.05 to 0.07 for more space
            
            # Position label radially outward from the center
            label_x = x + np.cos(angle) * label_distance
            label_y = y + np.sin(angle) * label_distance
            
            # Determine text alignment based on position
            if x > 0:  # Right side
                ha = 'left'
            else:  # Left side
                ha = 'right'
            
            # No rotation - always horizontal text for better readability
            # This is more like the reference image
            plt.text(label_x, label_y, label,
                    color='white',
                    ha=ha,
                    va='center',
                    fontsize=8,
                    fontweight='bold',
                    zorder=2)
        
        # Create legend
        legend_handles = []
        for region_type, color in self.data_handler.category_colors.items():
            patch = patches.Patch(color=color, label=region_type)
            legend_handles.append(patch)
        
        # Position the legend below the title - moved down to avoid overlap
        ax.legend(handles=legend_handles, loc='upper center', 
                 bbox_to_anchor=(0.5, 0.92), ncol=5, frameon=False,  # Changed from 0.95 to 0.92
                 fontsize=10, labelcolor='white')

        # Set the title higher to avoid overlap with the legend
        ax.set_title(title, color='white', fontsize=16, pad=10, y=1.05)
        
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Set axis limits to ensure full circle is visible but smaller to avoid overlap
        ax.set_xlim(-1.0, 1.0)  # Keep these limits the same
        ax.set_ylim(-1.0, 1.0)  # Keep these limits the same
        
        return fig

    def generate_connectivity_report(self, matrix, method_freq, analysis_type, output_path):
        """Generate a detailed connectivity report"""
        
        # Create report content
        report = [
            f"CONNECTIVITY REPORT: {method_freq} {analysis_type}",
            "=" * 80,
            f"\nTotal connections: {np.count_nonzero(matrix)}\n",
            "\nSUMMARY OF OUTGOING CONNECTIONS BY SOURCE NODE",
            "=" * 80 + "\n"
        ]
        
        # Track statistics
        left_to_left = 0
        right_to_right = 0
        left_to_right = 0
        right_to_left = 0
        positive_connections = 0
        negative_connections = 0
        outgoing_counts = defaultdict(int)
        incoming_counts = defaultdict(int)
        
        # Process nodes in the same order as v3
        for j, source in enumerate(self.data_handler.region_labels):  # j is column (source)
            outgoing = []
            for i, target in enumerate(self.data_handler.region_labels):  # i is row (target)
                if i != j:  # Skip self-connections
                    weight = matrix[i, j]  # Connection FROM source (column j) TO target (row i)
                    if weight != 0:
                        connection_str = f"{source} {'-->' if weight > 0 else '--|'} {target}: {weight:.6f}"
                        outgoing.append((abs(weight), connection_str))
                    
                    # Update statistics
                    if weight > 0:
                        positive_connections += 1
                    elif weight < 0:  # Only count actual negative values
                        negative_connections += 1
                    
                    if weight != 0:  # Only count non-zero connections
                        outgoing_counts[source] += 1
                        incoming_counts[target] += 1
                    
                    # Update hemisphere statistics
                    if weight != 0:  # Only count non-zero connections
                        source_hem = 'L' if source.endswith('-L') else 'R'
                        target_hem = 'L' if target.endswith('-L') else 'R'
                        if source_hem == 'L' and target_hem == 'L':
                            left_to_left += 1
                        elif source_hem == 'R' and target_hem == 'R':
                            right_to_right += 1
                        elif source_hem == 'L' and target_hem == 'R':
                            left_to_right += 1
                        else:
                            right_to_left += 1
            
            # Add node's connections to report
            if outgoing:
                report.append(f"\n{source}: {len(outgoing)} outgoing connections")
                report.append("-" * 50)
                # Sort by absolute weight
                for _, conn in sorted(outgoing, reverse=True):
                    report.append(f"  {conn}")  # Add two spaces for indentation
            else:
                report.append(f"\n{source}: No outgoing connections")
        
        # Add overall statistics
        report.extend([
            "\n" + "=" * 80,
            "OVERALL CONNECTIVITY STATISTICS",
            "=" * 80,
            f"\nLeft --> Left connections: {left_to_left}",
            f"Right --> Right connections: {right_to_right}",
            f"Left --> Right connections: {left_to_right}",
            f"Right --> Left connections: {right_to_left}\n",
            f"Positive connections: {positive_connections}",
            f"Negative connections: {negative_connections}\n",
            "Top 5 regions with most outgoing connections:"
        ])
        
        # Add top outgoing connections
        for node, count in sorted(outgoing_counts.items(), key=lambda x: (-x[1], x[0]))[:5]:
            report.append(f"  {node}: {count} connections")
        
        report.append("\nTop 5 regions with most incoming connections:")
        for node, count in sorted(incoming_counts.items(), key=lambda x: (-x[1], x[0]))[:5]:
            report.append(f"  {node}: {count} connections")
        
        report.append("\nAnalysis complete.")
        
        # Write report to file
        try:
            with open(output_path, 'w') as f:
                f.write('\n'.join(report))
        except Exception as e:
            logging.error(f"Error writing report to {output_path}: {str(e)}")
            raise
        
        return output_path

class BrainSurfaceVisualizer:
    """Creates 3D brain surface visualizations with connections"""
    
    def __init__(self, data_handler):
        """Initialize with data handler"""
        self.data_handler = data_handler
        
        # Define standard views
        self.views = [
            {'title': 'Top View', 'view_angle': (90, -90)},
            {'title': 'Side View (Left)', 'view_angle': (0, 180)},
            {'title': 'Side View (Right)', 'view_angle': (0, 0)}
        ]
    
    def create_3d_visualization(self, matrix: np.ndarray, output_path: FilePath, 
                              title: str, connection_type: str = "all") -> plt.Figure:
        """Create 3D brain visualization with connections"""
        # Fetch the fsaverage surface
        fsaverage = datasets.fetch_surf_fsaverage()
        
        # Create figure with black background - adjusted for 3 views instead of 6
        fig = plt.figure(figsize=(36, 12), facecolor='black')  # Height reduced from 30 to 12
        
        # Process each view
        for i, view_info in enumerate(self.views):
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
            max_weight = np.max(np.abs(matrix))
            
            for source_idx, source in enumerate(self.data_handler.region_labels):
                for target_idx, target in enumerate(self.data_handler.region_labels):
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
                        source_coords = self.data_handler.talairach_coordinates[source]
                        target_coords = self.data_handler.talairach_coordinates[target]
                        
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
            for region in self.data_handler.region_labels:
                coords = self.data_handler.talairach_coordinates[region]
                color = self.data_handler.get_region_color(region)
                
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
        
        return fig

    def save_visualization(self, fig, output_path, timestamp=None):
        """Save the visualization to file"""
        if timestamp is None:
            timestamp = self.data_handler.generate_timestamp()
        
        # Add timestamp to filename
        output_path = self.data_handler.add_timestamp_to_filename(output_path, timestamp)
        
        # Save the PNG version with black background
        png_path = output_path.with_suffix('.png')
        fig.savefig(png_path, dpi=600, bbox_inches='tight', facecolor='black')  # Increased from 300 to 600 DPI
        
        # Save the SVG version
        svg_path = output_path.with_suffix('.svg')
        fig.savefig(svg_path, format='svg', bbox_inches='tight', facecolor='black')
        
        plt.close(fig)
        
        return png_path, svg_path

class GlassBrainVisualizer:
    """Creates glass brain visualizations with connections"""
    def __init__(self, data_handler: MEGDataHandler):
        self.data_handler = data_handler
        
        # Define views for glass brain
        self.views = [
            {'title': 'Sagittal', 'kwargs': {'display_mode': 'x', 'cut_coords': 3}},
            {'title': 'Coronal', 'kwargs': {'display_mode': 'y', 'cut_coords': 3}},
            {'title': 'Axial', 'kwargs': {'display_mode': 'z', 'cut_coords': 3}}
        ]

    def create_glass_visualization(self, matrix: np.ndarray, output_path: FilePath, 
                                 title: str, connection_type: str = "all") -> plt.Figure:
        """
        Create glass brain visualization with connections
        
        Parameters:
        -----------
        matrix : np.ndarray
            32x32 connectivity matrix
        output_path : FilePath
            Base path for saving the visualization
        title : str
            Title for the visualization
        connection_type : str
            'all', 'positive', or 'negative'
            
        Returns:
        --------
        plt.Figure
            The generated figure
        """
        # Create figure with black background
        fig = plt.figure(figsize=(24, 8), facecolor='black')
        
        # Calculate maximum weight for scaling
        max_weight = np.max(np.abs(matrix))
        
        # Process each view
        for i, view_info in enumerate(self.views, 1):
            # Create subplot for each view
            ax = plt.subplot(1, 3, i)
            ax.set_facecolor('black')
            
            # Create glass brain
            display = plotting.plot_glass_brain(
                None,
                figure=fig,
                axes=ax,
                **view_info['kwargs'],
                alpha=0.1,
                color='white'
            )
            
            # Draw connections
            for source_idx, source in enumerate(self.data_handler.region_labels):
                for target_idx, target in enumerate(self.data_handler.region_labels):
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
                        source_coords = self.data_handler.talairach_coordinates[source]
                        target_coords = self.data_handler.talairach_coordinates[target]
                        
                        # Set color based on connection weight
                        if weight > 0:
                            color = plt.cm.Reds(min(abs(weight) / max_weight, 1.0))
                        else:
                            color = plt.cm.Blues(min(abs(weight) / max_weight, 1.0))
                        
                        # Draw connection line
                        display.add_edges(
                            [source_coords],
                            [target_coords],
                            edge_cmap=None,
                            edge_color=color,
                            alpha=0.6,
                            edge_kwargs={
                                'linewidth': 1 + 2 * abs(weight) / max_weight,
                                'zorder': 1
                            }
                        )
            
            # Plot regions as spheres
            for region in self.data_handler.region_labels:
                coords = self.data_handler.talairach_coordinates[region]
                color = self.data_handler.get_region_color(region)
                
                display.add_markers(
                    [coords],
                    marker_color=color,
                    marker_size=50,
                    alpha=0.9,
                    edge_color='white',
                    edge_width=0.5
                )
            
            # Set title for this view - moved down to avoid overlap
            ax.set_title(view_info['title'], color='white', pad=40, fontsize=16, y=-0.1)
        
        # Add main title
        plt.suptitle(title, color='white', fontsize=22, y=0.98)
        
        # Adjust layout
        plt.subplots_adjust(
            left=0.02, right=0.98,
            top=0.9, bottom=0.1,
            wspace=0.0, hspace=0.0
        )
        
        return fig

    def add_connection_legend(self, fig: plt.Figure, max_weight: float):
        """Add legend showing connection strength and colors"""
        # Create custom legend elements
        legend_elements = []
        
        # Add positive connection examples
        legend_elements.extend([
            Line2D([0], [0], color=plt.cm.Reds(0.3), lw=1, label='Weak positive'),
            Line2D([0], [0], color=plt.cm.Reds(0.6), lw=2, label='Medium positive'),
            Line2D([0], [0], color=plt.cm.Reds(0.9), lw=3, label='Strong positive')
        ])
        
        # Add negative connection examples
        legend_elements.extend([
            Line2D([0], [0], color=plt.cm.Blues(0.3), lw=1, label='Weak negative'),
            Line2D([0], [0], color=plt.cm.Blues(0.6), lw=2, label='Medium negative'),
            Line2D([0], [0], color=plt.cm.Blues(0.9), lw=3, label='Strong negative')
        ])
        
        # Add legend to figure
        fig.legend(
            handles=legend_elements,
            loc='center right',
            bbox_to_anchor=(0.98, 0.5),
            facecolor='black',
            edgecolor='none',
            fontsize=10,
            labelcolor='white'
        )

class MEGVisualizer:
    """Main coordinator class for MEG visualizations"""
    def __init__(self, config=None):
        """
        Initialize the MEG visualizer with all components
        
        Parameters:
        -----------
        config : dict, optional
            Configuration dictionary with visualization parameters
        """
        # Default configuration
        self.config = {
            'dpi': 300,
            'save_svg': True,
            'figure_sizes': {
                'circular': (15, 15),
                'brain_3d': (36, 30),
                'glass': (24, 8)
            },
            'connection_colors': {
                'positive': plt.cm.Reds,
                'negative': plt.cm.Blues
            },
            'alpha': {
                'connections': 0.6,
                'markers': 0.9,
                'brain': 0.2
            }
        }
        
        # Update with user config if provided
        if config is not None:
            self.config.update(config)
        
        # Validate requirements
        if not NILEARN_AVAILABLE:
            raise ImportError("Nilearn is required for brain visualizations")
        
        # Validate directory structure
        data_dir = FilePath("Data")
        if not data_dir.exists():
            raise FileNotFoundError("Data directory not found")
        
        # Validate directory contents
        datasets = [d for d in data_dir.iterdir() 
                   if d.is_dir() and d.name.startswith("DataSet")]
        if not datasets:
            raise FileNotFoundError("No datasets found in Data directory")
        
        # Initialize components
        self.data_handler = MEGDataHandler()
        self.circular_viz = CircularConnectogramVisualizer(self.data_handler)
        self.surface_viz = BrainSurfaceVisualizer(self.data_handler)
        self.glass_viz = GlassBrainVisualizer(self.data_handler)
        
        # Initialize counters
        self.folders_processed = 0
        self.visualizations_created = 0
        self.errors = 0

    def create_visualizations(self, matrix: np.ndarray, folder_path: FilePath, method_freq: str) -> None:
        timestamp = self.data_handler.generate_timestamp()
        
        try:
            plt.close('all')
            
            # Parse method and freq from filename
            method, freq = method_freq.split('_')
            
            # Generate connectivity report with timestamp
            report_name = f"{method}_{freq}_L-R_{self.data_handler.analysis_type.lower()}_connectivity_report.txt"
            report_path = folder_path / report_name  # Remove timestamp from report name to match v3
            
            self.circular_viz.generate_connectivity_report(
                matrix, method_freq, self.data_handler.analysis_type, report_path)
            logging.info(f"Created connectivity report: {report_path}")
            
            if self.data_handler.connection_display == "separate":
                connection_types = ["positive", "negative"]
            else:
                connection_types = ["all"]
            
            for conn_type in connection_types:
                # Create title
                title = f"{method} {freq} {self.data_handler.analysis_type}\n"
                if conn_type != "all":
                    title += f"{conn_type.capitalize()} Connectivity"
                
                # Base filename pattern matching v3
                base_name = f"{method}_{freq}_L-R_{self.data_handler.analysis_type.lower()}"
                
                if self.data_handler.visualization_type in ["circular", "all"]:
                    # Circular connectogram with v3 naming
                    circular_name = f"{base_name}_circular_connectogram_{conn_type}_{timestamp}"
                    circular_path = folder_path / f"{circular_name}.png"
                    circular_svg = folder_path / f"{circular_name}.svg"
                    
                    fig_circular = self.circular_viz.create_connectogram(
                        matrix, circular_path, title, conn_type)
                    fig_circular.savefig(circular_path, 
                                       dpi=self.config['dpi'], 
                                       bbox_inches='tight', 
                                       facecolor='black')
                    
                    if self.config['save_svg']:
                        fig_circular.savefig(circular_svg, format='svg', 
                                           bbox_inches='tight', 
                                           facecolor='black')
                    plt.close(fig_circular)
                    logging.info(f"Created circular connectogram: {circular_path}")
                    self.visualizations_created += 1
                
                if self.data_handler.visualization_type in ["brain_3d", "all"]:
                    # 3D brain visualization
                    brain_name = f"{base_name}_brain3d_connectogram_{conn_type}_{timestamp}"
                    brain_path = folder_path / f"{brain_name}.png"
                    brain_svg = folder_path / f"{brain_name}.svg"
                    
                    fig_brain = self.surface_viz.create_3d_visualization(
                        matrix, brain_path, title, conn_type)
                    fig_brain.savefig(brain_path, dpi=self.config['dpi'], 
                                    bbox_inches='tight', facecolor='black')
                    if self.config['save_svg']:
                        fig_brain.savefig(brain_svg, format='svg', 
                                        bbox_inches='tight', facecolor='black')
                    plt.close(fig_brain)
                    logging.info(f"Created 3D brain visualization: {brain_path}")
                    self.visualizations_created += 1
                
                if self.data_handler.visualization_type in ["glass_brain", "all"]:
                    # Glass brain visualization
                    glass_name = f"{base_name}_glassbrain_connectogram_{conn_type}_{timestamp}"
                    glass_path = folder_path / f"{glass_name}.png"
                    glass_svg = folder_path / f"{glass_name}.svg"
                    
                    fig_glass = self.glass_viz.create_glass_visualization(
                        matrix, glass_path, title, conn_type)
                    self.glass_viz.add_connection_legend(fig_glass, np.max(np.abs(matrix)))
                    fig_glass.savefig(glass_path, dpi=self.config['dpi'], 
                                    bbox_inches='tight', facecolor='black')
                    if self.config['save_svg']:
                        fig_glass.savefig(glass_svg, format='svg', 
                                        bbox_inches='tight', facecolor='black')
                    plt.close(fig_glass)
                    logging.info(f"Created glass brain visualization: {glass_path}")
                    self.visualizations_created += 1
                
        except Exception as e:
            self.errors += 1
            logging.error(f"Error creating visualizations: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
        finally:
            plt.close('all')

    def process_all_folders(self):
        """Process all method/frequency folders in the selected analysis folder"""
        logging.info(f"Processing all folders in {self.data_handler.analysis_dir.name}...")
        
        # Get all method folders and count total
        method_folders = [d for d in self.data_handler.analysis_dir.iterdir() 
                         if d.is_dir()]
        total_folders = len(method_folders)
        
        if not method_folders:
            logging.warning(f"No method folders found in {self.data_handler.analysis_dir.name}")
            return
        
        logging.info(f"Found {total_folders} folders to process")
        
        for i, folder in enumerate(method_folders, 1):
            try:
                logging.info(f"Processing folder {i}/{total_folders}: {folder.name}")
                
                # Get matrix and output information
                matrix, method_freq, output_base = self.data_handler.process_folder(folder)
                
                if matrix is not None:
                    self.create_visualizations(matrix, folder, method_freq)
                    self.folders_processed += 1
                    logging.info(f"Progress: {self.folders_processed}/{total_folders} folders completed")
                
            except Exception as e:
                self.errors += 1
                logging.error(f"Error processing folder {folder.name}: {str(e)}")
                import traceback
                logging.error(traceback.format_exc())
        
        # Print summary
        logging.info("\n=== Processing Summary ===")
        logging.info(f"Folders processed: {self.folders_processed}")
        logging.info(f"Visualizations created: {self.visualizations_created}")
        logging.info(f"Errors encountered: {self.errors}")
        logging.info(f"Process completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def main():
    """Main entry point"""
    visualizer = MEGVisualizer()
    visualizer.process_all_folders()

if __name__ == "__main__":
    main() 