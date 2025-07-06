#!/usr/bin/env python3

"""
MEG Circular Connectogram Visualization V2
------------------
This script creates professional Circos-style connectogram visualizations from MEG connectivity matrices.
It reads CSV files with improved parsing that correctly handles "0.0" strings as no connection.

Key improvements:
1. Correctly interprets "0.0" strings as no connection
2. Properly handles positive and negative connection values
3. Creates high-quality circular connectograms with directional arrows
4. Provides detailed console output of connectivity patterns
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
from collections import defaultdict
import matplotlib.path as mpath
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Circle
from scipy.special import comb

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
        
        # Get user preference for connection visualization
        self.connection_display = self.select_connection_display()
        
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
            print(f"No folders found for {self.analysis_type} analysis.")
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
    
    def setup_logging(self):
        """Setup logging to file and console"""
        # Create logs directory if it doesn't exist
        self.logs_dir.mkdir(exist_ok=True)
        
        # Create a timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_file = self.logs_dir / f"connectogram_{timestamp}.log"
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        logging.info(f"Starting MEG Connectogram Visualization V2")
        logging.info(f"Selected dataset: {self.dataset_dir.name}")
        logging.info(f"Selected analysis type: {self.analysis_type}")
        logging.info(f"Selected analysis folder: {self.analysis_dir.name}")
        logging.info(f"Log file: {log_file}")

    def get_region_category(self, region):
        """Get the category of a brain region based on its name."""
        for category, regions in self.region_categories.items():
            if region in regions:
                return category
        return 'Other'

    def create_connectogram(self, matrix, output_path, title, connection_type="all"):
        """Create a circular connectogram visualization"""
        # Import required for Bezier curves
        from scipy.special import comb
        import matplotlib.path as mpath
        from matplotlib.patches import FancyArrowPatch, Circle
        import matplotlib.patches as mpatches
        
        # Function to compute Bezier curve points
        def bezier_curve(points, num=200):
            N = len(points)
            t = np.linspace(0, 1, num)
            curve = np.zeros((num, 2))
            for i in range(N):
                curve += np.outer(comb(N-1, i) * t**i * (1-t)**(N-1-i), points[i])
            return curve
        
        # Create a directed graph
        G = nx.DiGraph()
        
        # Define region colors and their labels for the legend according to Andy's request
        region_colors = {
            'Visual': 'dodgerblue',
            'Parietal': 'darkorange',
            'Temporal': 'limegreen',
            'Sensorimotor': 'crimson',
            'Frontal': 'mediumpurple'
        }
        
        # Add nodes with labels and colors
        for i, label in enumerate(self.region_labels):
            # Determine node color based on region type according to Andy's categorization
            if label.startswith('V'):  # Visual regions
                color = region_colors['Visual']
                region_type = 'Visual'
            elif (label.startswith('SPOC') or label.startswith('AG') or 
                  label.startswith('POJ') or label.startswith('SPL') or 
                  label.startswith('mIPS') or label.startswith('VIP') or 
                  label.startswith('IPL')):  # Parietal regions
                color = region_colors['Parietal']
                region_type = 'Parietal'
            elif label.startswith('STS'):  # Temporal regions
                color = region_colors['Temporal']
                region_type = 'Temporal'
            elif label.startswith('S1') or label.startswith('M1'):  # Sensorimotor regions
                color = region_colors['Sensorimotor']
                region_type = 'Sensorimotor'
            elif (label.startswith('SMA') or label.startswith('PMd') or 
                  label.startswith('FEF') or label.startswith('PMv')):  # Frontal regions
                color = region_colors['Frontal']
                region_type = 'Frontal'
            else:
                color = 'gray'
                region_type = 'Other'
            
            G.add_node(i, label=label, color=color, region_type=region_type)
        
        # Add edges based on the connection type
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if i != j:  # Skip self-connections
                    weight = matrix[i, j]
                    
                    # Only add connections based on the selected type
                    if connection_type == "all" and weight != 0:
                        G.add_edge(i, j, weight=weight)
                    elif connection_type == "positive" and weight > 0:
                        G.add_edge(i, j, weight=weight)
                    elif connection_type == "negative" and weight < 0:
                        G.add_edge(i, j, weight=abs(weight))  # Use absolute value for visualization
        
        # Create figure
        plt.figure(figsize=(14, 14), facecolor='black')
        ax = plt.gca()
        ax.set_facecolor('black')
        
        # Separate nodes into left and right hemisphere
        left_nodes = []
        right_nodes = []
        
        for i, label in enumerate(self.region_labels):
            if label.endswith('-L'):
                left_nodes.append((i, label))
            else:  # label.endswith('-R')
                right_nodes.append((i, label))
        
        # Sort nodes by region name to ensure consistent ordering
        left_nodes.sort(key=lambda x: x[1])
        right_nodes.sort(key=lambda x: x[1])
        
        # Function to get region type for a label
        def get_region_type(label):
            if label.startswith('V'):  # Visual regions
                return 'Visual'
            elif (label.startswith('SPOC') or label.startswith('AG') or 
                  label.startswith('POJ') or label.startswith('SPL') or 
                  label.startswith('mIPS') or label.startswith('VIP') or 
                  label.startswith('IPL')):  # Parietal regions
                return 'Parietal'
            elif label.startswith('STS'):  # Temporal regions
                return 'Temporal'
            elif label.startswith('S1') or label.startswith('M1'):  # Sensorimotor regions
                return 'Sensorimotor'
            elif (label.startswith('SMA') or label.startswith('PMd') or 
                  label.startswith('FEF') or label.startswith('PMv')):  # Frontal regions
                return 'Frontal'
            else:
                return 'Other'
        
        # Group nodes by region type
        left_groups = {}
        right_groups = {}
        
        for node_idx, label in left_nodes:
            region_type = get_region_type(label)
            if region_type not in left_groups:
                left_groups[region_type] = []
            left_groups[region_type].append((node_idx, label))
        
        for node_idx, label in right_nodes:
            region_type = get_region_type(label)
            if region_type not in right_groups:
                right_groups[region_type] = []
            right_groups[region_type].append((node_idx, label))
        
        # Define the order of region types (from top to bottom)
        region_order = ['Frontal', 'Sensorimotor', 'Parietal', 'Temporal', 'Visual']
        
        # Position nodes manually with gaps between region types
        pos = {}
        
        # Calculate the angular gap to leave at the top and bottom
        top_gap = np.pi/6  # Gap at top
        bottom_gap = np.pi/6  # Gap at bottom
        
        # Define the gap size between region groups (in radians)
        group_gap = np.pi/12  # Gap between region groups
        
        # Calculate positions for right hemisphere nodes (left side of circle)
        current_angle = np.pi/2 + top_gap
        
        for region_type in region_order:
            if region_type in right_groups:
                nodes = right_groups[region_type]
                num_nodes = len(nodes)
                
                # Calculate angle span for this group
                if region_type == region_order[-1]:  # Last group
                    angle_span = 3*np.pi/2 - bottom_gap - current_angle
                else:
                    # Estimate angle needed for this group based on node count
                    angle_span = (np.pi - top_gap - bottom_gap - (len(region_order)-1)*group_gap) * (num_nodes / len(right_nodes))
                
                # Position nodes in this group
                for i, (node_idx, _) in enumerate(nodes):
                    angle = current_angle + (angle_span * i / max(1, num_nodes-1))
                    pos[node_idx] = (-np.cos(angle), np.sin(angle))
                
                # Move to next group
                current_angle += angle_span + group_gap
        
        # Calculate positions for left hemisphere nodes (right side of circle)
        current_angle = np.pi/2 + top_gap
        
        for region_type in region_order:
            if region_type in left_groups:
                nodes = left_groups[region_type]
                num_nodes = len(nodes)
                
                # Calculate angle span for this group
                if region_type == region_order[-1]:  # Last group
                    angle_span = 3*np.pi/2 - bottom_gap - current_angle
                else:
                    # Estimate angle needed for this group based on node count
                    angle_span = (np.pi - top_gap - bottom_gap - (len(region_order)-1)*group_gap) * (num_nodes / len(left_nodes))
                
                # Position nodes in this group
                for i, (node_idx, _) in enumerate(nodes):
                    angle = current_angle + (angle_span * i / max(1, num_nodes-1))
                    pos[node_idx] = (np.cos(angle), np.sin(angle))
                
                # Move to next group
                current_angle += angle_span + group_gap
        
        # Create a dictionary to map node indices to hemisphere
        node_hemisphere = {}
        for node_idx, _ in left_nodes:
            node_hemisphere[node_idx] = 'L'
        for node_idx, _ in right_nodes:
            node_hemisphere[node_idx] = 'R'
        
        # Find min and max weights for colorbar
        if len(G.edges()) > 0:
            if connection_type == "all":
                min_weight = min([G[u][v]['weight'] for u, v in G.edges()])
                max_weight = max([G[u][v]['weight'] for u, v in G.edges()])
            elif connection_type == "positive":
                min_weight = min([G[u][v]['weight'] for u, v in G.edges()]) if G.edges() else 0
                max_weight = max([G[u][v]['weight'] for u, v in G.edges()]) if G.edges() else 1
            else:  # "negative"
                # For negative connections, we're using absolute values for visualization
                min_weight = min([abs(G[u][v]['weight']) for u, v in G.edges()]) if G.edges() else 0
                max_weight = max([abs(G[u][v]['weight']) for u, v in G.edges()]) if G.edges() else 1
        else:
            min_weight = 0
            max_weight = 1
        
        # Define node size and border width
        node_size = 0.03
        node_border_width = 1.5
        
        # Define min and max line width for connections
        min_width = 0.5  # Minimum line width
        max_width = 4.0  # Maximum line width
        
        # Calculate normalized weights and scale to desired width range
        if max_weight > min_weight:
            edge_widths = [min_width + (max_width - min_width) * 
                          (G[u][v]['weight'] - min_weight) / (max_weight - min_weight) 
                          for u, v in G.edges()]
        else:
            edge_widths = [min_width for _ in G.edges()]
        
        # Define edge colors based on connection type
        if connection_type == "positive":
            # Use a red colormap for positive connections
            edge_colors = [plt.cm.Reds(G[u][v]['weight'] / max_weight) for u, v in G.edges()]
        elif connection_type == "negative":
            # Use a blue colormap for negative connections
            edge_colors = [plt.cm.Blues(G[u][v]['weight'] / max_weight) for u, v in G.edges()]
        else:
            # For "all", use red for positive and blue for negative
            edge_colors = []
            for u, v in G.edges():
                weight = G[u][v]['weight']
                if weight > 0:
                    edge_colors.append(plt.cm.Reds(weight / max_weight))
                else:
                    edge_colors.append(plt.cm.Blues(abs(weight) / max_weight))
        
        # Draw connections with Bezier curves and arrow heads
        for i, (u, v) in enumerate(G.edges()):
            # Get positions of source and target nodes
            u_x, u_y = pos[u]
            v_x, v_y = pos[v]
            
            # Calculate control points for the Bezier curve
            # For a smoother curve, use a control point that's offset from the midpoint
            mid_x = (u_x + v_x) / 2
            mid_y = (u_y + v_y) / 2
            
            # Calculate perpendicular offset for control point
            # The offset is larger for connections between hemispheres
            dx = v_x - u_x
            dy = v_y - u_y
            dist = np.sqrt(dx**2 + dy**2)
            
            # Determine if this is a cross-hemisphere connection
            cross_hemisphere = node_hemisphere[u] != node_hemisphere[v]
            
            # Adjust control point based on connection type
            if cross_hemisphere:
                # For cross-hemisphere connections, use a control point at the center
                control_x = 0
                control_y = 0
            else:
                # For within-hemisphere connections, use an offset from the midpoint
                # The offset is perpendicular to the line connecting the nodes
                perp_x = -dy / dist
                perp_y = dx / dist
                
                # The offset is larger for connections within the same hemisphere
                offset = 0.3
                control_x = mid_x + perp_x * offset
                control_y = mid_y + perp_y * offset
            
            # Create Bezier curve points
            if cross_hemisphere:
                # For cross-hemisphere, use a quadratic Bezier curve with one control point
                curve_points = bezier_curve([
                    [u_x, u_y],
                    [control_x, control_y],
                    [v_x, v_y]
                ])
            else:
                # For within-hemisphere, use a cubic Bezier curve with two control points
                # This creates a more pronounced curve
                control1_x = u_x + (control_x - u_x) * 0.5
                control1_y = u_y + (control_y - u_y) * 0.5
                control2_x = v_x + (control_x - v_x) * 0.5
                control2_y = v_y + (control_y - v_y) * 0.5
                
                curve_points = bezier_curve([
                    [u_x, u_y],
                    [control1_x, control1_y],
                    [control2_x, control2_y],
                    [v_x, v_y]
                ])
            
            # Calculate direction vector at the end of the curve
            # This is used to position the arrow head
            dx = curve_points[-1, 0] - curve_points[-2, 0]
            dy = curve_points[-1, 1] - curve_points[-2, 1]
            
            # Normalize the direction vector
            mag = np.sqrt(dx**2 + dy**2)
            dx = dx / mag
            dy = dy / mag
            
            # Calculate the point on the node's surface where the arrow should end
            # Move back from the center of the node by the node radius in the direction of the curve
            target_x = v_x - dx * node_size
            target_y = v_y - dy * node_size
            
            # Adjust the curve to end at the node's surface instead of center
            # Replace the last point in the curve with the surface point
            curve_points[-1, 0] = target_x
            curve_points[-1, 1] = target_y
            
            # Draw the connection line
            plt.plot(curve_points[:, 0], curve_points[:, 1], 
                     color=edge_colors[i], linewidth=edge_widths[i], alpha=0.7, zorder=0)
            
            # Calculate arrow points
            # Move slightly back from the target point along the curve
            arrow_base_x = target_x - dx * 0.03
            arrow_base_y = target_y - dy * 0.03
            
            # Calculate perpendicular vector for arrow width
            perp_x = -dy
            perp_y = dx
            
            # Arrow size proportional to edge width but with a minimum and maximum size
            arrow_size = min(max(0.025, edge_widths[i] * 0.01), 0.04)
            
            # Calculate arrow points
            arrow_left_x = arrow_base_x + perp_x * arrow_size
            arrow_left_y = arrow_base_y + perp_y * arrow_size
            arrow_right_x = arrow_base_x - perp_x * arrow_size
            arrow_right_y = arrow_base_y - perp_y * arrow_size
            
            # Draw the arrow head
            plt.fill([target_x, arrow_left_x, arrow_right_x], 
                     [target_y, arrow_left_y, arrow_right_y], 
                     color=edge_colors[i], alpha=0.9, zorder=1)
        
        # Draw nodes on top of connections
        for node in G.nodes():
            x, y = pos[node]
            color = G.nodes[node]['color']
            
            # Create a circle patch for the node
            circle = Circle((x, y), node_size, facecolor=color, edgecolor='white', 
                           linewidth=node_border_width, zorder=2)
            ax.add_patch(circle)
        
        # Add node labels with proper positioning
        for node in G.nodes():
            x, y = pos[node]
            
            # Calculate angle for this node (in radians)
            angle = np.arctan2(y, x)
            angle_deg = angle * 180 / np.pi
            
            # Position labels slightly outside the circle
            label_radius = 1.15
            label_x = label_radius * x
            label_y = label_radius * y
            
            # Get the node label
            node_label = G.nodes[node]['label']
            
            # Set horizontal alignment and rotation based on position
            if x > 0:  # Right half
                ha = 'left'
                rotation = angle_deg
            else:  # Left half
                ha = 'right'
                rotation = angle_deg + 180
            
            # Add the label
            ax.text(label_x, label_y, node_label, 
                   color='white', 
                   fontsize=9,
                   ha=ha, va='center', 
                   rotation=rotation, 
                   rotation_mode='anchor',
                   fontweight='bold')
        
        # Add title
        plt.title(title, color='white', fontsize=16, pad=20)
        
        # Create legend for node colors
        legend_handles = []
        for region_name, color in region_colors.items():
            patch = mpatches.Patch(color=color, label=region_name)
            legend_handles.append(patch)
        
        # Add the legend below the title
        legend = plt.legend(handles=legend_handles, loc='upper center', 
                           bbox_to_anchor=(0.5, 0.95), ncol=len(region_colors),
                           frameon=False, fontsize=10)
        
        # Set legend text color to white
        for text in legend.get_texts():
            text.set_color('white')
        
        # Define colorbar position
        cbar_x = 0.92
        cbar_y = 0.3
        cbar_width = 0.02
        cbar_height = 0.4
        
        # Add colorbar based on connection type with min and max values
        if connection_type == "positive":
            sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=plt.Normalize(0, max_weight))
            sm.set_array([])
            cbar_label = "Positive Connection Strength"
        elif connection_type == "negative":
            sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=plt.Normalize(0, max_weight))
            sm.set_array([])
            cbar_label = "Negative Connection Strength"
        else:
            # For "all", we'll use the positive colormap for the colorbar
            sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=plt.Normalize(0, max_weight))
            sm.set_array([])
            cbar_label = "Connection Strength"
        
        cbar_ax = plt.axes([cbar_x, cbar_y, cbar_width, cbar_height])
        cbar = plt.colorbar(sm, cax=cbar_ax)
        cbar.set_label(cbar_label, color='white')
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
        
        # Add min and max values to colorbar (positioned above and below)
        if connection_type == "negative":
            # For negative connections, show absolute values
            plt.figtext(cbar_x + cbar_width/2, cbar_y + cbar_height + 0.02, 
                       f"Max: {max_weight:.3f}", color='white', ha='center', fontsize=10)
            plt.figtext(cbar_x + cbar_width/2, cbar_y - 0.02, 
                       f"Min: {min_weight:.3f}", color='white', ha='center', fontsize=10)
        else:
            plt.figtext(cbar_x + cbar_width/2, cbar_y + cbar_height + 0.02, 
                       f"Max: {max_weight:.3f}", color='white', ha='center', fontsize=10)
            plt.figtext(cbar_x + cbar_width/2, cbar_y - 0.02, 
                       f"Min: {min_weight:.3f}", color='white', ha='center', fontsize=10)
        
        # Set equal aspect ratio to ensure circle looks like a circle
        ax.set_aspect('equal')
        
        # Set axis limits to ensure proper centering
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.3, 1.3)
        
        # Remove axis
        plt.axis('off')
        
        return plt.gcf()

    def process_folder(self, folder_path):
        """Process a single method/frequency folder with improved CSV parsing"""
        try:
            # Get the method and frequency from the folder name
            method_freq = folder_path.name.split('_', 1)[1]  # Skip the "Cue_" prefix
            
            # Find the specific CSV file based on analysis type
            csv_pattern = f"*L-R_{self.analysis_type.lower()}.csv"
            csv_files = list(folder_path.glob(csv_pattern))
            
            if not csv_files:
                logging.warning(f"No {csv_pattern} file found in {folder_path}")
                return
            
            # Use the first matching CSV file found
            csv_file = csv_files[0]
            logging.info(f"Processing {csv_file}")
            
            # Read the raw content of the file as text
            with open(csv_file, 'r') as f:
                lines = f.readlines()
            
            # Create a matrix to store the connections
            matrix = np.zeros((32, 32))
            connection_count = 0
            
            # Parse the CSV file line by line - USING THE EXACT SAME LOGIC FROM helper.meg_circular_1file.py
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
                        matrix[i, j] = float(val)
                        connection_count += 1
            
            logging.info(f"Matrix shape: {matrix.shape}")
            logging.info(f"Found {connection_count} connections out of {32*32} possible connections")
            
            # Generate timestamp for the filename
            timestamp = datetime.now().strftime("%Y%m%d-%H%M")
            
            # Process based on connection display preference
            if self.connection_display == "all":
                # Create the output path in the same folder as the input file with timestamp
                output_path = folder_path / f"{method_freq}_L-R_{self.analysis_type.lower()}_circular_connectogram_all_{timestamp}.png"
                title = f"{method_freq} {self.analysis_type.replace('_', ' ')} Connectivity"
                
                # Create the visualization
                fig = self.create_connectogram(matrix, output_path, title, connection_type="all")
                
                # Save the PNG version with black background
                plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='black')
                
                # Create SVG version with transparent background
                svg_path = output_path.with_suffix('.svg')
                plt.savefig(svg_path, format='svg', dpi=300, bbox_inches='tight', transparent=True)
                
                plt.close(fig)
                
                self.visualizations_created += 1
                logging.info(f"Created 'all connections' connectogram for {method_freq}")
                logging.info(f"Saved PNG: {output_path}")
                logging.info(f"Saved SVG: {svg_path}")
                
            else:  # "separate"
                # Create visualization for positive connections
                pos_output_path = folder_path / f"{method_freq}_L-R_{self.analysis_type.lower()}_circular_connectogram_positive_{timestamp}.png"
                pos_title = f"{method_freq} {self.analysis_type.replace('_', ' ')} Positive Connectivity"
                
                pos_fig = self.create_connectogram(matrix, pos_output_path, pos_title, connection_type="positive")
                
                # Save the PNG version with black background
                plt.savefig(pos_output_path, dpi=300, bbox_inches='tight', facecolor='black')
                
                # Create SVG version with transparent background
                pos_svg_path = pos_output_path.with_suffix('.svg')
                plt.savefig(pos_svg_path, format='svg', dpi=300, bbox_inches='tight', transparent=True)
                
                plt.close(pos_fig)
                
                self.visualizations_created += 1
                logging.info(f"Created 'positive connections' connectogram for {method_freq}")
                logging.info(f"Saved PNG: {pos_output_path}")
                logging.info(f"Saved SVG: {pos_svg_path}")
                
                # Create visualization for negative connections
                neg_output_path = folder_path / f"{method_freq}_L-R_{self.analysis_type.lower()}_circular_connectogram_negative_{timestamp}.png"
                neg_title = f"{method_freq} {self.analysis_type.replace('_', ' ')} Negative Connectivity"
                
                neg_fig = self.create_connectogram(matrix, neg_output_path, neg_title, connection_type="negative")
                
                # Save the PNG version with black background
                plt.savefig(neg_output_path, dpi=300, bbox_inches='tight', facecolor='black')
                
                # Create SVG version with transparent background
                neg_svg_path = neg_output_path.with_suffix('.svg')
                plt.savefig(neg_svg_path, format='svg', dpi=300, bbox_inches='tight', transparent=True)
                
                plt.close(neg_fig)
                
                self.visualizations_created += 1
                logging.info(f"Created 'negative connections' connectogram for {method_freq}")
                logging.info(f"Saved PNG: {neg_output_path}")
                logging.info(f"Saved SVG: {neg_svg_path}")
            
            # Generate detailed connectivity report
            self.generate_connectivity_report(matrix, folder_path, method_freq, timestamp)
            
        except Exception as e:
            self.errors += 1
            logging.error(f"Error processing {folder_path}: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())

    def generate_connectivity_report(self, matrix, folder_path, method_freq, timestamp):
        """Generate a detailed report of connectivity patterns"""
        # Create a mapping from region name to matrix index
        region_to_idx = {region: idx for idx, region in enumerate(self.region_labels)}
        
        # Find all connections
        connections = []
        for source_region in self.region_labels:
            for target_region in self.region_labels:
                if source_region != target_region:  # Skip self-connections
                    source_idx = region_to_idx[source_region]
                    target_idx = region_to_idx[target_region]
                    
                    # The connection is FROM source TO target
                    # So we look at matrix[target_idx, source_idx]
                    strength = matrix[target_idx, source_idx]
                    
                    if strength != 0.0:  # If there's a connection
                        connections.append((source_region, target_region, strength))
        
        # Organize connections by source region
        source_connections = defaultdict(list)
        for source, target, strength in connections:
            source_connections[source].append((target, strength))
        
        # Create a report file
        report_path = folder_path / f"{method_freq}_connectivity_report_{timestamp}.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"CONNECTIVITY REPORT: {method_freq} {self.analysis_type}\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Total connections: {len(connections)}\n\n")
            
            f.write("SUMMARY OF OUTGOING CONNECTIONS BY SOURCE NODE\n")
            f.write("="*80 + "\n\n")
            
            # Sort regions by hemisphere and then by name
            sorted_regions = sorted(self.region_labels, key=lambda x: (x.endswith('-R'), x))
            
            for source in sorted_regions:
                outgoing = source_connections[source]
                if outgoing:
                    # Sort connections by absolute strength (strongest first)
                    outgoing.sort(key=lambda x: abs(x[1]), reverse=True)
                    
                    f.write(f"\n{source}: {len(outgoing)} outgoing connections\n")
                    f.write("-" * 50 + "\n")
                    
                    # Print each outgoing connection
                    for i, (target, strength) in enumerate(outgoing, 1):
                        # Use simple ASCII characters instead of Unicode
                        direction = "-->" if strength > 0 else "--|"  # ASCII arrow for positive, block for negative
                        f.write(f"  {i}. {source} {direction} {target}: {strength:.6f}\n")
                else:
                    f.write(f"\n{source}: No outgoing connections\n")
            
            # Print overall statistics
            f.write("\n" + "="*80 + "\n")
            f.write("OVERALL CONNECTIVITY STATISTICS\n")
            f.write("="*80 + "\n\n")
            
            # Count connections by hemisphere
            ll_count = len([c for c in connections if c[0].endswith('-L') and c[1].endswith('-L')])
            rr_count = len([c for c in connections if c[0].endswith('-R') and c[1].endswith('-R')])
            lr_count = len([c for c in connections if c[0].endswith('-L') and c[1].endswith('-R')])
            rl_count = len([c for c in connections if c[0].endswith('-R') and c[1].endswith('-L')])
            
            f.write(f"Left --> Left connections: {ll_count}\n")
            f.write(f"Right --> Right connections: {rr_count}\n")
            f.write(f"Left --> Right connections: {lr_count}\n")
            f.write(f"Right --> Left connections: {rl_count}\n")
            
            # Count positive and negative connections
            pos_count = len([c for c in connections if c[2] > 0])
            neg_count = len([c for c in connections if c[2] < 0])
            
            f.write(f"\nPositive connections: {pos_count}\n")
            f.write(f"Negative connections: {neg_count}\n")
            
            # Find regions with most outgoing connections
            outgoing_counts = {region: len(source_connections[region]) for region in self.region_labels}
            top_outgoing = sorted(outgoing_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            
            f.write("\nTop 5 regions with most outgoing connections:\n")
            for region, count in top_outgoing:
                f.write(f"  {region}: {count} connections\n")
            
            # Find regions with most incoming connections
            incoming_connections = defaultdict(list)
            for source, target, strength in connections:
                incoming_connections[target].append((source, strength))
            
            incoming_counts = {region: len(incoming_connections[region]) for region in self.region_labels}
            top_incoming = sorted(incoming_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            
            f.write("\nTop 5 regions with most incoming connections:\n")
            for region, count in top_incoming:
                f.write(f"  {region}: {count} connections\n")
            
            f.write("\nAnalysis complete.\n")
        
        logging.info(f"Saved connectivity report to: {report_path}")

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