#!/usr/bin/env python3

"""
MEG Circular Connectogram Visualization with File Selection
---------------------------------------------------------
This script allows the user to select a file from the Visual folder
and create circular connectogram visualizations.
"""

import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import logging
import networkx as nx
from matplotlib.patches import FancyArrowPatch
from collections import defaultdict
from matplotlib.patheffects import SimpleLineShadow, Normal
import matplotlib.patheffects as path_effects

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logging.info("Starting MEG Circular Visualization with File Selection")

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
    
    print("\nAvailable files:")
    print("================")
    
    for idx, file_path in enumerate(csv_files, 1):
        print(f"{idx}. {file_path.name}")
    
    while True:
        try:
            choice = int(input("\nSelect file number: "))
            if 1 <= choice <= len(csv_files):
                return csv_files[choice-1]
            else:
                print(f"Please enter a number between 1 and {len(csv_files)}")
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

def create_connectogram(matrix, output_path, title, connection_type="all"):
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
    for i, label in enumerate(region_labels):
        region_type = next(region for region, nodes in region_categories.items() 
                         if label in nodes)
        G.add_node(i, label=label, color=category_colors[region_type])
    
    # Add edges with correct direction based on the matrix structure
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
    
    # Map region labels to their indices in the region_labels list
    label_to_idx = {label: i for i, label in enumerate(region_labels)}
    
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
            visual_start_angle_right = angle_start_right
            
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
            
            parietal_end_angle_right = angle_start_right - region_span
        
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
            
            parietal_end_angle_left = angle_start_left - region_span
        
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
    
    # Draw connections with arrows
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
    
    # Draw nodes
    for node, (x, y) in pos.items():
        node_color = G.nodes[node]['color']
        ax.plot(x, y, 'o', markersize=10, color=node_color, alpha=0.9, zorder=2)
    
    # Add node labels
    for node, (x, y) in pos.items():
        label = G.nodes[node]['label']
        # Adjust label position based on which side of the circle it's on
        if x < 0:
            # Left side - align right
            ax.text(x - 0.05, y, label, color='white', ha='right', va='center', fontsize=8)
        else:
            # Right side - align left
            ax.text(x + 0.05, y, label, color='white', ha='left', va='center', fontsize=8)
    
    # Add title
    ax.set_title(title, color='white', fontsize=16, pad=10, y=1.05)
    
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
    
    # Add colorbar for positive connections
    cbar_ax = fig.add_axes([0.90, 0.25, 0.02, 0.5])  # [left, bottom, width, height]
    cbar = plt.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Positive Connectivity' if connection_type == "positive" else 
                  'Negative Connectivity' if connection_type == "negative" else 
                  'Connectivity', color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    
    # Add max value at the top of the colorbar
    plt.text(0.92, 0.76, f'Max: {max_weight:.3f}', 
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
    
    # Position the legend below the title
    ax.legend(handles=legend_elements, loc='upper center', 
             bbox_to_anchor=(0.5, 0.98), ncol=5, frameon=False, 
             fontsize=10, labelcolor='white')
    
    # Set axis limits slightly larger than the circle
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    
    # Remove axis ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    
    # Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    return fig

def generate_connectivity_report(matrix, output_path):
    """Generate a detailed report of connectivity patterns"""
    # Create a mapping from region name to matrix index
    region_to_idx = {region: idx for idx, region in enumerate(region_labels)}
    
    # Track statistics
    left_to_left = 0
    right_to_right = 0
    left_to_right = 0
    right_to_left = 0
    positive_connections = 0
    negative_connections = 0
    outgoing_counts = defaultdict(int)
    incoming_counts = defaultdict(int)
    
    # Find all connections
    connections = []
    source_connections = defaultdict(list)

    # Process the matrix directly
    for j, source in enumerate(region_labels):  # j is column (source)
        outgoing = []
        for i, target in enumerate(region_labels):  # i is row (target)
            if i != j:  # Skip self-connections
                weight = matrix[i, j]  # Connection FROM source (column j) TO target (row i)
                
                if weight != 0.0:  # If there's a connection
                    connections.append((source, target, weight))
                    source_connections[source].append((target, weight))
                    
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
    
    # Create a report file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"CONNECTIVITY REPORT: gDTF_10Hz Wilcoxon\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Total connections: {len(connections)}\n\n")
        
        f.write("SUMMARY OF OUTGOING CONNECTIONS BY SOURCE NODE\n")
        f.write("="*80 + "\n\n")
        
        # Sort regions by hemisphere and then by name
        sorted_regions = sorted(region_labels, key=lambda x: (x.endswith('-R'), x))
        
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
        f.write(f"Left --> Left connections: {left_to_left}\n")
        f.write(f"Right --> Right connections: {right_to_right}\n")
        f.write(f"Left --> Right connections: {left_to_right}\n")
        f.write(f"Right --> Left connections: {right_to_left}\n")
        
        # Count positive and negative connections
        f.write(f"\nPositive connections: {positive_connections}\n")
        f.write(f"Negative connections: {negative_connections}\n")
        
        # Find regions with most outgoing connections
        top_outgoing = sorted(outgoing_counts.items(), key=lambda x: (-x[1], x[0]))[:5]
        
        f.write("\nTop 5 regions with most outgoing connections:\n")
        for region, count in top_outgoing:
            f.write(f"  {region}: {count} connections\n")
        
        # Find regions with most incoming connections
        top_incoming = sorted(incoming_counts.items(), key=lambda x: (-x[1], x[0]))[:5]
        
        f.write("\nTop 5 regions with most incoming connections:\n")
        for region, count in top_incoming:
            f.write(f"  {region}: {count} connections\n")
        
        f.write("\nAnalysis complete.\n")
    
    logging.info(f"Saved connectivity report to: {output_path}")

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
    method_freq = filename.split('_')[0] + "_" + filename.split('_')[1]
    analysis_type = filename.split('_')[-1]
    
    # Output folder is the Visual folder
    output_folder = visual_folder
    
    # Generate connectivity report
    report_path = output_folder / f"{method_freq}_{analysis_type}_connectivity_report_{timestamp}.txt"
    generate_connectivity_report(matrix, report_path)
    logging.info(f"Created connectivity report: {report_path}")
    
    if connection_display == "all":
        # Create single visualization with all connections
        title = f"{method_freq} {analysis_type}\nConnectivity"
        output_path = output_folder / f"{method_freq}_{analysis_type}_circular_connectogram_all_{timestamp}.png"
        svg_path = output_path.with_suffix('.svg')
        
        fig = create_connectogram(matrix, output_path, title, connection_type="all")
        
        # Save the PNG version with black background
        fig.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='black')
        
        # Save the SVG version
        fig.savefig(svg_path, format='svg', bbox_inches='tight', facecolor='black')
        
        plt.close(fig)
        logging.info(f"Created visualization: {output_path}")
        
    else:  # "separate"
        # Create positive connections visualization
        pos_title = f"{method_freq} {analysis_type}\nPositive Connectivity"
        pos_output_path = output_folder / f"{method_freq}_{analysis_type}_circular_connectogram_positive_{timestamp}.png"
        pos_svg_path = pos_output_path.with_suffix('.svg')
        
        pos_fig = create_connectogram(matrix, pos_output_path, pos_title, connection_type="positive")
        
        # Save the PNG version with black background
        pos_fig.savefig(pos_output_path, dpi=600, bbox_inches='tight', facecolor='black')
        
        # Save the SVG version
        pos_fig.savefig(pos_svg_path, format='svg', bbox_inches='tight', facecolor='black')
        
        plt.close(pos_fig)
        logging.info(f"Created positive connections visualization: {pos_output_path}")
        
        # Create negative connections visualization
        neg_title = f"{method_freq} {analysis_type}\nNegative Connectivity"
        neg_output_path = output_folder / f"{method_freq}_{analysis_type}_circular_connectogram_negative_{timestamp}.png"
        neg_svg_path = neg_output_path.with_suffix('.svg')
        
        neg_fig = create_connectogram(matrix, neg_output_path, neg_title, connection_type="negative")
        
        # Save the PNG version with black background
        neg_fig.savefig(neg_output_path, dpi=600, bbox_inches='tight', facecolor='black')
        
        # Save the SVG version
        neg_fig.savefig(neg_svg_path, format='svg', bbox_inches='tight', facecolor='black')
        
        plt.close(neg_fig)
        logging.info(f"Created negative connections visualization: {neg_output_path}")
    
    logging.info("Visualization completed successfully!")

if __name__ == "__main__":
    run() 