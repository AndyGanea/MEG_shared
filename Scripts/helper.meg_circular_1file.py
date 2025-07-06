import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.path import Path
import networkx as nx
from pathlib import Path
from datetime import datetime
from collections import defaultdict

def analyze_csv_file():
    """
    Analyze a specific CSV file, treating "0.0" as no connection and any other value as a connection.
    """
    # Path to the specific CSV file
    csv_path = Path("Data/DataSet5_Align_cue_626_to_938_unc/Wilcoxon_03102025-1305/Cue_gPDC_20Hz/gPDC_20Hz_L-R_wilcoxon.csv")
    
    # Check if the file exists
    if not csv_path.exists():
        print(f"File not found: {csv_path}")
        return
    
    print(f"Analyzing file: {csv_path}")
    
    # Generate timestamp for the filename
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    
    # Create output paths in the same folder as the input file
    output_folder = csv_path.parent
    matrix_viz_path = output_folder / f"matrix_visualization_{timestamp}.png"
    connectogram_path = output_folder / f"circular_connectogram_{timestamp}.png"
    
    print(f"Will save visualizations to: {output_folder}")
    
    # Read the raw content of the file as text
    with open(csv_path, 'r') as f:
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
                matrix[i, j] = float(val)
                connection_count += 1
    
    print(f"Matrix shape: {matrix.shape}")
    print(f"Found {connection_count} connections out of {32*32} possible connections")
    
    # Define the CORRECT region labels as provided
    region_labels = [
        "V1-L", "V3-L", "SPOC-L", "AG-L", "POJ-L", "SPL-L", "mIPS-L", "VIP-L", 
        "IPL-L", "STS-L", "S1-L", "M1-L", "SMA-L", "PMd-L", "FEF-L", "PMv-L", 
        "V1-R", "V3-R", "SPOC-R", "AG-R", "POJ-R", "SPL-R", "mIPS-R", "VIP-R", 
        "IPL-R", "STS-R", "S1-R", "M1-R", "SMA-R", "PMd-R", "FEF-R", "PMv-R"
    ]
    
    # Create a mapping from region name to matrix index
    region_to_idx = {region: idx for idx, region in enumerate(region_labels)}
    
    # Find all connections
    connections = []
    for source_region in region_labels:
        for target_region in region_labels:
            if source_region != target_region:  # Skip self-connections
                source_idx = region_to_idx[source_region]
                target_idx = region_to_idx[target_region]
                
                # The connection is FROM source TO target
                # So we look at matrix[target_idx, source_idx]
                strength = matrix[target_idx, source_idx]
                
                if strength != 0.0:  # If there's a connection
                    connections.append((source_region, target_region, strength))
    
    print(f"Found {len(connections)} connections between regions")
    
    # Check if the S1-R to VIP-L connection exists
    s1r_to_vipl = next((conn for conn in connections if conn[0] == "S1-R" and conn[1] == "VIP-L"), None)
    if s1r_to_vipl:
        print(f"Connection FROM S1-R TO VIP-L has strength {s1r_to_vipl[2]}")
    else:
        # Check the raw value in the matrix
        s1r_idx = region_to_idx["S1-R"]
        vipl_idx = region_to_idx["VIP-L"]
        raw_value = matrix[vipl_idx, s1r_idx]
        print(f"No connection FROM S1-R TO VIP-L in the connections list")
        print(f"Raw value in matrix[{vipl_idx}, {s1r_idx}] (VIP-L, S1-R) = {raw_value}")
        
        # Check the raw string value in the CSV
        raw_string = lines[vipl_idx].strip().split(',')[s1r_idx]
        print(f"Raw string value in CSV at row {vipl_idx}, col {s1r_idx}: '{raw_string}'")
    
    # Organize connections by source region
    source_connections = defaultdict(list)
    for source, target, strength in connections:
        source_connections[source].append((target, strength))
    
    # Print summary of outgoing connections for each source node
    print("\n" + "="*80)
    print("SUMMARY OF OUTGOING CONNECTIONS BY SOURCE NODE")
    print("="*80)
    
    # Sort regions by hemisphere and then by name
    sorted_regions = sorted(region_labels, key=lambda x: (x.endswith('-R'), x))
    
    for source in sorted_regions:
        outgoing = source_connections[source]
        if outgoing:
            # Sort connections by absolute strength (strongest first)
            outgoing.sort(key=lambda x: abs(x[1]), reverse=True)
            
            print(f"\n{source}: {len(outgoing)} outgoing connections")
            print("-" * 50)
            
            # Print each outgoing connection
            for i, (target, strength) in enumerate(outgoing, 1):
                direction = "→" if strength > 0 else "⊣"  # Arrow for positive, block for negative
                print(f"  {i}. {source} {direction} {target}: {strength:.6f}")
        else:
            print(f"\n{source}: No outgoing connections")
    
    # Create a visualization of the matrix
    plt.figure(figsize=(12, 10))
    plt.imshow(matrix, cmap='coolwarm', vmin=-np.max(np.abs(matrix)), vmax=np.max(np.abs(matrix)))
    plt.colorbar(label='Connection Strength')
    plt.title('Connection Matrix (0.0 = No Connection)')
    plt.xticks(np.arange(32), region_labels, rotation=90)
    plt.yticks(np.arange(32), region_labels)
    plt.tight_layout()
    plt.savefig(matrix_viz_path)
    print(f"\nSaved matrix visualization to: {matrix_viz_path}")
    
    # Create a circular connectogram
    create_connectogram(matrix, region_labels, connectogram_path)
    print(f"Saved connectogram to: {connectogram_path}")
    
    # Print overall statistics
    print("\n" + "="*80)
    print("OVERALL CONNECTIVITY STATISTICS")
    print("="*80)
    
    # Count connections by hemisphere
    ll_count = len([c for c in connections if c[0].endswith('-L') and c[1].endswith('-L')])
    rr_count = len([c for c in connections if c[0].endswith('-R') and c[1].endswith('-R')])
    lr_count = len([c for c in connections if c[0].endswith('-L') and c[1].endswith('-R')])
    rl_count = len([c for c in connections if c[0].endswith('-R') and c[1].endswith('-L')])
    
    print(f"Left → Left connections: {ll_count}")
    print(f"Right → Right connections: {rr_count}")
    print(f"Left → Right connections: {lr_count}")
    print(f"Right → Left connections: {rl_count}")
    
    # Count positive and negative connections
    pos_count = len([c for c in connections if c[2] > 0])
    neg_count = len([c for c in connections if c[2] < 0])
    
    print(f"\nPositive connections: {pos_count}")
    print(f"Negative connections: {neg_count}")
    
    # Find regions with most outgoing connections
    outgoing_counts = {region: len(source_connections[region]) for region in region_labels}
    top_outgoing = sorted(outgoing_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    
    print("\nTop 5 regions with most outgoing connections:")
    for region, count in top_outgoing:
        print(f"  {region}: {count} connections")
    
    # Find regions with most incoming connections
    incoming_connections = defaultdict(list)
    for source, target, strength in connections:
        incoming_connections[target].append((source, strength))
    
    incoming_counts = {region: len(incoming_connections[region]) for region in region_labels}
    top_incoming = sorted(incoming_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    
    print("\nTop 5 regions with most incoming connections:")
    for region, count in top_incoming:
        print(f"  {region}: {count} connections")
    
    print("\nAnalysis complete. Visualizations saved.")

def get_region_category(region):
    """Get the category of a brain region based on its name."""
    base_region = region.split('-')[0]
    
    if base_region in ['V1', 'V3']:
        return 'Visual'
    elif base_region in ['AG', 'IPL', 'SPL', 'mIPS', 'VIP', 'POJ', 'SPOC']:
        return 'Parietal'
    elif base_region in ['FEF', 'PMd', 'PMv', 'SMA']:
        return 'Frontal'
    elif base_region in ['M1', 'S1']:
        return 'Sensorimotor'
    elif base_region in ['STS']:
        return 'Temporal'
    else:
        return 'Other'

def create_connectogram(matrix, region_labels, output_path):
    """
    Create a circular connectogram visualization of brain connectivity.
    Using improved visualization from meg_circular_connectogram.py
    """
    # Create figure with black background
    fig = plt.figure(figsize=(14, 14), facecolor='black')
    ax = fig.add_subplot(111)
    ax.set_facecolor('black')
    
    # Define category colors
    category_colors = {
        'Visual': 'skyblue',
        'Parietal': 'orange',
        'Frontal': 'purple',
        'Sensorimotor': 'red',
        'Temporal': 'green',
        'Other': 'gray'
    }
    
    # Create a graph
    G = nx.DiGraph()
    
    # Reorganize region labels by hemisphere
    left_regions = [label for label in region_labels if label.endswith('-L')]
    right_regions = [label for label in region_labels if label.endswith('-R')]
    
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
        G.add_node(node_index, label=region, category=get_region_category(region))
        node_index += 1
    
    # Add left hemisphere nodes (on the right side) - SWAPPED
    for i, region in enumerate(left_regions):
        angle = np.pi * (i / len(left_regions) + 0.5) + rotation_angle  # Add rotation
        x = np.cos(angle)
        y = np.sin(angle)
        pos[node_index] = (x, y)
        region_to_node[region] = node_index
        G.add_node(node_index, label=region, category=get_region_category(region))
        node_index += 1
    
    # Create a mapping from region name to matrix index
    region_to_matrix_idx = {region: idx for idx, region in enumerate(region_labels)}
    
    # Find all non-zero connections
    edges_to_draw = []
    
    # In the matrix, [row, col] means FROM col TO row
    for target_region in region_labels:
        for source_region in region_labels:
            if target_region != source_region:  # Skip self-connections
                target_idx = region_to_matrix_idx[target_region]
                source_idx = region_to_matrix_idx[source_region]
                
                # The connection is FROM source TO target
                # So we look at matrix[target_idx, source_idx]
                strength = matrix[target_idx, source_idx]
                
                # Use exact comparison with 0.0
                if strength != 0.0:
                    source_node = region_to_node[source_region]
                    target_node = region_to_node[target_region]
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
    neg_cmap = plt.cm.Blues  # Blue colormap for negative values
    
    # Draw edges with increased curvature and arrows
    for source, target, weight in edges_to_draw:
        source_x, source_y = pos[source]
        target_x, target_y = pos[target]
        
        # Determine color based on weight
        if weight > 0:
            color = pos_cmap(weight / max_pos_weight)
        else:
            color = neg_cmap(weight / min_neg_weight)
        
        # Calculate the angle between the two nodes
        angle = np.arctan2(target_y - source_y, target_x - source_x)
        
        # Calculate the distance between the two nodes
        distance = np.sqrt((target_x - source_x)**2 + (target_y - source_y)**2)
        
        # Determine if the edge crosses hemispheres
        source_is_right = list(region_to_node.keys())[source].endswith('-R')
        target_is_right = list(region_to_node.keys())[target].endswith('-R')
        crosses_hemispheres = source_is_right != target_is_right
        
        # Adjust curvature based on whether the edge crosses hemispheres
        rad = 0.2 if crosses_hemispheres else 0.1
        
        # Draw the curved edge with an arrow
        nx.draw_networkx_edges(
            G, pos,
            edgelist=[(source, target)],
            width=2.0 * abs(weight) / max(abs(max_pos_weight), abs(min_neg_weight)) * 3,
            edge_color=[color],
            connectionstyle=f'arc3, rad={rad}',
            arrowstyle='-|>',
            arrowsize=10,
            alpha=0.7
        )
    
    # Draw nodes with colors based on region category
    node_colors = []
    for node in G.nodes():
        category = G.nodes[node]['category']
        color = category_colors[category]
        node_colors.append(color)
    
    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos,
        node_size=500,
        node_color=node_colors,
        edgecolors='white',
        linewidths=1.5,
        alpha=1.0
    )
    
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
    plt.title('gPDC_20Hz Wilcoxon Connectivity', color='white', size=14, y=1.05)
    
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
    min_val = min_neg_weight
    max_val = max_pos_weight
    
    # Add colorbar
    cax = fig.add_axes([0.80, 0.05, 0.015, 0.20])  # [left, bottom, width, height]
    norm = mcolors.TwoSlopeNorm(vmin=min_val, vcenter=0, vmax=max_val)
    sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm)
    sm.set_array([])
    cb = plt.colorbar(sm, cax=cax)
    cb.set_label('Connection Strength', color='white', fontsize=8, labelpad=5)
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
    
    # Save the figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='black')
    plt.close()

if __name__ == "__main__":
    analyze_csv_file()
