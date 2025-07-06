#!/usr/bin/env python3
"""
Complete Brain Visualization Tool with Talairach Coordinates
-----------------------------------------------------------
This script creates comprehensive brain visualizations using Talairach coordinates.
It generates three types of visualizations:

1. 3D Brain Surface Visualization - Shows regions on a 3D cortical surface
2. Glass Brain Visualization - Shows regions in a transparent glass brain view
3. Larger Presentation Version - Higher resolution for presentations

All files are automatically timestamped to track revisions.

Requirements:
    - nilearn
    - matplotlib
    - numpy
    - scipy
    - datetime

Install with: pip install nilearn matplotlib numpy scipy
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import sys
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime
import matplotlib.patches as mpatches

# Import nilearn for brain visualization
try:
    from nilearn import datasets, plotting, surface
    import nibabel as nib
except ImportError:
    print("Installing required libraries...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "nilearn matplotlib numpy scipy"])
    from nilearn import datasets, plotting, surface
    import nibabel as nib

def generate_timestamp():
    """Generate a timestamp string for file naming"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def add_timestamp_to_filename(output_path, timestamp=None):
    """
    Add a timestamp to a filename while avoiding double timestamps.
    
    Parameters:
    -----------
    output_path : str
        Original filename
    timestamp : str, optional
        Timestamp string to use. If None, will generate a new timestamp.
        
    Returns:
    --------
    str : Filename with a single timestamp
    """
    # Generate timestamp if not provided
    if timestamp is None:
        timestamp = generate_timestamp()
    
    # Check if the filename already has a timestamp pattern (YYYYMMDD_HHMMSS)
    if '_20' in output_path and any(char.isdigit() for char in output_path.split('_20')[1][:6]):
        # File already has a timestamp, return as is
        return output_path
    
    # Add timestamp before extension
    name, ext = os.path.splitext(output_path)
    return f"{name}_{timestamp}{ext}"

def visualize_regions_on_brain_surface(output_path=None, resolution=300, figsize=(24, 18), bg_color='black', show_legend=True):
    """
    Create a 3D brain visualization with regions on a cortical surface.
    
    Parameters:
    -----------
    output_path : str or None
        Path to save the output image. If None, a timestamp-based name is generated.
    resolution : int
        DPI resolution for output image
    figsize : tuple
        Figure size (width, height) in inches
    bg_color : str
        Background color for the figure
    show_legend : bool
        Whether to show the legend
    """
    # Generate timestamp for filename
    timestamp = generate_timestamp()
    
    # Create output path with timestamp
    if output_path is None:
        output_path = f"brain_regions_3d_surface_{timestamp}.png"
    else:
        output_path = add_timestamp_to_filename(output_path, timestamp)
    
    # Define the region labels
    region_labels = [
        'V1-L', 'V3-L', 'SPOC-L', 'AG-L', 'POJ-L', 'SPL-L', 'mIPS-L', 'VIP-L', 
        'IPL-L', 'STS-L', 'S1-L', 'M1-L', 'SMA-L', 'PMd-L', 'FEF-L', 'PMv-L',
        'V1-R', 'V3-R', 'SPOC-R', 'AG-R', 'POJ-R', 'SPL-R', 'mIPS-R', 'VIP-R', 
        'IPL-R', 'STS-R', 'S1-R', 'M1-R', 'SMA-R', 'PMd-R', 'FEF-R', 'PMv-R'
    ]
    
    # Define region categories
    region_categories = {
        'Visual': ['V1-L', 'V3-L', 'V1-R', 'V3-R'],
        'Parietal': ['SPOC-L', 'AG-L', 'POJ-L', 'SPL-L', 'mIPS-L', 'VIP-L', 'IPL-L',
                     'SPOC-R', 'AG-R', 'POJ-R', 'SPL-R', 'mIPS-R', 'VIP-R', 'IPL-R'],
        'Temporal': ['STS-L', 'STS-R'],
        'Sensorimotor': ['S1-L', 'M1-L', 'S1-R', 'M1-R'],
        'Frontal': ['SMA-L', 'PMd-L', 'FEF-L', 'PMv-L', 'SMA-R', 'PMd-R', 'FEF-R', 'PMv-R']
    }
    
    # Define colors for region categories
    category_colors = {
        'Visual': 'dodgerblue',      # Blue
        'Parietal': 'darkorange',    # Orange
        'Temporal': 'limegreen',     # Green
        'Sensorimotor': 'crimson',   # Red
        'Frontal': 'mediumpurple'    # Purple
    }
    
    # Define Talairach coordinates for each region
    talairach_coordinates = {
        # Left hemisphere
        'V1-L': [-8, -91, 0],
        'V3-L': [-21, -85, 16],
        'SPOC-L': [-9, -71, 37],
        'AG-L': [-35, -61, 35],
        'POJ-L': [-18, -79, 43],
        'SPL-L': [-23, -54, 46],
        'mIPS-L': [-22, -61, 40],
        'VIP-L': [-37, -40, 44],
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
        'SPOC-R': [10, -77, 34],
        'AG-R': [32, -70, 35],
        'POJ-R': [16, -79, 43],
        'SPL-R': [27, -55, 49],
        'mIPS-R': [23, -62, 40],
        'VIP-R': [37, -44, 47],
        'IPL-R': [41, -41, 39],
        'STS-R': [49, -41, 12],
        'S1-R': [39, -26, 40],
        'M1-R': [37, -23, 52],
        'SMA-R': [3, -7, 49],
        'PMd-R': [21, -14, 61],
        'FEF-R': [31, -2, 45],
        'PMv-R': [48, 8, 21]
    }
    
    # Map region to category and color
    region_to_category = {}
    region_to_color = {}
    
    for category, regions in region_categories.items():
        for region in regions:
            region_to_category[region] = category
            region_to_color[region] = category_colors[category]
    
    # Define views
    views = [
        {'view_angle': (0, 90), 'title': 'Left Lateral'},
        {'view_angle': (0, -90), 'title': 'Right Lateral'},
        {'view_angle': (0, 180), 'title': 'Posterior'},
        {'view_angle': (0, 0), 'title': 'Anterior'},
        {'view_angle': (90, 0), 'title': 'Superior'},
        {'view_angle': (-90, 0), 'title': 'Inferior'}
    ]
    
    # Fetch the fsaverage surface
    fsaverage = datasets.fetch_surf_fsaverage()
    
    # Modify figure and subplot parameters for larger brain display
    fig = plt.figure(figsize=(36, 30), facecolor=bg_color)  # Increased from (30, 24)
    
    # Adjust the plotting parameters for each view
    for i, view_info in enumerate(views):
        # Create larger subplot with minimal spacing
        ax = plt.subplot(2, 3, i+1, projection='3d', facecolor=bg_color)
        
        # Load both hemispheres
        mesh_left = surface.load_surf_mesh(fsaverage.pial_left)
        mesh_right = surface.load_surf_mesh(fsaverage.pial_right)
        
        # Plot hemispheres with adjusted size and position
        x, y, z = mesh_left[0].T
        ax.plot_trisurf(x, y, z, triangles=mesh_left[1], alpha=0.2, color='gray', 
                       linewidth=0, edgecolor='none')
        
        x, y, z = mesh_right[0].T
        ax.plot_trisurf(x, y, z, triangles=mesh_right[1], alpha=0.2, color='gray', 
                       linewidth=0, edgecolor='none')
        
        # Adjust view distance for larger brain display
        ax.dist = 4  # Decreased from 5 to 4 to make brain even larger
        
        # Adjust axis limits for tighter fit around brain
        ax.set_xlim(-70, 70)  # Reduced from (-80, 80)
        ax.set_ylim(-70, 70)  # Reduced from (-80, 80)
        ax.set_zlim(-70, 70)  # Reduced from (-80, 80)
        
        # Function to calculate offset direction based on region position
        def get_label_offset(coords, view_angle):
            elev, azim = view_angle
            
            # Convert azimuth to radians for math calculations
            azim_rad = azim * np.pi / 180
            
            # For superior and inferior views, use x,y position
            if elev == 90 or elev == -90:
                # If point is in left hemisphere
                if coords[0] < 0:
                    return (-5, 0, 0)  # Offset to the left
                else:
                    return (5, 0, 0)   # Offset to the right
            
            # For other views, consider the viewing angle
            # Left lateral view
            if abs(azim - 90) < 30:
                return (0, -6, 0)  # Offset perpendicular to view
            # Right lateral view
            elif abs(azim + 90) < 30:
                return (0, 6, 0)   # Offset perpendicular to view
            # Posterior view
            elif abs(azim - 180) < 30 or abs(azim + 180) < 30:
                # If point is in left hemisphere
                if coords[0] < 0:
                    return (-6, 0, 0)  # Offset to the left
                else:
                    return (6, 0, 0)   # Offset to the right
            # Anterior view
            else:
                # If point is in left hemisphere
                if coords[0] < 0:
                    return (-6, 0, 0)  # Offset to the left
                else:
                    return (6, 0, 0)   # Offset to the right
        
        # Plot the regions as smaller spheres
        for region in region_labels:
            if region in talairach_coordinates:
                coords = talairach_coordinates[region]
                color = region_to_color[region]
                # Smaller dots (reduced size from 100 to 50)
                ax.scatter(coords[0], coords[1], coords[2], 
                          color=color, 
                          s=50,  # Smaller size for dots
                          alpha=0.9,
                          edgecolors='white', 
                          linewidth=0.5, 
                          zorder=10)
                
                # Get appropriate offset for this region and view
                offset = get_label_offset(coords, view_info['view_angle'])
                
                # Add region label at offset position
                # Just show the region code without L/R suffix for cleaner appearance
                region_code = region.split('-')[0]
                ax.text(coords[0] + offset[0], coords[1] + offset[1], coords[2] + offset[2],
                       region_code, 
                       color='white', 
                       fontsize=7,  # Slightly smaller font
                       ha='center', 
                       va='center', 
                       fontweight='bold',
                       zorder=11)
        
        # Set the view angle
        elev, azim = view_info['view_angle']
        ax.view_init(elev=elev, azim=azim)
        
        # Set title
        ax.set_title(view_info['title'], color='white', fontsize=16)
        
        # Remove axes
        ax.set_axis_off()
    
    # Add a title
    plt.suptitle("Brain Regions - Talairach Coordinates", color='white', fontsize=22, y=0.98)
    
    # Add a legend
    if show_legend:
        legend_elements = []
        for category, color in category_colors.items():
            legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                         markerfacecolor=color, markersize=10, label=category))
        
        # Add legend at the bottom
        plt.figlegend(handles=legend_elements, loc='lower center', 
                     ncol=len(category_colors), fontsize=14, frameon=False)
        for text in plt.gcf().legends[0].get_texts():
            text.set_color('white')
    
    # Adjust subplot spacing for better layout
    plt.subplots_adjust(
        left=0.01,    # Minimal left margin
        right=0.85,   # Keep space for region list
        top=0.95,     # Minimal top margin
        bottom=0.02,  # Minimal bottom margin
        wspace=0.0,   # No horizontal spacing between plots
        hspace=0.0    # No vertical spacing between plots
    )
    
    # Move region list panel slightly to the right
    ax_list = plt.axes([0.87, 0.3, 0.12, 0.4], frameon=True)
    ax_list.set_facecolor((0.1, 0.1, 0.1, 0.7))  # Dark semi-transparent background
    ax_list.set_axis_off()
    
    y_pos = 0.95
    ax_list.text(0.5, y_pos, "Region List", ha='center', color='white', fontsize=14, fontweight='bold')
    y_pos -= 0.05
    
    for category, color in category_colors.items():
        y_pos -= 0.03
        ax_list.text(0.5, y_pos, category, ha='center', color=color, fontsize=12, fontweight='bold')
        y_pos -= 0.02
        
        # Get regions in this category and sort them
        regions = sorted([r for r in region_categories.get(category, []) if r.endswith('-L')])
        
        # Display left and right hemisphere regions
        for region_l in regions:
            region_base = region_l[:-2]  # Remove the -L suffix
            region_r = region_base + '-R'
            
            y_pos -= 0.02
            ax_list.text(0.5, y_pos, f"{region_base} (L/R)", ha='center', color='white', fontsize=10)
    
    # Add timestamp to the figure
    fig.text(0.02, 0.02, f"Generated: {timestamp.replace('_', ' ')}", 
             color='white', fontsize=8, ha='left')
    
    # Save the figure
    plt.savefig(output_path, dpi=resolution, bbox_inches='tight', facecolor=bg_color)
    print(f"3D brain visualization saved to: {output_path}")
    
    return fig, output_path

def visualize_with_glass_brain(output_path=None, resolution=300, figsize=(24, 18), bg_color='black'):
    """
    Create a glass brain visualization with properly positioned elements and single legend.
    
    Parameters:
    -----------
    output_path : str or None
        Path to save the output image. If None, a timestamp-based name is generated.
    resolution : int
        DPI resolution for output image
    figsize : tuple
        Figure size (width, height) in inches
    bg_color : str
        Background color for the figure
    """
    # Generate timestamp for filename
    timestamp = generate_timestamp()
    
    # Create output path with timestamp
    if output_path is None:
        output_path = f"brain_regions_glass_brain_{timestamp}.png"
    else:
        output_path = add_timestamp_to_filename(output_path, timestamp)
    
    # Define the region labels
    region_labels = [
        'V1-L', 'V3-L', 'SPOC-L', 'AG-L', 'POJ-L', 'SPL-L', 'mIPS-L', 'VIP-L', 
        'IPL-L', 'STS-L', 'S1-L', 'M1-L', 'SMA-L', 'PMd-L', 'FEF-L', 'PMv-L',
        'V1-R', 'V3-R', 'SPOC-R', 'AG-R', 'POJ-R', 'SPL-R', 'mIPS-R', 'VIP-R', 
        'IPL-R', 'STS-R', 'S1-R', 'M1-R', 'SMA-R', 'PMd-R', 'FEF-R', 'PMv-R'
    ]
    
    # Define region categories
    region_categories = {
        'Visual': ['V1-L', 'V3-L', 'V1-R', 'V3-R'],
        'Parietal': ['SPOC-L', 'AG-L', 'POJ-L', 'SPL-L', 'mIPS-L', 'VIP-L', 'IPL-L',
                     'SPOC-R', 'AG-R', 'POJ-R', 'SPL-R', 'mIPS-R', 'VIP-R', 'IPL-R'],
        'Temporal': ['STS-L', 'STS-R'],
        'Sensorimotor': ['S1-L', 'M1-L', 'S1-R', 'M1-R'],
        'Frontal': ['SMA-L', 'PMd-L', 'FEF-L', 'PMv-L', 'SMA-R', 'PMd-R', 'FEF-R', 'PMv-R']
    }
    
    # Define colors for region categories
    category_colors = {
        'Visual': 'dodgerblue',      # Blue
        'Parietal': 'darkorange',    # Orange
        'Temporal': 'limegreen',     # Green
        'Sensorimotor': 'crimson',   # Red
        'Frontal': 'mediumpurple'    # Purple
    }
    
    # Define Talairach coordinates for each region
    talairach_coordinates = {
        # Left hemisphere
        'V1-L': [-8, -91, 0],
        'V3-L': [-21, -85, 16],
        'SPOC-L': [-9, -71, 37],
        'AG-L': [-35, -61, 35],
        'POJ-L': [-18, -79, 43],
        'SPL-L': [-23, -54, 46],
        'mIPS-L': [-22, -61, 40],
        'VIP-L': [-37, -40, 44],
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
        'SPOC-R': [10, -77, 34],
        'AG-R': [32, -70, 35],
        'POJ-R': [16, -79, 43],
        'SPL-R': [27, -55, 49],
        'mIPS-R': [23, -62, 40],
        'VIP-R': [37, -44, 47],
        'IPL-R': [41, -41, 39],
        'STS-R': [49, -41, 12],
        'S1-R': [39, -26, 40],
        'M1-R': [37, -23, 52],
        'SMA-R': [3, -7, 49],
        'PMd-R': [21, -14, 61],
        'FEF-R': [31, -2, 45],
        'PMv-R': [48, 8, 21]
    }
    
    # Map region to category and color
    region_to_category = {}
    region_to_color = {}
    
    for category, regions in region_categories.items():
        for region in regions:
            region_to_category[region] = category
            region_to_color[region] = category_colors[category]
    
    # Create figure
    fig = plt.figure(figsize=figsize, facecolor=bg_color)
    plt.suptitle("Brain Regions - Talairach Coordinates (Glass Brain View)", 
                color='white', fontsize=22, y=0.98)
    
    # Use valid display modes for Nilearn
    display_modes = ['x', 'y', 'z']  # Sagittal, Coronal, Axial
    titles = ['Sagittal View', 'Coronal View', 'Axial View']
    
    # First create the top row with single views
    for i, (mode, title) in enumerate(zip(display_modes, titles)):
        ax = plt.subplot(2, 3, i+1)
        
        # Create glass brain
        display = plotting.plot_glass_brain(
            None,
            display_mode=mode,
            title=title,
            alpha=0.6,
            black_bg=True,
            colorbar=False,
            figure=fig,
            axes=ax
        )
        
        # Prepare region display data
        for category, color in category_colors.items():
            # Get all regions in this category
            category_regions = [r for r in region_labels if r in region_categories.get(category, [])]
            
            # Get coordinates for these regions
            coords = [talairach_coordinates[r] for r in category_regions if r in talairach_coordinates]
            
            if coords:
                # Plot regions with their category color
                display.add_markers(
                    coords,
                    marker_color=color,
                    marker_size=8,
                    alpha=0.8
                )
    
    # Now create separate multi-views on the bottom row - one for each orientation
    # This prevents overlapping of the views
    for i, (mode, title) in enumerate(zip(display_modes, ['Sagittal Multi-View', 'Coronal Multi-View', 'Axial Multi-View'])):
        # Create subplot in the bottom row
        ax = plt.subplot(2, 3, i+4)  # Bottom row positions 4, 5, 6
        
        # Create glass brain
        display = plotting.plot_glass_brain(
            None,
            display_mode=mode,
            title=title,
            alpha=0.6,
            black_bg=True,
            colorbar=False,
            figure=fig,
            axes=ax
        )
        
        # Add regions by category
        for category, color in category_colors.items():
            # Get all regions in this category
            category_regions = [r for r in region_labels if r in region_categories.get(category, [])]
            
            # Get coordinates for these regions
            coords = [talairach_coordinates[r] for r in category_regions if r in talairach_coordinates]
            
            if coords:
                # Plot regions with their category color
                display.add_markers(
                    coords,
                    marker_color=color,
                    marker_size=8,
                    alpha=0.8
                )
    
    # Add a legend with category labels - SINGLE LEGEND with dots only
    legend_elements = []
    for category, color in category_colors.items():
        legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor=color, markersize=10, label=category))
    
    # Position the legend at the bottom of the figure
    legend = fig.legend(handles=legend_elements, loc='lower center', 
                      ncol=len(category_colors), fontsize=14, frameon=False,
                      bbox_to_anchor=(0.5, 0.05))
    
    for text in legend.get_texts():
        text.set_color('white')
    
    # Add region list in a separate panel - MOVED FURTHER RIGHT to avoid overlap
    # Shifted from 0.85 to 0.88 to avoid overlap with axial view
    ax_list = plt.axes([0.88, 0.3, 0.12, 0.4], frameon=True)
    ax_list.set_facecolor((0.1, 0.1, 0.1, 0.7))  # Dark semi-transparent background
    ax_list.set_axis_off()
    
    y_pos = 0.95
    ax_list.text(0.5, y_pos, "Region List", ha='center', color='white', fontsize=14, fontweight='bold')
    y_pos -= 0.05
    
    for category, color in category_colors.items():
        y_pos -= 0.03
        ax_list.text(0.5, y_pos, category, ha='center', color=color, fontsize=12, fontweight='bold')
        y_pos -= 0.02
        
        # Get regions in this category and sort them
        regions = sorted([r for r in region_categories.get(category, []) if r.endswith('-L')])
        
        # Display left and right hemisphere regions
        for region_l in regions:
            region_base = region_l[:-2]  # Remove the -L suffix
            region_r = region_base + '-R'
            
            y_pos -= 0.02
            ax_list.text(0.5, y_pos, f"{region_base} (L/R)", ha='center', color='white', fontsize=10)
    
    # Add timestamp to the figure
    fig.text(0.02, 0.02, f"Generated: {timestamp.replace('_', ' ')}", 
             color='white', fontsize=8, ha='left')
    
    # Adjust layout - UPDATED to accommodate the shifted region list
    plt.subplots_adjust(left=0.05, right=0.86, top=0.92, bottom=0.15, wspace=0.2, hspace=0.3)
    
    # Save the figure
    plt.savefig(output_path, dpi=resolution, bbox_inches='tight', facecolor=bg_color)
    print(f"Glass brain visualization saved to: {output_path}")
    
    return fig, output_path

def create_larger_brain_visualization(output_path=None):
    """Create an even larger version of the 3D brain visualization for presentation purposes"""
    # Generate timestamp for filename
    timestamp = generate_timestamp()
    
    # Create output path with timestamp
    if output_path is None:
        output_path = f"brain_regions_3d_surface_larger_{timestamp}.png"
    else:
        output_path = add_timestamp_to_filename(output_path, timestamp)
    
    return visualize_regions_on_brain_surface(
        output_path=output_path, 
        resolution=600,     # Increased resolution
        figsize=(36, 28),  # Increased figure size
        bg_color='black',
        show_legend=True
    )

def generate_all_visualizations():
    """Generate all types of brain visualizations with timestamped filenames"""
    timestamp = generate_timestamp()
    
    # Base filenames with timestamp
    surface_output = f"brain_regions_3d_surface_{timestamp}.png"
    glass_output = f"brain_regions_glass_brain_{timestamp}.png"
    larger_output = f"brain_regions_3d_surface_larger_{timestamp}.png"
    
    # Create all visualizations
    print(f"Creating a full set of brain visualizations...")
    
    print(f"1. Creating 3D brain surface visualization...")
    fig_surface, surface_path = visualize_regions_on_brain_surface(surface_output)
    plt.close(fig_surface)
    
    print(f"2. Creating glass brain visualization...")
    fig_glass, glass_path = visualize_with_glass_brain(glass_output)
    plt.close(fig_glass)
    
    print(f"3. Creating larger brain visualization for presentations...")
    fig_larger, larger_path = create_larger_brain_visualization(larger_output)
    plt.close(fig_larger)
    
    print(f"Visualization complete!")
    print(f"Files saved:")
    print(f"  - {surface_path}")
    print(f"  - {glass_path}")
    print(f"  - {larger_path}")
    
    return surface_path, glass_path, larger_path

def main():
    """Main function to create the brain visualizations"""
    # Get output type from command line argument
    if len(sys.argv) > 1:
        viz_type = sys.argv[1].lower()
        
        # Generate specific visualization based on type
        if viz_type == "surface" or viz_type == "3d":
            print(f"Creating 3D brain surface visualization...")
            fig, path = visualize_regions_on_brain_surface()
            print(f"File saved: {path}")
            
        elif viz_type == "glass":
            print(f"Creating glass brain visualization...")
            fig, path = visualize_with_glass_brain()
            print(f"File saved: {path}")
            
        elif viz_type == "large" or viz_type == "presentation":
            print(f"Creating larger brain visualization for presentations...")
            fig, path = create_larger_brain_visualization()
            print(f"File saved: {path}")
            
        else:
            # If type not recognized, generate all
            print(f"Visualization type '{viz_type}' not recognized. Generating all visualizations...")
            generate_all_visualizations()
    else:
        # No arguments provided, generate all visualizations
        generate_all_visualizations()

if __name__ == "__main__":
    main()