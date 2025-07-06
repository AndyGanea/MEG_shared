import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from mne_connectivity.viz import plot_connectivity_circle
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from tqdm import tqdm
from nilearn import plotting
from nilearn import datasets
from datetime import datetime
import glob
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

class MEGConnectivityAnalysis:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.matrix1 = None
        self.matrix2 = None
        
        # Define valid parameters
        self.valid_params = {
            'Alignment': ['cue', 'mov'],
            'Posture': ['RT_Pronation', 'RT_Down', 'RT_Upright', 'LT_Pronation'],
            'Movement': ['pro', 'anti'],
            'Target': ['L', 'R'],
            'Subject': ['DOC', 'GB', 'JDC', 'JFXD', 'JZ', 'LT', 'NvA', 'RR', 'SJB', 'BG'],
            'Method': ['iDTF', 'gDTF', 'iPDC', 'gPDC'],
            'Frequency': ['10Hz', '20Hz', '100Hz']
        }
        
        # Define coordinates for our 32 regions
        self.region_coords = {
            # Visual (Purple)
            'V1L': [-10, -85, 5], 'V1R': [10, -85, 5],
            'V2L': [-15, -95, 5], 'V2R': [15, -95, 5],
            'V4L': [-30, -80, 5], 'V4R': [30, -80, 5],
            'VOTL': [-45, -70, 5], 'VOTR': [45, -70, 5],
            
            # Parietal (Blue)
            'SPL1L': [-15, -55, 65], 'SPL1R': [15, -55, 65],
            'SPL2L': [-30, -50, 65], 'SPL2R': [30, -50, 65],
            'IPL1L': [-40, -45, 45], 'IPL1R': [40, -45, 45],
            'IPL2L': [-50, -40, 45], 'IPL2R': [50, -40, 45],
            
            # Temporal (Green)
            'STG1L': [-60, -25, 5], 'STG1R': [60, -25, 5],
            'STG2L': [-65, -35, 15], 'STG2R': [65, -35, 15],
            
            # Sensorimotor (Red)
            'S1L': [-30, -30, 65], 'S1R': [30, -30, 65],
            'S2L': [-40, -25, 65], 'S2R': [40, -25, 65],
            'M1L': [-30, -20, 65], 'M1R': [30, -20, 65],
            'M2L': [-40, -15, 65], 'M2R': [40, -15, 65],
            
            # Premotor/Motor (Orange)
            'PM1L': [-30, -10, 65], 'PM1R': [30, -10, 65],
            'PM2L': [-40, -5, 65], 'PM2R': [40, -5, 65]
        }
        
        # Define colors for different brain regions
        self.region_colors = []
        for region in self.region_coords.keys():
            if region.startswith('V'):  # Visual
                self.region_colors.append('#A020F0')  # Purple
            elif region.startswith(('SPL', 'IPL')):  # Parietal
                self.region_colors.append('#0000FF')  # Blue
            elif region.startswith('STG'):  # Temporal
                self.region_colors.append('#00FF00')  # Green
            elif region.startswith(('S', 'M1', 'M2')):  # Sensorimotor
                self.region_colors.append('#FF0000')  # Red
            elif region.startswith('PM'):  # Premotor
                self.region_colors.append('#FFA500')  # Orange
        
        self.n_regions = 32
        
        # Define brain regions
        left_regions = [
            'V1-L', 'V3-L', 'SPOC-L', 'AG-L', 'POJ-L', 'SPL-L', 'mIPS-L', 'VIP-L',
            'IPL-L', 'STS-L', 'S1-L', 'M1-L', 'SMA-L', 'PMd-L', 'FEF-L', 'PMv-L'
        ]
        
        # Create right hemisphere regions by replacing '-L' with '-R'
        right_regions = [region.replace('-L', '-R') for region in left_regions]
        
        # Combine into full list - first left hemisphere, then right hemisphere
        self.brain_regions = left_regions + right_regions
        
    def load_two_matrices(self, cue_file, mov_file):
        """Load two connectivity matrices from CSV files"""
        try:
            # Store the filenames
            self.cue_file = cue_file
            self.mov_file = mov_file
            
            # Load the matrices
            self.matrix1 = np.loadtxt(cue_file, delimiter=',')
            self.matrix2 = np.loadtxt(mov_file, delimiter=',')
            
            print(f"Loaded matrices from:\n{os.path.basename(cue_file)}\n{os.path.basename(mov_file)}")
            
        except Exception as e:
            print(f"Error loading matrices: {str(e)}")
            raise

    def generate_unique_gif_filename(self, base_name='connectivity_transition'):
        """
        Generate a unique filename for the GIF with timestamp
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Get list of existing files with similar names
        existing_files = glob.glob(f"{base_name}_*_*.gif")
        
        # Find the highest n value
        n = 1
        if existing_files:
            for file in existing_files:
                try:
                    n_str = file.split('_')[2]
                    current_n = int(n_str)
                    n = max(n, current_n + 1)
                except (IndexError, ValueError):
                    continue
        
        return f"{base_name}_{n}_{timestamp}.gif"

    def create_transition_animation(self, n_frames=50, threshold_percentile=90):
        """
        Create animation showing transition between matrices
        """
        if self.matrix1 is None or self.matrix2 is None:
            raise ValueError("Please load both matrices first")
            
        # Create figure with extra space for labels and legend
        fig = plt.figure(figsize=(20, 12))
        
        # Calculate threshold (using both matrices)
        all_values = np.concatenate([self.matrix1.flatten(), self.matrix2.flatten()])
        threshold = np.percentile(all_values, threshold_percentile)
        
        # Initialize progress bar
        progress_bar = tqdm(total=n_frames, desc="Creating animation")
        
        def update(frame):
            plt.clf()
            
            # Calculate interpolated matrix
            t = frame / (n_frames - 1)
            current_matrix = (1 - t) * self.matrix1 + t * self.matrix2
            
            # Apply threshold
            current_matrix[current_matrix < threshold] = 0
            
            # Update title based on transition
            title = f"Transition from Cue to Movement: {t*100:.1f}%"
            
            # Plot current state
            plot_connectivity_circle(
                current_matrix,
                self.brain_regions,
                node_colors=self.region_colors,
                title=title,
                show=False,
                vmin=threshold,
                vmax=max(np.max(self.matrix1), np.max(self.matrix2)),
                colormap='hot',
                colorbar=True,
                fig=fig,
                subplot=111,
                fontsize_names=14,
                textcolor='white',
                node_width=3,
                padding=6,
                fontsize_colorbar=10,
                node_linewidth=2,
                node_edgecolor='white'
            )
            
            # Add legend
            legend_elements = [
                plt.Line2D([0], [0], color='#A020F0', label='Visual Areas'),
                plt.Line2D([0], [0], color='#0000FF', label='Parietal Areas'),
                plt.Line2D([0], [0], color='#00FF00', label='Temporal Areas'),
                plt.Line2D([0], [0], color='#FF0000', label='Sensorimotor Areas'),
                plt.Line2D([0], [0], color='#FFA500', label='Premotor/Motor Areas')
            ]
            plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.0, 0.5))
            
            # Update progress bar
            progress_bar.update(1)
            
            return fig,
        
        # Create animation
        anim = FuncAnimation(
            fig, 
            update, 
            frames=n_frames,
            interval=100,
            blit=False
        )
        
        # Generate unique filename
        filename = self.generate_unique_gif_filename()
        
        # Save animation
        print("\nSaving animation...")
        anim.save(filename, writer='pillow', dpi=200)
        plt.close()
        
        # Close progress bar
        progress_bar.close()
        
        print(f"\nAnimation has been saved as '{filename}'")

    def plot_brain_connectivity(self, matrix, title=None, display_mode='ortho', threshold_percentile=40):
        """
        Plot connectivity values on a brain surface using nilearn
        """
        # Create main figure with adjusted size to accommodate legend
        fig = plt.figure(figsize=(15, 10))
        
        # Create brain plot in the main area
        ax_brain = plt.subplot2grid((4, 1), (0, 0), rowspan=3)
        
        # Convert coordinates and node sizes to lists
        coords = list(self.region_coords.values())
        node_sizes = np.sum(matrix, axis=0)
        node_sizes = 2 + (node_sizes - np.min(node_sizes)) / (np.max(node_sizes) - np.min(node_sizes)) * 8
        
        # Calculate value ranges for distinct colors
        non_zero_vals = matrix[matrix > 0]
        min_val = np.min(non_zero_vals)
        max_val = np.max(non_zero_vals)
        
        # Set threshold based on parameter
        threshold = np.percentile(non_zero_vals, threshold_percentile)
        
        # Create custom colormap with red progression
        colors = [
            (1, 0.9, 0.9),    # Very light red (weakest)
            (1, 0.7, 0.7),    # Light red
            (1, 0.4, 0.4),    # Medium red
            (0.9, 0.1, 0.1),  # Dark red
            (0.7, 0, 0)       # Very dark red (strongest)
        ]
        
        # Create a colormap with smooth transitions
        custom_cmap = LinearSegmentedColormap.from_list('custom_reds', colors)
        
        # Create connectome plot with refined visibility
        display = plotting.plot_connectome(
            matrix,
            coords,
            node_size=node_sizes,
            node_color=self.region_colors,
            title=title,
            display_mode=display_mode,
            colorbar=True,
            node_kwargs={'alpha': 0.8},
            edge_kwargs={
                'alpha': 0.8,           
                'linewidth': 0.2        # Even thinner lines
            },
            edge_threshold=threshold,    
            edge_cmap=custom_cmap,      # Red color progression
            edge_vmin=min_val,          
            edge_vmax=max_val,          
            axes=ax_brain
        )
        
        # Add legend at the bottom of the same figure
        ax_legend = plt.subplot2grid((4, 1), (3, 0))
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor=color, label=label, markersize=10)
            for color, label in zip(
                ['#A020F0', '#0000FF', '#00FF00', '#FF0000', '#FFA500'],
                ['Visual', 'Parietal', 'Temporal', 'Sensorimotor', 'Premotor/Motor']
            )
        ]
        
        ax_legend.legend(handles=legend_elements, loc='center', ncol=5)
        ax_legend.axis('off')
        
        # Adjust layout
        plt.tight_layout()
        
        # Show the figure
        plt.show()
        
        return display

    def parse_filename_parameters(self, filename):
        """Parse parameters from filename"""
        # Remove .csv and get just the filename
        basename = os.path.basename(filename).replace('.csv', '')
        parts = basename.split('_')
        
        params = {}
        
        try:
            # For filename like: BG_RT_Down_anti_L_gDTF_100Hz.csv
            current_idx = 0
            
            # First part is Subject (BG)
            params['Subject'] = 'BG'  # Always BG
            current_idx += 1
            
            # Handle Posture (RT_Down)
            params['Posture'] = 'RT-Down'  # Always RT-Down with hyphen
            current_idx += 2  # Skip RT and Down parts
            
            # Get Movement (anti)
            params['Movement'] = 'anti'  # Always anti
            current_idx += 1
            
            # Get Target (L)
            params['Target'] = 'L'  # Always L
            current_idx += 1
            
            # Get Method (gDTF)
            params['Method'] = 'gDTF'  # Always gDTF
            current_idx += 1
            
            # Get Frequency (100Hz)
            params['Frequency'] = '100Hz'  # Always 100Hz
            
        except Exception as e:
            print(f"Error parsing filename {basename}: {str(e)}")
            return {
                'Subject': 'Unknown',
                'Posture': 'Unknown',
                'Movement': 'Unknown',
                'Target': 'Unknown',
                'Method': 'Unknown',
                'Frequency': 'Unknown'
            }
        
        return params

    def plot_side_by_side_comparison(self, threshold_percentile=40):
        """
        Create a single figure with side-by-side comparison of cue and movement phases
        """
        if self.matrix1 is None or self.matrix2 is None:
            raise ValueError("Please load both matrices first")
        
        # Parse parameters from filenames
        params = self.parse_filename_parameters(self.cue_file)
        
        # Create title with parameters in exact order
        title_params = (
            f"Subject: {params['Subject']}, "  # Should show BG
            f"Posture: {params['Posture']}, "  # Should show RT-Down
            f"Movement: {params['Movement']}, "  # Should show anti
            f"Target: {params['Target']}\n"  # Should show L
            f"Method: {params['Method']}, "  # Should show gDTF
            f"Frequency: {params['Frequency']}"  # Should show 100Hz
        )
        
        # Create a wide figure to accommodate both plots
        fig = plt.figure(figsize=(20, 10))
        
        # Add main title
        fig.suptitle(title_params, fontsize=12, y=0.98)
        
        # Create subplot for Cue phase (left side)
        ax_cue = plt.subplot2grid((4, 2), (0, 0), rowspan=3)
        
        # Create subplot for Movement phase (right side)
        ax_mov = plt.subplot2grid((4, 2), (0, 1), rowspan=3)
        
        # Calculate value ranges and thresholds for consistent visualization
        non_zero_vals = np.concatenate([self.matrix1[self.matrix1 > 0], 
                                      self.matrix2[self.matrix2 > 0]])
        min_val = np.min(non_zero_vals)
        max_val = np.max(non_zero_vals)
        threshold = np.percentile(non_zero_vals, threshold_percentile)
        
        # Create custom colormap with red progression
        colors = [
            (1, 0.9, 0.9),    # Very light red (weakest)
            (1, 0.7, 0.7),    # Light red
            (1, 0.4, 0.4),    # Medium red
            (0.9, 0.1, 0.1),  # Dark red
            (0.7, 0, 0)       # Very dark red (strongest)
        ]
        custom_cmap = LinearSegmentedColormap.from_list('custom_reds', colors)
        
        # Convert coordinates to list
        coords = list(self.region_coords.values())
        
        # Plot Cue phase
        display_cue = plotting.plot_connectome(
            self.matrix1,
            coords,
            node_color=self.region_colors,
            title="Cue Phase",
            display_mode='ortho',
            node_kwargs={'alpha': 0.8},
            edge_kwargs={'alpha': 0.8, 'linewidth': 0.2},
            edge_threshold=threshold,
            edge_cmap=custom_cmap,
            edge_vmin=min_val,
            edge_vmax=max_val,
            axes=ax_cue
        )
        
        # Plot Movement phase
        display_mov = plotting.plot_connectome(
            self.matrix2,
            coords,
            node_color=self.region_colors,
            title="Movement Phase",
            display_mode='ortho',
            node_kwargs={'alpha': 0.8},
            edge_kwargs={'alpha': 0.8, 'linewidth': 0.2},
            edge_threshold=threshold,
            edge_cmap=custom_cmap,
            edge_vmin=min_val,
            edge_vmax=max_val,
            axes=ax_mov
        )
        
        # Add shared legend at the bottom
        ax_legend = plt.subplot2grid((4, 2), (3, 0), colspan=2)
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor=color, label=label, markersize=10)
            for color, label in zip(
                ['#A020F0', '#0000FF', '#00FF00', '#FF0000', '#FFA500'],
                ['Visual', 'Parietal', 'Temporal', 'Sensorimotor', 'Premotor/Motor']
            )
        ]
        ax_legend.legend(handles=legend_elements, loc='center', ncol=5)
        ax_legend.axis('off')
        
        # Adjust layout
        plt.tight_layout()
        plt.show()

    def plot_connectivity_comparison(self, threshold_percentile=40):
        """
        Plot both cue and movement connectivity on brain surfaces
        """
        if self.matrix1 is None or self.matrix2 is None:
            raise ValueError("Please load both matrices first")
        
        print("Plotting Cue Phase Connectivity...")
        display_cue = self.plot_brain_connectivity(
            self.matrix1, 
            "Cue Phase Connectivity",
            threshold_percentile=threshold_percentile
        )
        
        print("Plotting Movement Phase Connectivity...")
        display_mov = self.plot_brain_connectivity(
            self.matrix2, 
            "Movement Phase Connectivity",
            threshold_percentile=threshold_percentile
        )
        
        print("Plotting Side-by-Side Comparison...")
        self.plot_side_by_side_comparison(threshold_percentile=threshold_percentile)
        
        print("Plots completed. Interactive windows should now be visible.")

def main():
    # Initialize MEG analysis
    data_dir = r".\Data"  # Base data folder
    meg = MEGConnectivityAnalysis(data_dir)
    
    # List all CSV files in the data folder
    print("\nLooking for data files...")
    available_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.csv'):
                available_files.append(os.path.join(root, file))
    
    if not available_files:
        print(f"No CSV files found in {data_dir}")
        return
    
    # Print available files
    print("\nAvailable files:")
    for i, file in enumerate(available_files):
        print(f"{i+1}. {os.path.basename(file)}")
    
    # Let user select files
    try:
        print("\nPlease select file numbers:")
        cue_idx = int(input("Enter number for cue file: ")) - 1
        mov_idx = int(input("Enter number for movement file: ")) - 1
        
        if 0 <= cue_idx < len(available_files) and 0 <= mov_idx < len(available_files):
            cue_file = available_files[cue_idx]
            mov_file = available_files[mov_idx]
            
            print(f"\nLoading files:")
            print(f"Cue: {os.path.basename(cue_file)}")
            print(f"Movement: {os.path.basename(mov_file)}")
            
            # Load the matrices
            meg.load_two_matrices(cue_file, mov_file)
            
            # Create GIF animation
            print("\nCreating GIF animation...")
            meg.create_transition_animation()
            
            # Or view brain connectivity
            print("\nViewing brain connectivity...")
            meg.plot_connectivity_comparison()
            
        else:
            print("Invalid file selection")
            
    except ValueError:
        print("Please enter valid numbers")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()