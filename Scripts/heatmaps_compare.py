#!/usr/bin/env python3

"""
MEG Signal Analysis - Heatmap Comparator
--------------------------------------
This script creates comparison heatmaps for connectivity matrices:
1. Overall averages (L vs R vs Difference)
2. Subject-specific averages (L vs R vs Difference)

For each combination of alignment (cue/mov), method (iDTF/gDTF/iPDC/gPDC),
and frequency (10Hz/20Hz/100Hz), the script generates three-panel comparison
plots showing left target, right target, and their difference.
"""

from pathlib import Path
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def setup_output_directory() -> Path:
    """
    Create the output directory for heatmap comparisons if it doesn't exist
    
    Returns:
        Path: Path object pointing to the output directory
    """
    output_dir = Path("Heatmaps_and_Plots/Heatmaps/Heatmaps_Comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Output directory set to: {output_dir}")
    return output_dir

def setup_normalized_output_directory() -> Path:
    """
    Create the output directory for normalized heatmap comparisons if it doesn't exist.
    This directory will contain heatmaps with consistent scales across all comparisons.
    
    Returns:
        Path: Path object pointing to the normalized output directory
    """
    output_dir = Path("Heatmaps_and_Plots/Heatmaps/Heatmaps_Comparison_Normalized")
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Normalized output directory set to: {output_dir}")
    return output_dir

class HeatmapComparator:
    """
    Class to handle the creation of comparison heatmaps for connectivity matrices.
    Generates both overall and subject-specific comparisons.
    """
    
    def __init__(self):
        """
        Initialize the comparator with necessary parameters and paths.
        Added global_max and global_diff_max to store normalization values.
        """
        self.data_dir = Path("Data")
        self.alignments = ['cue', 'mov']
        self.methods = ['iDTF', 'gDTF', 'iPDC', 'gPDC']
        self.frequencies = ['10Hz', '20Hz', '100Hz']
        self.subjects = ['DOC', 'GB', 'JDC', 'JFXD', 'JZ', 'LT', 'NvA', 'RR', 'SJB', 'BG']
        self.output_dir = setup_output_directory()
        self.normalized_output_dir = setup_normalized_output_directory()
        self.global_max = None
        self.global_diff_max = None

    def create_all_comparisons(self):
        """
        Create both overall and subject-specific comparison heatmaps.
        This is the main entry point for generating all comparisons.
        """
        logging.info("Creating overall comparison heatmaps...")
        self.create_comparison_heatmaps()
        
        logging.info("Creating subject-specific comparison heatmaps...")
        self.create_subject_comparison_heatmaps()

    def create_comparison_heatmaps(self):
        """
        Create all comparison heatmaps for overall averages.
        Generates 24 comparison plots (2 alignments × 4 methods × 3 frequencies).
        """
        for alignment in self.alignments:
            for method in self.methods:
                for freq in self.frequencies:
                    logging.info(f"Processing: {alignment}/{method}_{freq}")
                    self.create_single_comparison(alignment, method, freq)

    def create_single_comparison(self, alignment: str, method: str, freq: str):
        """
        Create a single comparison figure with three heatmaps side by side.
        
        Args:
            alignment (str): Alignment type ('cue' or 'mov')
            method (str): Analysis method ('iDTF', 'gDTF', 'iPDC', 'gPDC')
            freq (str): Frequency band ('10Hz', '20Hz', '100Hz')
        """
        try:
            # Load matrices
            left_matrix = self.load_average_matrix(alignment, 'L', method, freq)
            right_matrix = self.load_average_matrix(alignment, 'R', method, freq)
            
            if left_matrix is None or right_matrix is None:
                return

            # Calculate difference matrix
            diff_matrix = left_matrix - right_matrix

            # Set diagonals to zero
            np.fill_diagonal(left_matrix, 0)
            np.fill_diagonal(right_matrix, 0)
            np.fill_diagonal(diff_matrix, 0)

            # Create the comparison figure
            self.plot_comparison(left_matrix, right_matrix, diff_matrix,
                               alignment, method, freq)

        except Exception as e:
            logging.error(f"Error creating comparison for {alignment}_{method}_{freq}: {str(e)}")

    def load_average_matrix(self, alignment: str, target: str, method: str, freq: str) -> np.ndarray:
        """
        Load and return the average matrix for given parameters.
        
        Args:
            alignment (str): Alignment type ('cue' or 'mov')
            target (str): Target hemisphere ('L' or 'R')
            method (str): Analysis method
            freq (str): Frequency band
            
        Returns:
            np.ndarray: Loaded matrix or None if file not found
        """
        try:
            base_dir = self.data_dir / alignment / target / f"{method}_{freq}"
            filename = f"{alignment}_{target}_{method}_{freq}_average.csv"
            matrix_path = base_dir / filename
            
            if not matrix_path.exists():
                logging.warning(f"Matrix file not found: {matrix_path}")
                return None
                
            return pd.read_csv(matrix_path, header=None).values

        except Exception as e:
            logging.error(f"Error loading matrix {filename}: {str(e)}")
            return None

    def plot_comparison(self, left_matrix: np.ndarray, right_matrix: np.ndarray, 
                       diff_matrix: np.ndarray, alignment: str, method: str, freq: str):
        """
        Create and save the three-panel comparison plot.
        
        Args:
            left_matrix (np.ndarray): Matrix for left hemisphere
            right_matrix (np.ndarray): Matrix for right hemisphere
            diff_matrix (np.ndarray): Difference matrix (L - R)
            alignment (str): Alignment type
            method (str): Analysis method
            freq (str): Frequency band
        """
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))

        # Calculate common scale for L and R matrices
        vmin = min(np.min(left_matrix), np.min(right_matrix))
        vmax = max(np.max(left_matrix), np.max(right_matrix))

        # Left matrix
        sns.heatmap(left_matrix, ax=ax1, cmap='viridis', vmin=vmin, vmax=vmax)
        ax1.set_title(f'Left Target\n{alignment.upper()} {method} {freq}')
        ax1.set_xlabel('Target Region')
        ax1.set_ylabel('Source Region')

        # Right matrix
        sns.heatmap(right_matrix, ax=ax2, cmap='viridis', vmin=vmin, vmax=vmax)
        ax2.set_title(f'Right Target\n{alignment.upper()} {method} {freq}')
        ax2.set_xlabel('Target Region')
        ax2.set_ylabel('Source Region')

        # Difference matrix
        vmax_diff = max(abs(np.min(diff_matrix)), abs(np.max(diff_matrix)))
        sns.heatmap(diff_matrix, ax=ax3, cmap='RdBu_r', vmin=-vmax_diff, vmax=vmax_diff)
        ax3.set_title(f'Difference (L - R)\n{alignment.upper()} {method} {freq}')
        ax3.set_xlabel('Target Region')
        ax3.set_ylabel('Source Region')

        plt.tight_layout()
        
        # Save the figure
        output_filename = f"{alignment}_{method}_{freq}_comparison.png"
        plt.savefig(self.output_dir / output_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Created comparison heatmap: {output_filename}")

    def create_subject_comparison_heatmaps(self):
        """
        Create comparison heatmaps for each subject.
        Generates 240 comparison plots (24 combinations × 10 subjects).
        """
        for alignment in self.alignments:
            for method in self.methods:
                for freq in self.frequencies:
                    for subject in self.subjects:
                        logging.info(f"Processing subject {subject}: {alignment}/{method}_{freq}")
                        self.create_single_subject_comparison(alignment, method, freq, subject)

    def create_single_subject_comparison(self, alignment: str, method: str, freq: str, subject: str):
        """
        Create a single comparison figure for a specific subject.
        
        Args:
            alignment (str): Alignment type ('cue' or 'mov')
            method (str): Analysis method
            freq (str): Frequency band
            subject (str): Subject identifier
        """
        try:
            # Load subject-specific matrices
            left_matrix = self.load_subject_average_matrix(alignment, 'L', method, freq, subject)
            right_matrix = self.load_subject_average_matrix(alignment, 'R', method, freq, subject)
            
            if left_matrix is None or right_matrix is None:
                return

            # Calculate difference matrix
            diff_matrix = left_matrix - right_matrix

            # Set diagonals to zero
            np.fill_diagonal(left_matrix, 0)
            np.fill_diagonal(right_matrix, 0)
            np.fill_diagonal(diff_matrix, 0)

            # Create the comparison figure
            self.plot_subject_comparison(left_matrix, right_matrix, diff_matrix,
                                      alignment, method, freq, subject)

        except Exception as e:
            logging.error(f"Error creating comparison for subject {subject} {alignment}_{method}_{freq}: {str(e)}")

    def load_subject_average_matrix(self, alignment: str, target: str, method: str, freq: str, subject: str) -> np.ndarray:
        """
        Load and return the subject-specific average matrix.
        
        Args:
            alignment (str): Alignment type ('cue' or 'mov')
            target (str): Target hemisphere ('L' or 'R')
            method (str): Analysis method
            freq (str): Frequency band
            subject (str): Subject identifier
            
        Returns:
            np.ndarray: Loaded matrix or None if file not found
        """
        try:
            base_dir = self.data_dir / alignment / target / f"{method}_{freq}" / subject
            filename = f"{alignment}_{target}_{method}_{freq}_{subject}_average.csv"
            matrix_path = base_dir / filename
            
            if not matrix_path.exists():
                logging.warning(f"Subject matrix file not found: {matrix_path}")
                return None
                
            return pd.read_csv(matrix_path, header=None).values

        except Exception as e:
            logging.error(f"Error loading matrix for subject {subject}: {str(e)}")
            return None

    def plot_subject_comparison(self, left_matrix: np.ndarray, right_matrix: np.ndarray, 
                              diff_matrix: np.ndarray, alignment: str, method: str, 
                              freq: str, subject: str):
        """
        Create and save the three-panel comparison plot for a specific subject.
        
        Args:
            left_matrix (np.ndarray): Matrix for left hemisphere
            right_matrix (np.ndarray): Matrix for right hemisphere
            diff_matrix (np.ndarray): Difference matrix (L - R)
            alignment (str): Alignment type
            method (str): Analysis method
            freq (str): Frequency band
            subject (str): Subject identifier
        """
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))

        # Calculate common scale for L and R matrices
        vmin = min(np.min(left_matrix), np.min(right_matrix))
        vmax = max(np.max(left_matrix), np.max(right_matrix))

        # Left matrix
        sns.heatmap(left_matrix, ax=ax1, cmap='viridis', vmin=vmin, vmax=vmax)
        ax1.set_title(f'Subject {subject} - Left Target\n{alignment.upper()} {method} {freq}')
        ax1.set_xlabel('Target Region')
        ax1.set_ylabel('Source Region')

        # Right matrix
        sns.heatmap(right_matrix, ax=ax2, cmap='viridis', vmin=vmin, vmax=vmax)
        ax2.set_title(f'Subject {subject} - Right Target\n{alignment.upper()} {method} {freq}')
        ax2.set_xlabel('Target Region')
        ax2.set_ylabel('Source Region')

        # Difference matrix
        vmax_diff = max(abs(np.min(diff_matrix)), abs(np.max(diff_matrix)))
        sns.heatmap(diff_matrix, ax=ax3, cmap='RdBu_r', vmin=-vmax_diff, vmax=vmax_diff)
        ax3.set_title(f'Subject {subject} - Difference (L - R)\n{alignment.upper()} {method} {freq}')
        ax3.set_xlabel('Target Region')
        ax3.set_ylabel('Source Region')

        plt.tight_layout()
        
        # Save the figure
        output_filename = f"{alignment}_{method}_{freq}_{subject}_comparison.png"
        output_path = self.output_dir / output_filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Saved subject comparison heatmap to: {output_path}")

    def find_global_maxima(self):
        """
        Scan all matrices to find global maximum values for normalization.
        
        This method:
        1. Processes both overall and subject-specific matrices
        2. Ignores diagonal values
        3. Finds maximum value for L/R matrices
        4. Finds maximum absolute difference for difference matrices
        
        These values are used to create consistent scales across all heatmaps.
        """
        logging.info("Calculating global maximum values...")
        max_values = []  # Store all max values from L/R matrices
        diff_max_values = []  # Store all max absolute differences

        for alignment in self.alignments:
            for method in self.methods:
                for freq in self.frequencies:
                    # Process overall averages
                    left_matrix = self.load_average_matrix(alignment, 'L', method, freq)
                    right_matrix = self.load_average_matrix(alignment, 'R', method, freq)
                    
                    if left_matrix is not None and right_matrix is not None:
                        # Set diagonals to zero before finding max
                        np.fill_diagonal(left_matrix, 0)
                        np.fill_diagonal(right_matrix, 0)
                        max_values.extend([np.max(left_matrix), np.max(right_matrix)])
                        
                        # Calculate and store maximum absolute difference
                        diff_matrix = left_matrix - right_matrix
                        np.fill_diagonal(diff_matrix, 0)
                        diff_max_values.append(np.max(np.abs(diff_matrix)))

                    # Process subject-specific averages
                    for subject in self.subjects:
                        left_matrix = self.load_subject_average_matrix(alignment, 'L', method, freq, subject)
                        right_matrix = self.load_subject_average_matrix(alignment, 'R', method, freq, subject)
                        
                        if left_matrix is not None and right_matrix is not None:
                            np.fill_diagonal(left_matrix, 0)
                            np.fill_diagonal(right_matrix, 0)
                            max_values.extend([np.max(left_matrix), np.max(right_matrix)])
                            
                            diff_matrix = left_matrix - right_matrix
                            np.fill_diagonal(diff_matrix, 0)
                            diff_max_values.append(np.max(np.abs(diff_matrix)))

        # Store global maxima for normalization
        self.global_max = max(max_values)
        self.global_diff_max = max(diff_max_values)
        
        logging.info(f"Global maximum value for L/R matrices: {self.global_max:.4f}")
        logging.info(f"Global maximum absolute difference: {self.global_diff_max:.4f}")

    def create_normalized_comparisons(self):
        """
        Create all normalized comparison heatmaps.
        
        This method:
        1. Ensures global maxima are calculated
        2. Creates normalized versions of overall comparisons
        3. Creates normalized versions of subject-specific comparisons
        
        All heatmaps will use consistent scales for direct comparison:
        - L/R matrices: 0 to global_max
        - Difference matrices: -global_diff_max to +global_diff_max
        """
        if self.global_max is None or self.global_diff_max is None:
            self.find_global_maxima()
            
        logging.info("Creating normalized overall comparison heatmaps...")
        self.create_normalized_comparison_heatmaps()
        
        logging.info("Creating normalized subject-specific comparison heatmaps...")
        self.create_normalized_subject_comparison_heatmaps()

    def plot_normalized_comparison(self, left_matrix: np.ndarray, right_matrix: np.ndarray, 
                                 diff_matrix: np.ndarray, alignment: str, method: str, 
                                 freq: str, subject: str = None):
        """
        Create and save the normalized three-panel comparison plot.
        
        Uses consistent scales across all heatmaps:
        - L/R matrices use viridis colormap with range [0, global_max]
        - Difference matrices use RdBu_r colormap with range [-global_diff_max, +global_diff_max]
        
        Adds scale information to each subplot for reference.
        
        Args:
            left_matrix (np.ndarray): Matrix for left hemisphere
            right_matrix (np.ndarray): Matrix for right hemisphere
            diff_matrix (np.ndarray): Difference matrix (L - R)
            alignment (str): Alignment type
            method (str): Analysis method
            freq (str): Frequency band
            subject (str, optional): Subject identifier for subject-specific plots
        """
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))

        # Left matrix
        sns.heatmap(left_matrix, ax=ax1, cmap='viridis', vmin=0, vmax=self.global_max)
        title = f'Left Target\n{alignment.upper()} {method} {freq}'
        if subject:
            title = f'Subject {subject} - {title}'
        ax1.set_title(title)
        ax1.set_xlabel('Target Region')
        ax1.set_ylabel('Source Region')
        ax1.text(0.98, 1.02, f'Max: {self.global_max:.4f}', 
                transform=ax1.transAxes, ha='right')

        # Right matrix
        sns.heatmap(right_matrix, ax=ax2, cmap='viridis', vmin=0, vmax=self.global_max)
        title = f'Right Target\n{alignment.upper()} {method} {freq}'
        if subject:
            title = f'Subject {subject} - {title}'
        ax2.set_title(title)
        ax2.set_xlabel('Target Region')
        ax2.set_ylabel('Source Region')
        ax2.text(0.98, 1.02, f'Max: {self.global_max:.4f}', 
                transform=ax2.transAxes, ha='right')

        # Difference matrix
        sns.heatmap(diff_matrix, ax=ax3, cmap='RdBu_r', 
                   vmin=-self.global_diff_max, vmax=self.global_diff_max)
        title = f'Difference (L - R)\n{alignment.upper()} {method} {freq}'
        if subject:
            title = f'Subject {subject} - {title}'
        ax3.set_title(title)
        ax3.set_xlabel('Target Region')
        ax3.set_ylabel('Source Region')
        ax3.text(0.98, 1.02, f'Range: ±{self.global_diff_max:.4f}', 
                transform=ax3.transAxes, ha='right')

        plt.tight_layout()
        
        # Save the figure
        if subject:
            output_filename = f"{alignment}_{method}_{freq}_{subject}_comparison_normalized.png"
        else:
            output_filename = f"{alignment}_{method}_{freq}_comparison_normalized.png"
        
        output_path = self.normalized_output_dir / output_filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Saved normalized comparison heatmap to: {output_path}")

    def create_normalized_comparison_heatmaps(self):
        """
        Create all normalized comparison heatmaps for overall averages.
        
        Generates 24 normalized comparison plots (2 alignments × 4 methods × 3 frequencies)
        using consistent scales across all heatmaps.
        """
        for alignment in self.alignments:
            for method in self.methods:
                for freq in self.frequencies:
                    logging.info(f"Processing normalized: {alignment}/{method}_{freq}")
                    self.create_single_normalized_comparison(alignment, method, freq)

    def create_single_normalized_comparison(self, alignment: str, method: str, freq: str):
        """Create a single normalized comparison figure"""
        try:
            left_matrix = self.load_average_matrix(alignment, 'L', method, freq)
            right_matrix = self.load_average_matrix(alignment, 'R', method, freq)
            
            if left_matrix is None or right_matrix is None:
                return

            diff_matrix = left_matrix - right_matrix

            np.fill_diagonal(left_matrix, 0)
            np.fill_diagonal(right_matrix, 0)
            np.fill_diagonal(diff_matrix, 0)

            self.plot_normalized_comparison(left_matrix, right_matrix, diff_matrix,
                                         alignment, method, freq)

        except Exception as e:
            logging.error(f"Error creating normalized comparison for {alignment}_{method}_{freq}: {str(e)}")

    def create_normalized_subject_comparison_heatmaps(self):
        """
        Create normalized comparison heatmaps for each subject.
        
        Generates 240 normalized comparison plots (24 combinations × 10 subjects)
        using consistent scales across all heatmaps.
        """
        for alignment in self.alignments:
            for method in self.methods:
                for freq in self.frequencies:
                    for subject in self.subjects:
                        logging.info(f"Processing normalized subject {subject}: {alignment}/{method}_{freq}")
                        self.create_single_normalized_subject_comparison(alignment, method, freq, subject)

    def create_single_normalized_subject_comparison(self, alignment: str, method: str, 
                                                  freq: str, subject: str):
        """Create a single normalized comparison figure for a specific subject"""
        try:
            left_matrix = self.load_subject_average_matrix(alignment, 'L', method, freq, subject)
            right_matrix = self.load_subject_average_matrix(alignment, 'R', method, freq, subject)
            
            if left_matrix is None or right_matrix is None:
                return

            diff_matrix = left_matrix - right_matrix

            np.fill_diagonal(left_matrix, 0)
            np.fill_diagonal(right_matrix, 0)
            np.fill_diagonal(diff_matrix, 0)

            self.plot_normalized_comparison(left_matrix, right_matrix, diff_matrix,
                                         alignment, method, freq, subject)

        except Exception as e:
            logging.error(f"Error creating normalized comparison for subject {subject} {alignment}_{method}_{freq}: {str(e)}")

def main():
    """
    Main function to execute the script.
    Sets up logging and runs the comparison generation process.
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('heatmap_comparison.log'),
            logging.StreamHandler()
        ]
    )
    
    # Initialize and run comparator
    output_dir = setup_output_directory()
    logging.info(f"Output directory created/verified at: {output_dir.absolute()}")
    
    comparator = HeatmapComparator()
    comparator.create_all_comparisons()
    
    # Verify files were created
    png_files = list(output_dir.glob("*.png"))
    logging.info(f"Generated {len(png_files)} comparison heatmaps")
    logging.info("Comparison heatmaps generation completed")

    # Create both original and normalized comparisons
    logging.info("Creating normalized comparison heatmaps...")
    comparator.create_normalized_comparisons()
    
    # Verify files were created
    orig_files = list(output_dir.glob("*.png"))
    norm_files = list(comparator.normalized_output_dir.glob("*.png"))
    logging.info(f"Generated {len(orig_files)} original and {len(norm_files)} normalized comparison heatmaps")
    logging.info("Heatmap generation completed")

if __name__ == "__main__":
    main() 