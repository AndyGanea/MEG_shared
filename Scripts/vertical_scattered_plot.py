"""
MEG Signal Analysis - Scatter Plot Generator
-----------------------------------------
This program generates scatter plots comparing L and R target values for MEG connectivity matrices.
It processes all combinations of:
- Alignments (cue, mov)
- Methods (iDTF, gDTF, iPDC, gPDC)
- Frequencies (10Hz, 20Hz, 100Hz)

The program:
1. Reads average matrices and individual subject files
2. Extracts values from specified cell positions
3. Creates scatter plots comparing L and R values
4. Saves plots in organized folders under Data/scatter_plots/
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import random
import os
import logging
from typing import Tuple, List, Optional, Dict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scatter_plot_generator.log'),
        logging.StreamHandler()
    ]
)

class ScatterPlotAnalyzer:
    """
    Analyzes MEG connectivity matrices and generates comparative scatter plots.
    
    Attributes:
        data_dir (Path): Root directory for data files
        plots_dir (Path): Directory for output plots
        subjects (List[str]): List of subject identifiers
        alignments (List[str]): List of alignment types
        methods (List[str]): List of analysis methods
        frequencies (List[str]): List of frequencies
        cell_position (Tuple[int, int]): Target cell position in matrices
    """
    
    def __init__(self):
        """Initialize the analyzer with default parameters and create output directory."""
        try:
            self.data_dir = Path("Data")
            self.subjects = ['DOC', 'GB', 'JDC', 'JFXD', 'JZ', 'LT', 'NvA', 'RR', 'SJB', 'BG']
            self.alignments = ['cue', 'mov']
            self.methods = ['iDTF', 'gDTF', 'iPDC', 'gPDC']
            self.frequencies = ['10Hz', '20Hz', '100Hz']
            self.cell_position = (31, 31)
            
            # Create output directory
            self.plots_dir = self.data_dir / 'scatter_plots'
            self._create_output_directories()
            
        except Exception as e:
            logging.error(f"Initialization error: {str(e)}")
            raise

    def _create_output_directories(self) -> None:
        """Create necessary output directories for plots."""
        try:
            if not self.plots_dir.exists():
                self.plots_dir.mkdir(parents=True)
                logging.info(f"Created output directory: {self.plots_dir}")
            
            # Create alignment subdirectories
            for alignment in self.alignments:
                alignment_dir = self.plots_dir / alignment
                if not alignment_dir.exists():
                    alignment_dir.mkdir()
                    logging.info(f"Created alignment directory: {alignment_dir}")
                    
        except Exception as e:
            logging.error(f"Error creating directories: {str(e)}")
            raise

    def process_all_combinations(self) -> None:
        """Process all possible combinations of alignment, method, and frequency."""
        total_combinations = len(self.alignments) * len(self.methods) * len(self.frequencies)
        current_combination = 0
        
        logging.info(f"Starting processing of {total_combinations} combinations")
        
        for alignment in self.alignments:
            alignment_dir = self.plots_dir / alignment
            
            for method in self.methods:
                for freq in self.frequencies:
                    current_combination += 1
                    try:
                        logging.info(f"Processing combination {current_combination}/{total_combinations}")
                        logging.info(f"Parameters: Alignment={alignment}, Method={method}, Frequency={freq}")
                        
                        self.alignment = alignment
                        self.method = method
                        self.frequency = freq
                        
                        l_values, r_values, l_stored_avg, r_stored_avg = self.collect_subject_values()
                        
                        if l_values and r_values:
                            self.create_scatter_plot(l_values, r_values, l_stored_avg, r_stored_avg)
                        else:
                            logging.warning(f"Skipping plot creation for {alignment}_{method}_{freq} due to missing values")
                            
                    except Exception as e:
                        logging.error(f"Error processing combination {alignment}_{method}_{freq}: {str(e)}")
                        continue

    def _validate_matrix_dimensions(self, matrix: np.ndarray, filename: str) -> bool:
        """
        Validate matrix dimensions.
        
        Args:
            matrix: Input matrix to validate
            filename: Name of the file being validated
            
        Returns:
            bool: True if dimensions are valid, False otherwise
        """
        expected_shape = (32, 32)
        if matrix.shape != expected_shape:
            logging.error(f"Invalid matrix dimensions in {filename}. Expected {expected_shape}, got {matrix.shape}")
            return False
        return True

    def _load_matrix_file(self, file_path: Path) -> Optional[np.ndarray]:
        """
        Load and validate a matrix file.
        
        Args:
            file_path: Path to the matrix file
            
        Returns:
            Optional[np.ndarray]: Loaded matrix or None if error occurs
        """
        try:
            if not file_path.exists():
                logging.warning(f"File not found: {file_path}")
                return None
                
            matrix = pd.read_csv(file_path, header=None).values
            
            if self._validate_matrix_dimensions(matrix, file_path.name):
                return matrix
            return None
            
        except Exception as e:
            logging.error(f"Error loading matrix from {file_path}: {str(e)}")
            return None

    def collect_subject_values(self):
        """Collect values from subject averages for both L and R targets"""
        l_values = []
        r_values = []
        
        # First get the value from the overall average files
        l_avg_filename = f"{self.alignment}_L_{self.method}_{self.frequency}_average.csv"
        r_avg_filename = f"{self.alignment}_R_{self.method}_{self.frequency}_average.csv"
        
        # Process L average file to find maximum position (excluding diagonal)
        l_avg_file_path = self.data_dir / self.alignment / 'L' / f"{self.method}_{self.frequency}" / l_avg_filename
        r_avg_file_path = self.data_dir / self.alignment / 'R' / f"{self.method}_{self.frequency}" / r_avg_filename
        
        try:
            # Load L average matrix
            l_avg_matrix = pd.read_csv(l_avg_file_path, header=None).values
            
            # Create mask for diagonal
            mask = ~np.eye(l_avg_matrix.shape[0], dtype=bool)
            
            # Find maximum value excluding diagonal
            l_max_value = np.max(l_avg_matrix[mask])
            l_max_position = np.where((l_avg_matrix == l_max_value) & mask)
            
            # Update cell position to the new maximum (excluding diagonal)
            self.cell_position = (l_max_position[0][0] + 1, l_max_position[1][0] + 1)
            
            logging.info(f"Maximum value (excluding diagonal) in L average: {l_max_value:.6f}")
            logging.info(f"Position: {self.cell_position}")
            
            # Get stored averages at the identified position
            l_stored_avg = l_avg_matrix[self.cell_position[0]-1, self.cell_position[1]-1]
            
            # Load R average matrix and get corresponding position value
            r_avg_matrix = pd.read_csv(r_avg_file_path, header=None).values
            r_stored_avg = r_avg_matrix[self.cell_position[0]-1, self.cell_position[1]-1]
            
        except Exception as e:
            logging.error(f"Error processing overall average files: {str(e)}")
            return None, None, None, None
        
        # Collect values from subject average files for both L and R
        for subject in self.subjects:
            # L subject average file
            l_subject_avg_filename = f"{self.alignment}_L_{self.method}_{self.frequency}_{subject}_average.csv"
            l_subject_path = (self.data_dir / self.alignment / 'L' / 
                             f"{self.method}_{self.frequency}" / subject / l_subject_avg_filename)
            
            # R subject average file
            r_subject_avg_filename = f"{self.alignment}_R_{self.method}_{self.frequency}_{subject}_average.csv"
            r_subject_path = (self.data_dir / self.alignment / 'R' / 
                             f"{self.method}_{self.frequency}" / subject / r_subject_avg_filename)
            
            try:
                if l_subject_path.exists():
                    l_matrix = pd.read_csv(l_subject_path, header=None).values
                    l_value = l_matrix[self.cell_position[0]-1, self.cell_position[1]-1]
                    l_values.append((subject, l_value))  # Store subject name with value
                    logging.info(f"Subject {subject} L value: {l_value:.6f}")
                else:
                    logging.warning(f"L average file not found for subject {subject}")
                
                if r_subject_path.exists():
                    r_matrix = pd.read_csv(r_subject_path, header=None).values
                    r_value = r_matrix[self.cell_position[0]-1, self.cell_position[1]-1]
                    r_values.append((subject, r_value))  # Store subject name with value
                    logging.info(f"Subject {subject} R value: {r_value:.6f}")
                else:
                    logging.warning(f"R average file not found for subject {subject}")
                
            except Exception as e:
                logging.error(f"Error processing average files for subject {subject}: {str(e)}")
        
        # Sort values by subject name for consistent ordering
        l_values.sort(key=lambda x: x[0])
        r_values.sort(key=lambda x: x[0])
        
        # Separate subjects and values for plotting
        l_subjects, l_values = zip(*l_values) if l_values else ([], [])
        r_subjects, r_values = zip(*r_values) if r_values else ([], [])
        
        return list(l_values), list(r_values), l_stored_avg, r_stored_avg
    
    def create_scatter_plot(self, l_values, r_values, l_stored_avg, r_stored_avg):
        """Create vertical scatter plot comparing L and R values"""
        plt.figure(figsize=(12, 10))
        plt.style.use('default')
        
        # Sort values to help with label placement
        l_sorted = sorted(enumerate(l_values), key=lambda x: x[1])
        r_sorted = sorted(enumerate(r_values), key=lambda x: x[1])
        
        # Plot L values (blue) with smaller dots
        plt.scatter([1] * len(l_values), l_values, 
                   alpha=0.6, c='blue', s=50,
                   label='L Values')
        
        # Plot R values (green) with smaller dots
        plt.scatter([2] * len(r_values), r_values, 
                   alpha=0.6, c='green', s=50,
                   label='R Values')
        
        # Plot averages (red stars)
        plt.scatter(1, l_stored_avg, alpha=1.0, c='red', s=150, marker='*', label='Averages')
        plt.scatter(2, r_stored_avg, alpha=1.0, c='red', s=150, marker='*')
        
        # Add labels for L values with alternating sides based on sorted values
        for i, (orig_idx, value) in enumerate(l_sorted):
            if i % 2 == 0:  # Even indices go to the right
                x_pos = 1.05
                ha = 'left'
            else:  # Odd indices go to the left
                x_pos = 0.95
                ha = 'right'
            
            plt.text(x_pos, value, f'{self.subjects[orig_idx]}: {value:.6f}', 
                    horizontalalignment=ha,
                    verticalalignment='center',
                    fontsize=8)
        
        # Add labels for R values with alternating sides based on sorted values
        for i, (orig_idx, value) in enumerate(r_sorted):
            if i % 2 == 0:  # Even indices go to the right
                x_pos = 2.05
                ha = 'left'
            else:  # Odd indices go to the left
                x_pos = 1.95
                ha = 'right'
            
            plt.text(x_pos, value, f'{self.subjects[orig_idx]}: {value:.6f}', 
                    horizontalalignment=ha,
                    verticalalignment='center',
                    fontsize=8)
        
        # Add average labels
        plt.text(0.95, l_stored_avg, f'L Avg: {l_stored_avg:.6f}', 
                horizontalalignment='right',
                verticalalignment='center',
                color='red',
                fontweight='bold',
                fontsize=8)
        
        plt.text(1.95, r_stored_avg, f'R Avg: {r_stored_avg:.6f}', 
                horizontalalignment='right',
                verticalalignment='center',
                color='red',
                fontweight='bold',
                fontsize=8)
        
        # Convert cell position to regular integers for display
        pos_row = int(self.cell_position[0])
        pos_col = int(self.cell_position[1])
        
        # Customize plot
        plt.xlabel('Target')
        plt.ylabel('Connectivity Value')
        plt.title(f'Subject Average Values for {self.alignment.upper()}\n'
                 f'{self.method}, {self.frequency}\n'
                 f'Maximum Position (excluding diagonal): ({pos_row}, {pos_col})')
        
        plt.xticks([1, 2], ['L', 'R'])
        plt.grid(True, alpha=0.3)
        plt.xlim(0.5, 2.5)
        plt.legend()
        plt.tight_layout()
        
        # Ensure scatter_plots directory exists
        plots_dir = self.data_dir / 'scatter_plots' / self.alignment
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Save plot in the correct directory
        output_filename = f"scatter_plot_{self.alignment}_{self.method}_{self.frequency}.png"
        output_path = plots_dir / output_filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nScatter plot saved as: {output_path}")
        
        # Print statistics
        print("\nValue Statistics:")
        print(f"Number of L values: {len(l_values)}")
        print(f"Number of R values: {len(r_values)}")
        print(f"L Average (overall): {l_stored_avg:.6f}")
        print(f"R Average (overall): {r_stored_avg:.6f}")
        print(f"L Range: {min(l_values):.6f} to {max(l_values):.6f}")
        print(f"R Range: {min(r_values):.6f} to {max(r_values):.6f}")
        print(f"L Std: {np.std(l_values):.6f}")
        print(f"R Std: {np.std(r_values):.6f}")

def main():
    """Main entry point of the program."""
    logging.info("Starting Scatter Plot Generator")
    
    try:
        analyzer = ScatterPlotAnalyzer()
        
        response = input("Do you want to create all scatter plots? (y/n): ").lower()
        if response == 'y':
            analyzer.process_all_combinations()
            logging.info("All plots have been generated successfully")
            logging.info(f"Plots are saved in: {analyzer.plots_dir}")
        else:
            logging.info("Program terminated by user")
            
    except Exception as e:
        logging.error(f"Program terminated due to error: {str(e)}")
        raise

if __name__ == "__main__":
    main() 