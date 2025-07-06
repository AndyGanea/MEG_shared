#!/usr/bin/env python3

"""
MEG Top 10 Analysis
------------------
This script analyzes MEG (Magnetoencephalography) data to identify and visualize 
differences between left and right target measurements.

Data Structure:
/Data
    /DatasetX
        /[alignment] (cue/mov)
            /[target] (L/R)
                /[method_freq] (e.g., gDTF_10Hz)
                    - {alignment}_{target}_{method}_{freq}_average.csv
                    /[subject]
                        - {alignment}_{target}_{method}_{freq}_{subject}_average.csv

Key Features:
1. Identifies top 10 positive differences between L/R measurements
2. Creates scatter plots showing individual subject variations
3. Maintains consistent color coding for subjects across plots
4. Supports multiple methods (gDTF, iPDC, gPDC) and frequencies
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from datetime import datetime
import sys
import os
from dataclasses import dataclass
from typing import List, Dict, Set

@dataclass
class DatasetStructure:
    """
    Class to hold dataset structure information
    
    Attributes:
        alignments: Set of alignment types (e.g., {'cue', 'mov'})
        targets: Set of target directions (e.g., {'L', 'R'})
        methods: Set of analysis methods (e.g., {'gDTF', 'iPDC', 'gPDC'})
        frequencies: Set of frequencies (e.g., {'10Hz', '20Hz', '100Hz'})
        subjects: Set of subject IDs (e.g., {'BG', 'DOC', ...})
    """
    alignments: Set[str]
    targets: Set[str]
    methods: Set[str]
    frequencies: Set[str]
    subjects: Set[str]
    
    def __str__(self):
        """Provides formatted string representation of dataset structure"""
        return (f"Alignments: {sorted(self.alignments)}\n"
                f"Targets: {sorted(self.targets)}\n"
                f"Methods: {sorted(self.methods)}\n"
                f"Frequencies: {sorted(self.frequencies)}\n"
                f"Subjects: {sorted(self.subjects)}")

class MEGAnalyzer:
    """
    Base class for MEG data analysis
    
    Provides core functionality for:
    - Dataset selection and structure analysis
    - File reading and management
    - Logging setup and error handling
    """
    
    def __init__(self):
        """
        Initialize the MEG analysis framework
        - Sets up directory paths
        - Initializes logging
        - Analyzes dataset structure
        """
        self.data_dir = Path("Data")
        self.logs_dir = Path("Logs")
        
        # Get user selection for dataset folder
        self.dataset_dir = self.select_dataset_folder()
        
        # Ask about excluding iDTF
        self.exclude_idtf = self.ask_exclude_idtf()
        
        # Initialize counters for tracking progress
        self.files_processed = 0
        self.errors = 0
        
        # Setup logging
        self.setup_logging()
        
        # Analyze and store dataset structure
        self.structure = self.analyze_dataset_structure()

    def ask_exclude_idtf(self) -> bool:
        """Ask user whether to exclude iDTF method"""
        while True:
            response = input("\nExclude iDTF method? (y/n): ").lower()
            if response in ['y', 'n']:
                return response == 'y'
            print("Please enter 'y' or 'n'")

    def select_dataset_folder(self) -> Path:
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

    def setup_logging(self):
        """Setup logging to both file and console"""
        timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        log_file = self.logs_dir / f"meg_analysis_{timestamp}.txt"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        logging.info("MEG Analysis Framework")
        logging.info("=====================")
        logging.info(f"Selected dataset: {self.dataset_dir.name}")
        logging.info(f"Excluding iDTF: {self.exclude_idtf}")
        logging.info(f"Process started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info("---------------------\n")

    def analyze_dataset_structure(self) -> DatasetStructure:
        """Analyze and store the structure of the selected dataset"""
        logging.info("Analyzing dataset structure...")
        
        structure = DatasetStructure(
            alignments=set(),
            targets=set(),
            methods=set(),
            frequencies=set(),
            subjects=set()
        )
        
        try:
            # Analyze alignments (cue/mov)
            for align_dir in self.dataset_dir.iterdir():
                if align_dir.is_dir():
                    structure.alignments.add(align_dir.name)
                    
                    # Analyze targets (L/R)
                    for target_dir in align_dir.iterdir():
                        if target_dir.is_dir():
                            structure.targets.add(target_dir.name)
                            
                            # Analyze method_frequency folders
                            for method_freq_dir in target_dir.iterdir():
                                if method_freq_dir.is_dir():
                                    method, freq = method_freq_dir.name.split('_')
                                    structure.methods.add(method)
                                    structure.frequencies.add(freq)
                                    
                                    # Analyze subject folders
                                    for item in method_freq_dir.iterdir():
                                        if item.is_dir():
                                            structure.subjects.add(item.name)
            
            logging.info("\nDataset Structure:")
            logging.info(str(structure))
            return structure
            
        except Exception as e:
            logging.error(f"Error analyzing dataset structure: {str(e)}")
            self.errors += 1
            return structure

    def get_method_average(self, alignment: str, target: str, 
                          method: str, frequency: str) -> pd.DataFrame:
        """Get the method-level average file"""
        try:
            method_freq = f"{method}_{frequency}"
            avg_path = self.dataset_dir / alignment / target / method_freq
            # Look for the new average filename format
            avg_filename = f"{alignment}_{target}_{method}_{frequency}_average.csv"
            avg_file = avg_path / avg_filename
            
            if avg_file.exists():
                return self.read_meg_data(avg_file)
            return None
            
        except Exception as e:
            logging.error(f"Error getting method average: {str(e)}")
            return None

    def get_subject_average(self, alignment: str, target: str, 
                           method: str, frequency: str, subject: str) -> pd.DataFrame:
        """Get the subject-specific average file"""
        try:
            method_freq = f"{method}_{frequency}"
            subject_path = self.dataset_dir / alignment / target / method_freq / subject
            # Look for the new subject average filename format
            avg_filename = f"{alignment}_{target}_{method}_{frequency}_{subject}_average.csv"
            avg_file = subject_path / avg_filename
            
            if avg_file.exists():
                return self.read_meg_data(avg_file)
            return None
            
        except Exception as e:
            logging.error(f"Error getting subject average: {str(e)}")
            return None

    def read_meg_data(self, filepath: Path) -> pd.DataFrame:
        """Read MEG data from CSV file"""
        try:
            data = pd.read_csv(filepath, header=None)
            self.files_processed += 1
            return data
        except Exception as e:
            logging.error(f"Error reading file {filepath}: {str(e)}")
            self.errors += 1
            return None

class Top10Analyzer(MEGAnalyzer):
    """
    Specialized analyzer for identifying and visualizing top 10 differences
    between left and right target measurements
    
    Key Features:
    - Finds top 10 positive differences in group average matrices
    - Creates scatter plots showing individual subject variations
    - Maintains consistent color coding across plots
    """
    
    def __init__(self):
        """
        Initialize the Top 10 analyzer
        - Sets up output directory for plots
        - Defines color scheme for subjects
        - Initializes plot counter
        """
        super().__init__()
        
        # Create output directory under Plots folder with dataset name
        output_folder_name = f"Top10Analysis_{self.dataset_dir.name}"
        self.output_dir = Path("Plots") / output_folder_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define fixed color mapping for consistent subject identification across plots
        self.subject_colors = {
            'DOC': '#1f77b4',  # blue
            'GB': '#ff7f0e',   # orange
            'JDC': '#2ca02c',  # green
            'JFXD': '#d62728', # red
            'JZ': '#9467bd',   # purple
            'LT': '#8c564b',   # brown
            'NvA': '#e377c2',  # pink
            'RR': '#7f7f7f',   # gray
            'SJB': '#bcbd22',  # yellow-green
            'BG': '#17becf'    # cyan
        }
        
        # Track number of plots created
        self.plots_created = 0

    def get_top_10_positions(self, diff_matrix):
        """
        Identify top 10 differences by absolute value in the matrix, ignoring diagonal
        
        Args:
            diff_matrix: numpy array containing L-R difference values
            
        Returns:
            List of tuples: [((row, col), value), ...] for top 10 positions
            Values are original (signed) values, but sorted by absolute magnitude
            
        Algorithm:
        1. Create mask to ignore diagonal elements
        2. Get values and their positions from masked matrix
        3. Sort by absolute value (descending) and take top 10
        4. Return positions and original values as tuple pairs
        """
        # Create mask for diagonal (we don't want self-connections)
        mask = ~np.eye(diff_matrix.shape[0], dtype=bool)
        
        # Get values and positions from masked matrix
        masked_values = diff_matrix[mask]
        positions = np.where(mask)
        
        # Sort by absolute values (descending) and get top 10
        sorted_indices = np.argsort(np.abs(masked_values))[::-1]
        top_10_indices = sorted_indices[:10]
        
        # Get corresponding positions and original values
        top_10_positions = list(zip(
            positions[0][top_10_indices],
            positions[1][top_10_indices]
        ))
        top_10_values = masked_values[top_10_indices]
        
        return list(zip(top_10_positions, top_10_values))

    def create_scatter_plot(self, alignment, method, frequency, top_10_data, subject_values):
        """
        Create scatter plot showing subject variations for top 10 positions
        
        Args:
            alignment: 'cue' or 'mov'
            method: analysis method (e.g., 'gDTF')
            frequency: frequency value (e.g., '10Hz')
            top_10_data: list of ((row, col), value) tuples
            subject_values: dict mapping subjects to their values
            
        Creates a plot showing:
        - X-axis: Top 10 positions with their group difference values
        - Y-axis: Individual subject difference values
        - Different color for each subject
        - Grid and legend for better readability
        """
        plt.figure(figsize=(15, 8))
        
        # Create x-axis labels showing position and group difference value
        x_labels = [f"({pos[0]},{pos[1]}): {val:.4f}" 
                   for pos, val in top_10_data]
        x_positions = range(1, 11)
        
        # Plot each subject's values with consistent colors
        for subject in self.subject_colors.keys():
            if subject in subject_values:
                plt.scatter(x_positions, 
                          subject_values[subject], 
                          c=self.subject_colors[subject],
                          label=subject,
                          alpha=0.7)
        
        # Customize plot appearance
        plt.title(f"{alignment.upper()} - {method} {frequency}")
        plt.xlabel("Top 10 Positions (Row,Col): Group Difference Value")
        plt.ylabel("Subject Difference Values")
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Rotate x-axis labels for better readability
        plt.xticks(x_positions, x_labels, rotation=45, ha='right')
        
        # Adjust layout and save
        plt.tight_layout()
        filename = f"{alignment}_{method}_{frequency}_top10.png"
        plt.savefig(self.output_dir / filename, bbox_inches='tight', dpi=300)
        plt.close()
        
        self.plots_created += 1
        logging.info(f"Created plot: {filename}")

    def process_combination(self, alignment, method, frequency):
        """Process one alignment-method-frequency combination"""
        try:
            # Get group average difference matrix
            left_avg = self.get_method_average(alignment, 'L', method, frequency)
            right_avg = self.get_method_average(alignment, 'R', method, frequency)
            
            if left_avg is None or right_avg is None:
                raise ValueError("Could not find group average matrices")
            
            diff_matrix = left_avg.values - right_avg.values
            top_10_data = self.get_top_10_positions(diff_matrix)
            
            # Get subject values for these positions
            subject_values = {}
            for subject in self.subject_colors.keys():
                # Get subject averages
                left_subj = self.get_subject_average(alignment, 'L', method, frequency, subject)
                right_subj = self.get_subject_average(alignment, 'R', method, frequency, subject)
                
                if left_subj is not None and right_subj is not None:
                    diff_subj = left_subj.values - right_subj.values
                    
                    # Get values at top 10 positions
                    values = [diff_subj[pos[0], pos[1]] for pos, _ in top_10_data]
                    subject_values[subject] = values
            
            # Create plot
            self.create_scatter_plot(alignment, method, frequency, top_10_data, subject_values)
            
        except Exception as e:
            self.errors += 1
            logging.error(f"Error processing {alignment}_{method}_{frequency}: {str(e)}")

    def run_analysis(self):
        """Run the analysis for all combinations"""
        logging.info("Starting Top 10 Analysis...")
        
        for alignment in self.structure.alignments:
            for method in self.structure.methods:
                # Skip iDTF if requested
                if self.exclude_idtf and method == 'iDTF':
                    continue
                    
                for frequency in self.structure.frequencies:
                    logging.info(f"\nProcessing: {alignment}_{method}_{frequency}")
                    self.process_combination(alignment, method, frequency)
        
        logging.info(f"\nAnalysis complete. Created {self.plots_created} plots.")

    def print_summary(self):
        """Print summary of operations"""
        logging.info("\nOperation Summary")
        logging.info("================")
        logging.info(f"Files processed: {self.files_processed}")
        logging.info(f"Plots created: {self.plots_created}")
        logging.info(f"Errors encountered: {self.errors}")
        logging.info(f"\nProcess completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def main():
    """
    Main execution function
    
    Process:
    1. Initialize analyzer
    2. Run analysis for all alignment/method/frequency combinations
    3. Print summary of operations
    """
    analyzer = Top10Analyzer()
    analyzer.run_analysis()
    analyzer.print_summary()

if __name__ == "__main__":
    main() 