#!/usr/bin/env python3

"""
MEG Signal Analysis - Enhanced Average Matrix Creator
-------------------------------------------------
This script creates average matrices for MEG data, with dynamic folder selection.
It processes files in the selected dataset folder and creates:
1. Overall averages for each method_freq combination
2. Subject-specific averages in subject folders

The script maintains detailed logging and processes all CSV files found in the directories.
NaN values in input matrices are replaced with 0 before averaging.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
import sys
import os

class AverageMatrixCreator:
    def __init__(self):
        """Initialize the creator with necessary paths and variables"""
        self.data_dir = Path("Data")
        self.logs_dir = Path("Logs")
        
        # Get user selection for dataset folder
        self.dataset_dir = self.select_dataset_folder()
        
        # Ask about excluding iDTF
        self.exclude_idtf = self.ask_exclude_idtf()
        
        # Initialize counters
        self.matrices_processed = 0
        self.averages_created = 0
        self.errors = 0
        self.total_nan_count = 0
        
        # Setup logging
        self.setup_logging()

    def ask_exclude_idtf(self) -> bool:
        """Ask user whether to exclude iDTF method"""
        while True:
            response = input("\nExclude iDTF method (values may be >1)? (y/n): ").lower()
            if response in ['y', 'n']:
                return response == 'y'
            print("Please enter 'y' or 'n'")

    def select_dataset_folder(self) -> Path:
        """Present available folders and let user select one"""
        print("\nAvailable datasets:")
        print("================")
        
        # Get all directories under Data that start with DataSet
        available_dirs = [d for d in self.data_dir.iterdir() 
                         if d.is_dir() and d.name.startswith("DataSet")]
        
        if not available_dirs:
            raise ValueError("No DataSet folders found in Data directory")
        
        # Present options to user
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
        log_file = self.logs_dir / f"average_matrix_creation_{timestamp}.txt"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        logging.info("MEG Signal Analysis - Enhanced Average Matrix Creation")
        logging.info("==================================================")
        logging.info(f"Selected dataset: {self.dataset_dir.name}")
        logging.info(f"Excluding iDTF: {self.exclude_idtf}")
        logging.info(f"Process started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info("--------------------------------------------------\n")

    def create_average_matrix(self, matrices):
        """Create average matrix from a list of matrices"""
        if not matrices:
            return None
        
        # Convert all matrices to numpy arrays and handle NaN values
        numpy_matrices = []
        for m in matrices:
            # Convert to numpy array
            matrix = m.values if isinstance(m, pd.DataFrame) else m
            
            # Log NaN statistics
            nan_count = np.isnan(matrix).sum()
            if nan_count > 0:
                self.total_nan_count += nan_count
                logging.info(f"Found {nan_count} NaN values in matrix - replacing with 0")
            
            # Replace NaN with 0
            matrix = np.nan_to_num(matrix, nan=0.0)
            numpy_matrices.append(matrix)
        
        # Stack matrices and calculate mean
        stacked = np.stack(numpy_matrices)
        return np.mean(stacked, axis=0)

    def save_average_matrix(self, matrix, filepath):
        """Save average matrix to CSV file"""
        # Always overwrite existing files
        pd.DataFrame(matrix).to_csv(filepath, index=False, header=False, mode='w')
        logging.info(f"Created/Overwrote average matrix: {filepath}")
        self.averages_created += 1

    def process_dataset(self):
        """Process the entire dataset structure"""
        logging.info("Processing dataset...")
        
        # Process each alignment (cue/mov)
        for align_dir in self.dataset_dir.iterdir():
            if align_dir.is_dir():
                alignment = align_dir.name  # 'cue' or 'mov'
                logging.info(f"\nProcessing alignment: {alignment}")
                
                # Get all method_freq directories across L and R
                method_freq_dirs = {}  # key: method_freq, value: list of directories
                
                # First collect all method_freq directories
                for target_dir in align_dir.iterdir():
                    if target_dir.is_dir():
                        target = target_dir.name  # 'L' or 'R'
                        logging.info(f"Processing target: {target}")
                        
                        for method_freq_dir in target_dir.iterdir():
                            if method_freq_dir.is_dir():
                                key = (target, method_freq_dir.name)  # e.g., ('R', 'gDTF_10Hz')
                                if key not in method_freq_dirs:
                                    method_freq_dirs[key] = []
                                method_freq_dirs[key].append(method_freq_dir)
                                logging.info(f"Found method_freq directory: {method_freq_dir}")
                
                # Now process each method_freq combination
                for (target, method_freq), dirs in method_freq_dirs.items():
                    method, freq = method_freq.split('_')
                    
                    # Skip iDTF if requested
                    if self.exclude_idtf and method == 'iDTF':
                        logging.info(f"Skipping iDTF method as requested")
                        continue
                    
                    logging.info(f"\nProcessing {target} {method_freq}")
                    
                    # Debug: Print all files in directory
                    for dir_path in dirs:
                        logging.info(f"Looking for files in: {dir_path}")
                        all_files = list(dir_path.glob("*.csv"))
                        logging.info(f"Found {len(all_files)} files")
                        for file in all_files:
                            logging.info(f"Found file: {file.name}")
                    
                    # Collect all files for this method_freq
                    matrices = []
                    for dir_path in dirs:
                        for file in dir_path.glob("*.csv"):
                            if 'average' not in file.name:  # Skip existing average files
                                try:
                                    matrix = pd.read_csv(file, header=None)
                                    matrices.append(matrix)
                                    self.matrices_processed += 1
                                    logging.info(f"Successfully read matrix from: {file.name}")
                                except Exception as e:
                                    self.errors += 1
                                    logging.error(f"Error reading {file}: {str(e)}")
                    
                    # Create overall average for this method_freq
                    if matrices:
                        avg_matrix = self.create_average_matrix(matrices)
                        if avg_matrix is not None:
                            # Create average filename preserving original file pattern
                            is_unc = any('_unc.csv' in str(m) for m in matrices)
                            avg_filename = f"{alignment}_{target}_{method}_{freq}_average"
                            if is_unc:
                                avg_filename += "_unc"
                            avg_filename += ".csv"
                            
                            # Save in the method_freq directory
                            for dir_path in dirs:
                                avg_filepath = dir_path / avg_filename
                                self.save_average_matrix(avg_matrix, avg_filepath)
                    else:
                        logging.warning(f"No matrices found for {target} {method_freq}")
                    
                    # Process subject folders
                    for dir_path in dirs:
                        for subject_dir in dir_path.iterdir():
                            if subject_dir.is_dir():
                                subject = subject_dir.name
                                logging.info(f"\nProcessing subject: {subject}")
                                
                                # Process subject folder
                                matrices = []
                                for file in subject_dir.glob("*.csv"):
                                    if 'average' not in file.name:
                                        try:
                                            matrix = pd.read_csv(file, header=None)
                                            matrices.append(matrix)
                                            self.matrices_processed += 1
                                            logging.info(f"Successfully read matrix from: {file.name}")
                                        except Exception as e:
                                            self.errors += 1
                                            logging.error(f"Error reading {file}: {str(e)}")
                                
                                if matrices:
                                    avg_matrix = self.create_average_matrix(matrices)
                                    if avg_matrix is not None:
                                        # Create average filename for subject level
                                        is_unc = any('_unc.csv' in str(m) for m in matrices)
                                        avg_filename = f"{alignment}_{target}_{method}_{freq}_{subject}_average"
                                        if is_unc:
                                            avg_filename += "_unc"
                                        avg_filename += ".csv"
                                        avg_filepath = subject_dir / avg_filename
                                        self.save_average_matrix(avg_matrix, avg_filepath)
                                else:
                                    logging.warning(f"No matrices found for subject {subject}")

    def print_summary(self):
        """Print summary of operations"""
        logging.info("\nOperation Summary")
        logging.info("================")
        logging.info(f"Total matrices processed: {self.matrices_processed}")
        logging.info(f"Total averages created: {self.averages_created}")
        logging.info(f"Total NaN values replaced: {self.total_nan_count}")
        logging.info(f"Errors encountered: {self.errors}")
        logging.info(f"\nProcess completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def main():
    """Main function to execute the average creation"""
    creator = AverageMatrixCreator()
    creator.process_dataset()
    creator.print_summary()

if __name__ == "__main__":
    main() 