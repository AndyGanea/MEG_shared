#!/usr/bin/env python3

"""
MEG T-test Analysis
------------------
This script performs t-tests on MEG data differences between L and R conditions.
For each cell in the matrices (excluding diagonal), it:
1. Collects values from all subjects
2. Performs t-test against zero
3. Retains or zeros out values based on significance
4. Creates detailed logs and heatmap visualizations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import logging
from datetime import datetime
import sys
from typing import List, Tuple, Dict

class MEGTtestAnalyzer:
    def __init__(self):
        """Initialize the T-test analyzer"""
        self.data_dir = Path("Data")
        self.logs_dir = Path("Logs")
        
        # Get user selection for dataset folder
        self.dataset_dir = self.select_dataset_folder()
        
        # Ask about excluding iDTF
        self.exclude_idtf = self.ask_exclude_idtf()
        
        # Create T_Test directory if it doesn't exist
        self.ttest_dir = self.dataset_dir / "T_Test"
        if not self.ttest_dir.exists():
            logging.warning(f"T_Test directory not found in {self.dataset_dir}")
            logging.info("Please run prepare_ttest_data.py first to prepare the data.")
            sys.exit(1)
        
        # Initialize counters
        self.folders_processed = 0
        self.matrices_created = 0
        self.errors = 0
        
        # Setup logging
        self.setup_logging()

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

    def ask_exclude_idtf(self) -> bool:
        """Ask user whether to exclude iDTF method"""
        while True:
            response = input("\nExclude iDTF method? (y/n): ").lower()
            if response in ['y', 'n']:
                return response == 'y'
            print("Please enter 'y' or 'n'")

    def setup_logging(self):
        """Setup logging to both file and console"""
        timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        log_file = self.logs_dir / f"meg_ttest_analysis_{timestamp}.txt"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        logging.info("MEG T-test Analysis")
        logging.info("==================")
        logging.info(f"Selected dataset: {self.dataset_dir.name}")
        logging.info(f"Excluding iDTF: {self.exclude_idtf}")
        logging.info(f"Process started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info("---------------------\n")

    def create_heatmap(self, matrix: np.ndarray, output_path: Path, title: str):
        # """Create and save heatmap visualization"""
        # plt.figure(figsize=(10, 8))
        # sns.heatmap(matrix, cmap='RdBu_r', center=0, 
        #            xticklabels=False, yticklabels=False)
        # plt.title(title)
        # plt.tight_layout()
        # plt.savefig(output_path, dpi=300, bbox_inches='tight')
        # plt.close()
        plt.figure(figsize=(12, 10))  # Slightly larger figure to accommodate labels
    
        # Create labels from 1 to 32
        labels = list(range(1, 33))
        
        # Create heatmap with numbered labels
        sns.heatmap(matrix, 
                    cmap='RdBu_r', 
                    center=0,
                    xticklabels=labels,
                    yticklabels=labels) 
        

        plt.title(title)
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=0)
        plt.yticks(rotation=0)
        
        # Add axis labels
        plt.xlabel('Region')
        plt.ylabel('Region')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    def process_folder(self, folder_path: Path):
        """Process one alignment/method/frequency folder"""
        try:
            # Extract alignment, method, frequency from folder name
            logging.info(f"Starting to process folder: {folder_path.name}")
            align, method, freq = folder_path.name.split('_')
            
            # Skip if iDTF is excluded
            if self.exclude_idtf and method == 'iDTF':
                logging.info(f"Skipping iDTF folder: {folder_path.name}")
                return
            
            logging.info(f"\nProcessing: {folder_path.name}")
            
            # Read overall matrix
            overall_files = list(folder_path.glob("*_L-R_overall.csv"))
            if not overall_files:
                raise FileNotFoundError(f"Could not find overall matrix file in {folder_path}")
            
            overall_file = overall_files[0]
            logging.info(f"Found overall matrix: {overall_file.name}")
            overall_matrix = pd.read_csv(overall_file, header=None).values
            
            # Read all subject matrices
            logging.info("Starting to read subject matrices...")
            subject_matrices = {}
            
            for subj_file in folder_path.glob("*_L-R_[A-Z]*.csv"):
                if "overall" not in subj_file.name.lower():
                    subject = subj_file.name.split('_L-R_')[1].replace('.csv', '')
                    subject_matrices[subject] = pd.read_csv(subj_file, header=None).values
                    logging.info(f"Read subject matrix: {subj_file.name}")
            
            if not subject_matrices:
                raise FileNotFoundError(f"No subject matrices found in {folder_path}")
            
            # Create output matrix (copy of overall)
            output_matrix = np.zeros_like(overall_matrix)
            matrix_size = overall_matrix.shape[0]
            
            # Create text log file
            log_file = folder_path / f"{align}_{method}_{freq}_ttest.txt"
            with open(log_file, 'w') as f:
                # Write header
                f.write(f"MEG T-test Analysis Results\n")
                f.write(f"==========================\n")
                f.write(f"Alignment: {align}\n")
                f.write(f"Method: {method}\n")
                f.write(f"Frequency: {freq}\n")
                f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write("Cell-by-Cell Analysis\n")
                f.write("--------------------\n\n")
                
                # Process each cell
                significant_cells = 0
                nonsignificant_cells = 0
                
                for i in range(matrix_size):
                    for j in range(matrix_size):
                        if i != j:  # Skip diagonal
                            # Collect values from all subjects
                            values = [subj_matrix[i,j] for subj_matrix in subject_matrices.values()]
                            
                            # Perform t-test
                            t_stat, p_value = stats.ttest_1samp(values, 0)
                            is_significant = p_value < 0.05
                            
                            # Log results
                            f.write(f"Cell ({i}, {j}):\n")
                            f.write(f"Subject Values: {values}\n")
                            f.write(f"T-statistic: {t_stat:.4f}\n")
                            f.write(f"P-value: {p_value:.4f}\n")
                            f.write(f"Result: {'Significant (p < 0.05)' if is_significant else 'Not Significant (p >= 0.05)'}\n")
                            f.write(f"Overall Matrix Value: {overall_matrix[i,j]:.4f} ")
                            f.write(f"({'Retained' if is_significant else 'Set to zero'})\n\n")
                            
                            # Update output matrix
                            output_matrix[i,j] = overall_matrix[i,j] if is_significant else 0
                            
                            # Update counters
                            if is_significant:
                                significant_cells += 1
                            else:
                                nonsignificant_cells += 1
                
                # Write summary
                f.write("Analysis Summary\n")
                f.write("--------------\n")
                f.write(f"Total Cells Analyzed: {matrix_size * matrix_size - matrix_size}\n")
                f.write(f"Significant Cells: {significant_cells}\n")
                f.write(f"Non-significant Cells: {nonsignificant_cells}\n")
                f.write(f"Diagonal Cells: {matrix_size} (set to zero)\n")
            
            # Save output matrix
            output_file = folder_path / f"{align}_{method}_{freq}_L-R_ttest.csv"
            pd.DataFrame(output_matrix).to_csv(output_file, index=False, header=False)
            
            # Create heatmap
            heatmap_file = folder_path / f"{align}_{method}_{freq}_L-R_ttest_heatmap.png"
            self.create_heatmap(output_matrix, heatmap_file, 
                              f"{align}_{method}_{freq} T-test Filtered Matrix")
            
            self.matrices_created += 1
            self.folders_processed += 1
            
        except Exception as e:
            logging.error(f"Error processing folder {folder_path.name}: {str(e)}")
            self.errors += 1

    def run_analysis(self):
        """Run the t-test analysis"""
        logging.info("Starting T-test analysis...")
        
        # Process each folder in T_Test directory
        for folder in self.ttest_dir.iterdir():
            if folder.is_dir():
                self.process_folder(folder)
        
        logging.info("\nAnalysis complete!")
        self.print_summary()

    def print_summary(self):
        """Print summary of operations"""
        logging.info("\nOperation Summary")
        logging.info("================")
        logging.info(f"Folders processed: {self.folders_processed}")
        logging.info(f"Matrices created: {self.matrices_created}")
        logging.info(f"Errors encountered: {self.errors}")
        logging.info(f"\nProcess completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def main():
    """Main execution function"""
    analyzer = MEGTtestAnalyzer()
    analyzer.run_analysis()

if __name__ == "__main__":
    main() 