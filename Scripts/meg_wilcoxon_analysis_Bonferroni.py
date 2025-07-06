#!/usr/bin/env python3

"""
MEG Wilcoxon Analysis with Bonferroni Correction
------------------
This script performs Wilcoxon signed-rank tests on MEG data differences between L and R conditions,
applying Bonferroni correction for multiple comparisons.
For each cell in the matrices, it:
1. Collects values from all subjects
2. Performs Wilcoxon test against zero
3. Applies Bonferroni correction to p-values
4. Retains or zeros out values based on corrected significance
5. Creates detailed logs and heatmap visualizations
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
import shutil
from typing import List, Tuple, Dict

class MEGWilcoxonBonferroniAnalyzer:
    def __init__(self):
        """Initialize the Wilcoxon Bonferroni analyzer"""
        self.data_dir = Path("Data")
        self.logs_dir = Path("Logs")
        
        # Get user selection for dataset folder
        self.dataset_dir = self.select_dataset_folder()
        
        # Ask about excluding iDTF
        self.exclude_idtf = self.ask_exclude_idtf()
        
        # Ask for Bonferroni divisor
        self.bonferroni_divisor = self.ask_bonferroni_divisor()
        
        # Bonferroni correction parameters
        self.alpha = 0.05
        self.bonferroni_threshold = self.alpha / self.bonferroni_divisor
        
        # Create timestamp for directory naming
        self.timestamp = datetime.now().strftime("%m%d%y-%H%M")
        
        # Create or clean Wilcoxon_Bonferroni directory with timestamp and divisor
        self.wilcoxon_dir = self.dataset_dir / f"Wilcoxon_Bonferroni_p={self.bonferroni_divisor}_{self.timestamp}"
        if self.wilcoxon_dir.exists():
            shutil.rmtree(self.wilcoxon_dir)
            logging.info(f"Removed existing directory: {self.wilcoxon_dir.name}")
        
        self.wilcoxon_dir.mkdir(parents=True)
        logging.info(f"Created directory: {self.wilcoxon_dir.name}")
        
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
    
    def ask_bonferroni_divisor(self) -> int:
        """Ask user for Bonferroni divisor (number of tests)"""
        print("\nEnter Bonferroni divisor (integer number of tests):")
        attempts = 0
        max_attempts = 5
        
        while attempts < max_attempts:
            try:
                divisor = int(input("Divisor: "))
                if divisor <= 0:
                    print("Please enter a positive integer.")
                    attempts += 1
                else:
                    return divisor
            except ValueError:
                print(f"Please enter a valid integer. {max_attempts - attempts - 1} attempts remaining.")
                attempts += 1
        
        print(f"Maximum attempts ({max_attempts}) reached. Exiting program.")
        sys.exit(1)

    def setup_logging(self):
        """Setup logging to both file and console"""
        timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        log_file = self.logs_dir / f"meg_wilcoxon_bonferroni_analysis_{timestamp}.txt"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        logging.info("MEG Wilcoxon Analysis with Bonferroni Correction")
        logging.info("============================================")
        logging.info(f"Selected dataset: {self.dataset_dir.name}")
        logging.info(f"Excluding iDTF: {self.exclude_idtf}")
        logging.info(f"Bonferroni divisor: {self.bonferroni_divisor}")
        logging.info(f"Bonferroni threshold: {self.bonferroni_threshold:.8f}")
        logging.info(f"Process started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info("---------------------\n")

    def prepare_wilcoxon_data(self):
        """Prepare data structure and L-R files for Wilcoxon analysis"""
        logging.info("Preparing data for Wilcoxon analysis...")
        logging.info(f"Looking in dataset directory: {self.dataset_dir}")
        
        # Check for available condition folders (mov and/or cue)
        conditions = []
        if (self.dataset_dir / "Mov").exists():
            conditions.append(("Mov", "mov"))
        if (self.dataset_dir / "Cue").exists():
            conditions.append(("Cue", "cue"))
        
        if not conditions:
            logging.error("No Mov or Cue directories found!")
            return
        
        logging.info(f"Found conditions: {[c[0] for c in conditions]}")
        
        # Get all method-frequency combinations
        method_freq_dirs = []
        for method in ['gDTF', 'iPDC', 'gPDC', 'iDTF']:
            if self.exclude_idtf and method == 'iDTF':
                continue
            for freq in ['10Hz', '20Hz', '100Hz']:
                method_freq_dirs.append(f"{method}_{freq}")
        
        # Process each condition (Mov and/or Cue)
        for condition_dir, prefix in conditions:
            logging.info(f"\nProcessing condition: {condition_dir}")
            
            # Create directories and process files
            for method_freq in method_freq_dirs:
                try:
                    # Create method_freq directory with condition prefix
                    method_freq_dir = self.wilcoxon_dir / f"{condition_dir}_{method_freq}"
                    method_freq_dir.mkdir(parents=True, exist_ok=True)
                    logging.info(f"\nProcessing {method_freq}")
                    
                    # Define paths for L and R
                    l_base_dir = self.dataset_dir / condition_dir / "L" / method_freq
                    r_base_dir = self.dataset_dir / condition_dir / "R" / method_freq
                    
                    logging.info(f"Looking in directories:")
                    logging.info(f"L base directory: {l_base_dir}")
                    logging.info(f"R base directory: {r_base_dir}")
                    
                    if not l_base_dir.exists() or not r_base_dir.exists():
                        logging.warning(f"Skipping {method_freq} - directories not found")
                        continue
                    
                    # First, get list of subject folders
                    subject_folders = [d.name for d in l_base_dir.iterdir() if d.is_dir() 
                                     and d.name != self.dataset_dir.name]
                    
                    logging.info(f"\nFound subject folders: {subject_folders}")
                    
                    # Process each subject
                    all_differences = []
                    for subject in subject_folders:
                        # Look for files with both movement and cue patterns
                        l_patterns = [
                            f"{prefix}_L_{method_freq}_{subject}_average.csv",
                            f"{prefix}_L_{method_freq}_average.csv"
                        ]
                        r_patterns = [
                            f"{prefix}_R_{method_freq}_{subject}_average.csv",
                            f"{prefix}_R_{method_freq}_average.csv"
                        ]
                        
                        l_file = None
                        r_file = None
                        
                        # Try each pattern
                        for l_pattern in l_patterns:
                            found = list(l_base_dir.joinpath(subject).glob(l_pattern))
                            if found:
                                l_file = found[0]
                                break
                        
                        for r_pattern in r_patterns:
                            found = list(r_base_dir.joinpath(subject).glob(r_pattern))
                            if found:
                                r_file = found[0]
                                break
                        
                        if l_file and r_file:
                            logging.info(f"\nProcessing subject {subject}:")
                            logging.info(f"L file: {l_file}")
                            logging.info(f"R file: {r_file}")
                            
                            # Read matrices
                            l_matrix = pd.read_csv(l_file, header=None).values
                            r_matrix = pd.read_csv(r_file, header=None).values
                            
                            # Calculate difference
                            diff_matrix = l_matrix - r_matrix
                            all_differences.append(diff_matrix)
                            
                            # Save subject difference
                            diff_file = method_freq_dir / f"{method_freq}_L-R_{subject}.csv"
                            pd.DataFrame(diff_matrix).to_csv(diff_file, index=False, header=False)
                            logging.info(f"Created L-R difference file for subject {subject}")
                        else:
                            logging.warning(f"Missing files for subject {subject}")
                    
                    logging.info(f"\nTotal subjects processed: {len(all_differences)}")
                    
                    # Calculate and save overall difference
                    if all_differences:
                        overall_diff = np.mean(all_differences, axis=0)
                        overall_file = method_freq_dir / f"{method_freq}_L-R_overall.csv"
                        pd.DataFrame(overall_diff).to_csv(overall_file, index=False, header=False)
                        logging.info(f"Created overall L-R difference file")
                    else:
                        logging.error(f"No differences calculated for {method_freq}")
                        
                except Exception as e:
                    logging.error(f"Error processing {method_freq}: {str(e)}")
                    self.errors += 1

    def create_heatmap(self, matrix: np.ndarray, output_path: Path, title: str):
        """Create and save heatmap visualization"""
        plt.figure(figsize=(12, 10))
        
        # Create labels from 1 to 32
        labels = list(range(1, 33))
        
        # Create heatmap with numbered labels
        sns.heatmap(matrix, 
                    cmap='RdBu_r', 
                    center=0,
                    xticklabels=labels,
                    yticklabels=labels)
        
        plt.title(title)
        plt.xticks(rotation=0)
        plt.yticks(rotation=0)
        plt.xlabel('Region')
        plt.ylabel('Region')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    def process_folder(self, folder_path: Path):
        """Process one alignment/method/frequency folder with Bonferroni correction"""
        try:
            # Extract method_freq and condition from folder name
            folder_parts = folder_path.name.split('_')
            condition = folder_parts[0].lower()  # 'Mov' or 'Cue'
            method_freq = '_'.join(folder_parts[1:])  # 'gDTF_10Hz'
            
            if self.exclude_idtf and method_freq.startswith('iDTF'):
                logging.info(f"Skipping iDTF folder: {folder_path.name}")
                return
            
            logging.info(f"\nProcessing: {folder_path.name}")
            
            # Read all subject matrices
            logging.info("Starting to read subject matrices...")
            subject_matrices = {}
            
            # Look for all subject L-R files (should be 10 of them)
            subject_files = list(folder_path.glob("*_L-R_[A-Z]*.csv"))
            subject_files = [f for f in subject_files if "overall" not in f.name.lower()]
            
            if len(subject_files) != 10:
                logging.warning(f"Found {len(subject_files)} subject files (expected 10)")
            
            for subj_file in subject_files:
                subject = subj_file.name.split('_L-R_')[1].replace('.csv', '')
                subject_matrices[subject] = pd.read_csv(subj_file, header=None).values
                logging.info(f"Read subject matrix: {subj_file.name}")
            
            # Try both patterns for overall matrix
            overall_patterns = [
                f"{condition}_L_{method_freq}_average.csv",
                f"{method_freq}_L-R_overall.csv"
            ]
            
            overall_file = None
            for pattern in overall_patterns:
                potential_file = folder_path / pattern
                if potential_file.exists():
                    overall_file = potential_file
                    logging.info(f"Found overall matrix file: {pattern}")
                    break
            
            if not overall_file:
                raise FileNotFoundError(f"Could not find overall matrix file with patterns: {overall_patterns}")
            
            overall_matrix = pd.read_csv(overall_file, header=None).values
            matrix_size = overall_matrix.shape[0]
            output_matrix = np.zeros_like(overall_matrix)
            
            # Create text log file with Bonferroni details
            log_file = folder_path / f"{method_freq}_wilcoxon_Bonferroni.txt"
            with open(log_file, 'w') as f:
                f.write(f"MEG Wilcoxon Analysis Results with Bonferroni Correction\n")
                f.write(f"============================================\n")
                f.write(f"Method-Frequency: {method_freq}\n")
                f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Bonferroni divisor: {self.bonferroni_divisor}\n")
                f.write(f"Bonferroni threshold: {self.bonferroni_threshold:.8f}\n\n")
                f.write("Cell-by-Cell Analysis\n")
                f.write("--------------------\n\n")
                
                significant_cells = 0
                nonsignificant_cells = 0
                significant_positions = []  # Track positions of significant cells
                
                for i in range(matrix_size):
                    for j in range(matrix_size):
                        # Collect values from all subjects for this cell
                        values = []
                        for subject, matrix in subject_matrices.items():
                            values.append(matrix[i,j])
                        
                        if len(values) != 10:
                            logging.warning(f"Cell ({i},{j}): Found {len(values)} values (expected 10)")
                        
                        # Perform Wilcoxon test against zero
                        w_stat, p_value = stats.wilcoxon(values, alternative='two-sided')
                        is_significant = p_value < self.bonferroni_threshold
                        
                        # Log results
                        f.write(f"Cell ({i}, {j}):\n")
                        f.write(f"Subject Values: {values}\n")
                        f.write(f"W-statistic: {w_stat:.4f}\n")
                        f.write(f"P-value: {p_value:.8f}\n")
                        f.write(f"Bonferroni threshold: {self.bonferroni_threshold:.8f}\n")
                        f.write(f"Result: {'Significant' if is_significant else 'Not Significant'}\n")
                        f.write(f"Overall Matrix Value: {overall_matrix[i,j]:.4f} ")
                        f.write(f"({'Retained' if is_significant else 'Set to zero'})\n\n")
                        
                        # Update output matrix
                        output_matrix[i,j] = overall_matrix[i,j] if is_significant else 0
                        
                        if is_significant:
                            significant_cells += 1
                            significant_positions.append((i, j, overall_matrix[i,j]))
                        else:
                            nonsignificant_cells += 1
                
                # Write summary
                f.write("\nAnalysis Summary\n")
                f.write("--------------\n")
                f.write(f"Total Cells Analyzed: {significant_cells + nonsignificant_cells}\n")
                f.write(f"Significant Cells (p < {self.bonferroni_threshold:.8f}): {significant_cells}\n")
                f.write(f"Non-significant Cells: {nonsignificant_cells}\n\n")
                
                # Write significant positions
                f.write("Significant Cell Positions:\n")
                f.write("------------------------\n")
                if significant_positions:
                    # Sort by absolute value (descending)
                    significant_positions.sort(key=lambda x: abs(x[2]), reverse=True)
                    for pos in significant_positions:
                        f.write(f"Position ({pos[0]}, {pos[1]}): Value = {pos[2]:.4f}\n")
                else:
                    f.write("No significant cells found.\n")
            
            # Save output matrix
            output_file = folder_path / f"{method_freq}_L-R_wilcoxon_Bonferroni.csv"
            pd.DataFrame(output_matrix).to_csv(output_file, index=False, header=False)
            
            # Create heatmap
            heatmap_file = folder_path / f"{method_freq}_L-R_wilcoxon_Bonferroni_heatmap.png"
            self.create_heatmap(output_matrix, heatmap_file, 
                              f"{method_freq} Wilcoxon Bonferroni Filtered Matrix")
            
            self.matrices_created += 1
            self.folders_processed += 1
            
        except Exception as e:
            logging.error(f"Error processing folder {folder_path.name}: {str(e)}")
            self.errors += 1

    def run_analysis(self):
        """Run the Wilcoxon analysis with Bonferroni correction"""
        logging.info("Starting Wilcoxon analysis with Bonferroni correction...")
        
        # First prepare the data structure and L-R files
        self.prepare_wilcoxon_data()
        
        # Then process each folder
        for folder in self.wilcoxon_dir.iterdir():
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
        logging.info(f"Bonferroni divisor: {self.bonferroni_divisor}")
        logging.info(f"Bonferroni threshold used: {self.bonferroni_threshold:.8f}")
        logging.info(f"\nProcess completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def main():
    """Main execution function"""
    analyzer = MEGWilcoxonBonferroniAnalyzer()
    analyzer.run_analysis()

if __name__ == "__main__":
    main()