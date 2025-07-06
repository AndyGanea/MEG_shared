#!/usr/bin/env python3

"""
MEG Wilcoxon Analysis
------------------
This script performs Wilcoxon signed-rank tests on MEG data differences between L and R conditions.
For each cell in the matrices (excluding diagonal), it:
1. Collects values from all subjects
2. Performs Wilcoxon test against zero
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
import shutil
from typing import List, Tuple, Dict

class MEGWilcoxonAnalyzer:
    def __init__(self):
        """Initialize the Wilcoxon analyzer"""
        self.data_dir = Path("Data")
        self.logs_dir = Path("Logs")
        
        # Get user selection for dataset folder
        self.dataset_dir = self.select_dataset_folder()
        
        # Ask about excluding iDTF
        self.exclude_idtf = self.ask_exclude_idtf()
        
        # Create timestamp for directory naming
        self.timestamp = datetime.now().strftime("%m%d%Y-%H%M")
        
        # Create or clean Wilcoxon directory with timestamp
        self.wilcoxon_dir = self.dataset_dir / f"Wilcoxon_{self.timestamp}"
        if self.wilcoxon_dir.exists():
            # Remove existing directory and its contents
            shutil.rmtree(self.wilcoxon_dir)
            logging.info(f"Removed existing Wilcoxon directory in {self.dataset_dir}")
        
        # Create fresh Wilcoxon directory
        self.wilcoxon_dir.mkdir(parents=True)
        logging.info(f"Created Wilcoxon directory: {self.wilcoxon_dir.name}")
        
        # Initialize counters
        self.folders_processed = 0
        self.matrices_created = 0
        self.errors = 0
        self.folders_skipped = 0  # New counter for skipped folders
        
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
        log_file = self.logs_dir / f"meg_wilcoxon_analysis_{timestamp}.txt"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        logging.info("MEG Wilcoxon Analysis")
        logging.info("==================")
        logging.info(f"Selected dataset: {self.dataset_dir.name}")
        logging.info(f"Excluding iDTF: {self.exclude_idtf}")
        logging.info(f"Output directory: {self.wilcoxon_dir.name}")
        logging.info(f"Process started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info("---------------------\n")

    def has_required_files(self, l_base_dir: Path, r_base_dir: Path, prefix: str, method_freq: str) -> bool:
        """Check if the required L and R files exist for processing"""
        if not l_base_dir.exists() or not r_base_dir.exists():
            return False
        
        # Get subject folders
        subject_folders = [d.name for d in l_base_dir.iterdir() if d.is_dir() 
                         and d.name != self.dataset_dir.name]
        
        if not subject_folders:
            return False
        
        # Check if at least one subject has the required files
        for subject in subject_folders:
            l_patterns = [
                f"{prefix}_L_{method_freq}_{subject}_average.csv",
                f"{prefix}_L_{method_freq}_average.csv"
            ]
            r_patterns = [
                f"{prefix}_R_{method_freq}_{subject}_average.csv",
                f"{prefix}_R_{method_freq}_average.csv"
            ]
            
            l_file_found = any(list(l_base_dir.joinpath(subject).glob(pattern)) for pattern in l_patterns)
            r_file_found = any(list(r_base_dir.joinpath(subject).glob(pattern)) for pattern in r_patterns)
            
            if l_file_found and r_file_found:
                return True
        
        return False

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
            for freq in ['10Hz', '20Hz', '25Hz', '100Hz']:
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
                    
                    if not self.has_required_files(l_base_dir, r_base_dir, prefix, method_freq):
                        logging.warning(f"Skipping {method_freq} - required files not found")
                        # Remove the empty directory we just created
                        if method_freq_dir.exists():
                            shutil.rmtree(method_freq_dir)
                        continue
                    
                    logging.info(f"Required files found for {method_freq}, proceeding with processing...")
                    
                    # First, get list of subject folders
                    subject_folders = [d.name for d in l_base_dir.iterdir() if d.is_dir() 
                                     and d.name != self.dataset_dir.name]
                    
                    logging.info(f"\nFound subject folders: {subject_folders}")
                    
                    # Process each subject
                    all_differences = []
                    for subject in subject_folders:
                        # Look for files with both movement and cue patterns
                        l_patterns = [
                            f"{prefix}_L_{method_freq}_{subject}_average.csv",  # Individual subject pattern
                            f"{prefix}_L_{method_freq}_average.csv"             # Overall pattern
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
                        # Remove the empty directory
                        if method_freq_dir.exists():
                            shutil.rmtree(method_freq_dir)
                        
                except Exception as e:
                    logging.error(f"Error processing {method_freq}: {str(e)}")
                    self.errors += 1

    def create_heatmap(self, matrix: np.ndarray, output_path: Path, title: str):
        """Create and save heatmap visualization"""
        plt.figure(figsize=(14, 12))
        
        # Create region labels as provided
        region_labels = [
            'V1-L', 'V3-L', 'SPOC-L', 'AG-L', 'POJ-L', 'SPL-L', 'mIPS-L', 'VIP-L', 
            'IPL-L', 'STS-L', 'S1-L', 'M1-L', 'SMA-L', 'PMd-L', 'FEF-L', 'PMv-L',
            'V1-R', 'V3-R', 'SPOC-R', 'AG-R', 'POJ-R', 'SPL-R', 'mIPS-R', 'VIP-R', 
            'IPL-R', 'STS-R', 'S1-R', 'M1-R', 'SMA-R', 'PMd-R', 'FEF-R', 'PMv-R'
        ]
        
        # Create heatmap with region labels and store the returned mappable
        heatmap = sns.heatmap(matrix, 
                    cmap='RdBu_r', 
                    center=0,
                    xticklabels=region_labels,
                    yticklabels=region_labels)
        
        plt.title(title)
        
        # Set x-axis labels to 45 degrees with right alignment for better readability
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.xlabel('Region')
        plt.ylabel('Region')
        
        # Add colorbar label using the figure's colorbar
        cbar = heatmap.collections[0].colorbar
        cbar.set_label('Connectivity Strength')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    def process_folder(self, folder_path: Path):
        """Process one alignment/method/frequency folder"""
        try:
            # Extract method_freq and condition from folder name
            folder_parts = folder_path.name.split('_')
            condition = folder_parts[0].lower()  # 'Mov' or 'Cue'
            method_freq = '_'.join(folder_parts[1:])  # 'gDTF_10Hz'
            
            if self.exclude_idtf and method_freq.startswith('iDTF'):
                logging.info(f"Skipping iDTF folder: {folder_path.name}")
                self.folders_skipped += 1
                return
            
            logging.info(f"\nProcessing: {folder_path.name}")
            
            # Check if folder has sufficient data for analysis
            if not self.has_sufficient_data(folder_path, condition, method_freq):
                logging.warning(f"Skipping {folder_path.name} - insufficient data for analysis")
                self.folders_skipped += 1
                return
            
            # Read all subject matrices
            logging.info("Starting to read subject matrices...")
            subject_matrices = {}
            
            # Look for all subject L-R files
            subject_files = list(folder_path.glob("*_L-R_[A-Z]*.csv"))
            subject_files = [f for f in subject_files if "overall" not in f.name.lower()]
            
            if len(subject_files) != 10:
                logging.warning(f"Found {len(subject_files)} subject files (expected 10)")
            
            for subj_file in subject_files:
                subject = subj_file.name.split('_L-R_')[1].replace('.csv', '')
                subject_matrices[subject] = pd.read_csv(subj_file, header=None).values
                logging.info(f"Read subject matrix: {subj_file.name}")
            
            # Get overall matrix file
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
            
            overall_matrix = pd.read_csv(overall_file, header=None).values
            matrix_size = overall_matrix.shape[0]
            output_matrix = np.zeros_like(overall_matrix)
            
            # Create text log file
            log_file = folder_path / f"{method_freq}_wilcoxon.txt"
            with open(log_file, 'w') as f:
                f.write(f"MEG Wilcoxon Analysis Results\n")
                f.write(f"============================\n")
                f.write(f"Method-Frequency: {method_freq}\n")
                f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write("Cell-by-Cell Analysis\n")
                f.write("--------------------\n\n")
                
                significant_cells = 0
                nonsignificant_cells = 0
                
                for i in range(matrix_size):
                    for j in range(matrix_size):
                        if i != j:  # Skip diagonal
                            # Collect values from all subjects for this cell
                            values = []
                            for subject, matrix in subject_matrices.items():
                                values.append(matrix[i,j])
                            
                            if len(values) != 10:
                                logging.warning(f"Cell ({i},{j}): Found {len(values)} values (expected 10)")
                            
                            # Perform Wilcoxon test against zero
                            w_stat, p_value = stats.wilcoxon(values, alternative='two-sided')
                            is_significant = p_value < 0.05
                            
                            # Log results
                            f.write(f"Cell ({i}, {j}):\n")
                            f.write(f"Subject Values: {values}\n")  # Now will show all 10 values
                            f.write(f"W-statistic: {w_stat:.4f}\n")
                            f.write(f"P-value: {p_value:.4f}\n")
                            f.write(f"Result: {'Significant (p < 0.05)' if is_significant else 'Not Significant (p >= 0.05)'}\n")
                            f.write(f"Overall Matrix Value: {overall_matrix[i,j]:.4f} ")
                            f.write(f"({'Retained' if is_significant else 'Set to zero'})\n\n")
                            
                            # Update output matrix
                            output_matrix[i,j] = overall_matrix[i,j] if is_significant else 0
                            
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
            output_file = folder_path / f"{method_freq}_L-R_wilcoxon.csv"
            pd.DataFrame(output_matrix).to_csv(output_file, index=False, header=False)
            
            # Create heatmap
            heatmap_file = folder_path / f"{method_freq}_L-R_wilcoxon_heatmap.png"
            self.create_heatmap(output_matrix, heatmap_file, 
                              f"{method_freq} Wilcoxon Filtered Matrix")
            
            self.matrices_created += 1
            self.folders_processed += 1
            
        except Exception as e:
            logging.error(f"Error processing folder {folder_path.name}: {str(e)}")
            self.errors += 1

    def run_analysis(self):
        """Run the Wilcoxon analysis"""
        logging.info("Starting Wilcoxon analysis...")
        
        # First prepare the data structure and L-R files
        self.prepare_wilcoxon_data()
        
        # Then process each folder for Wilcoxon analysis
        folders_found = list(self.wilcoxon_dir.iterdir())
        folders_found = [f for f in folders_found if f.is_dir()]
        
        logging.info(f"\nFound {len(folders_found)} folders to process:")
        for folder in folders_found:
            logging.info(f"  - {folder.name}")
        
        if not folders_found:
            logging.warning("No folders found to process!")
            return
        
        for folder in folders_found:
            self.process_folder(folder)
        
        logging.info("\nAnalysis complete!")
        self.print_summary()

    def has_sufficient_data(self, folder_path: Path, condition: str, method_freq: str) -> bool:
        """Check if a folder has sufficient data for Wilcoxon analysis"""
        # Check for subject files
        subject_files = list(folder_path.glob("*_L-R_[A-Z]*.csv"))
        subject_files = [f for f in subject_files if "overall" not in f.name.lower()]
        
        if len(subject_files) < 3:  # Need at least 3 subjects for meaningful analysis
            return False
        
        # Check for overall matrix file
        overall_patterns = [
            f"{condition}_L_{method_freq}_average.csv",
            f"{method_freq}_L-R_overall.csv"
        ]
        
        overall_file_exists = any((folder_path / pattern).exists() for pattern in overall_patterns)
        
        return overall_file_exists

    def print_summary(self):
        """Print summary of operations"""
        logging.info("\nOperation Summary")
        logging.info("================")
        logging.info(f"Folders processed: {self.folders_processed}")
        logging.info(f"Matrices created: {self.matrices_created}")
        logging.info(f"Folders skipped: {self.folders_skipped}")
        logging.info(f"Errors encountered: {self.errors}")
        
        total_folders = self.folders_processed + self.folders_skipped + self.errors
        if total_folders > 0:
            success_rate = (self.folders_processed / total_folders) * 100
            logging.info(f"Success rate: {success_rate:.1f}%")
        
        logging.info(f"\nProcess completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def main():
    """Main execution function"""
    analyzer = MEGWilcoxonAnalyzer()
    analyzer.run_analysis()

if __name__ == "__main__":
    main()