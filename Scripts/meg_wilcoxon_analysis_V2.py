#!/usr/bin/env python3

"""
MEG Wilcoxon Analysis V2
---------------------
This script performs Wilcoxon signed-rank tests on MEG data differences between Left_movement and Right_movement conditions.
V2 is compatible with the new Left_movement/Right_movement folder structure.

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

class MEGWilcoxonAnalyzerV2:
    def __init__(self):
        """Initialize the Wilcoxon analyzer V2"""
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
        log_file = self.logs_dir / f"meg_wilcoxon_analysis_V2_{timestamp}.txt"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        logging.info("MEG Wilcoxon Analysis V2")
        logging.info("=======================")
        logging.info(f"Selected dataset: {self.dataset_dir.name}")
        logging.info(f"Excluding iDTF: {self.exclude_idtf}")
        logging.info(f"Process started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info("--------------------------------------------------\n")

    def has_required_files(self, left_base_dir: Path, right_base_dir: Path, prefix: str, method_freq: str) -> bool:
        """Check if required files exist in both Left_movement and Right_movement directories"""
        logging.info(f"Checking for required files in {method_freq}...")
        
        # Check if directories exist
        if not left_base_dir.exists():
            logging.warning(f"Left_movement directory does not exist: {left_base_dir}")
            return False
        if not right_base_dir.exists():
            logging.warning(f"Right_movement directory does not exist: {right_base_dir}")
            return False
        
        # Check for average files in both directories
        left_file_found = False
        right_file_found = False
        
        # Look for overall average files
        left_pattern = f"{prefix}_{method_freq.split('_')[0]}_{method_freq.split('_')[1]}_average.csv"
        right_pattern = f"{prefix}_{method_freq.split('_')[0]}_{method_freq.split('_')[1]}_average.csv"
        
        left_files = list(left_base_dir.glob(left_pattern))
        right_files = list(right_base_dir.glob(right_pattern))
        
        if left_files:
            left_file_found = True
            logging.info(f"Found Left_movement file: {left_files[0].name}")
        else:
            logging.warning(f"No Left_movement file found matching pattern: {left_pattern}")
        
        if right_files:
            right_file_found = True
            logging.info(f"Found Right_movement file: {right_files[0].name}")
        else:
            logging.warning(f"No Right_movement file found matching pattern: {right_pattern}")
            
        if left_file_found and right_file_found:
            return True
        
        return False

    def prepare_wilcoxon_data(self):
        """Prepare data structure and Left_movement-Right_movement files for Wilcoxon analysis"""
        logging.info("Preparing data for Wilcoxon analysis...")
        logging.info(f"Looking in dataset directory: {self.dataset_dir}")
        
        # Check for available condition folders (mov and/or cue)
        conditions = []
        if (self.dataset_dir / "mov").exists():
            conditions.append(("mov", "mov"))
        if (self.dataset_dir / "cue").exists():
            conditions.append(("cue", "cue"))
        
        if not conditions:
            logging.error("No mov or cue directories found!")
            return
        
        logging.info(f"Found conditions: {[c[0] for c in conditions]}")
        
        # Get all method-frequency combinations
        method_freq_dirs = []
        for method in ['gDTF', 'iPDC', 'gPDC', 'iDTF']:
            if self.exclude_idtf and method == 'iDTF':
                continue
            for freq in ['10Hz', '20Hz', '25Hz', '100Hz']:
                method_freq_dirs.append(f"{method}_{freq}")
        
        # Process each condition (mov and/or cue)
        for condition_dir, prefix in conditions:
            logging.info(f"\nProcessing condition: {condition_dir}")
            
            # Create directories and process files
            for method_freq in method_freq_dirs:
                try:
                    # Create method_freq directory with condition prefix
                    method_freq_dir = self.wilcoxon_dir / f"{condition_dir}_{method_freq}"
                    method_freq_dir.mkdir(parents=True, exist_ok=True)
                    logging.info(f"\nProcessing {method_freq}")
                    
                    # Define paths for Left_movement and Right_movement
                    left_base_dir = self.dataset_dir / condition_dir / "Left_movement" / method_freq
                    right_base_dir = self.dataset_dir / condition_dir / "Right_movement" / method_freq
                    
                    logging.info(f"Looking in directories:")
                    logging.info(f"Left_movement base directory: {left_base_dir}")
                    logging.info(f"Right_movement base directory: {right_base_dir}")
                    
                    if not self.has_required_files(left_base_dir, right_base_dir, prefix, method_freq):
                        logging.warning(f"Skipping {method_freq} - required files not found")
                        # Remove the empty directory we just created
                        if method_freq_dir.exists():
                            shutil.rmtree(method_freq_dir)
                        continue
                    
                    logging.info(f"Required files found for {method_freq}, proceeding with processing...")
                    
                    # Get list of subject folders from Left_movement directory
                    subject_folders = [d.name for d in left_base_dir.iterdir() if d.is_dir() 
                                     and d.name != self.dataset_dir.name]
                    
                    logging.info(f"\nFound subject folders: {subject_folders}")
                    
                    # Process each subject
                    all_differences = []
                    for subject in subject_folders:
                        # Look for files with both movement and cue patterns
                        method, freq = method_freq.split('_')
                        left_patterns = [
                            f"{prefix}_{method}_{freq}_{subject}_average.csv",  # Individual subject pattern
                            f"{prefix}_{method}_{freq}_average.csv"             # Overall pattern
                        ]
                        right_patterns = [
                            f"{prefix}_{method}_{freq}_{subject}_average.csv",
                            f"{prefix}_{method}_{freq}_average.csv"
                        ]
                        
                        left_file = None
                        right_file = None
                        
                        # Try each pattern
                        for left_pattern in left_patterns:
                            found = list(left_base_dir.joinpath(subject).glob(left_pattern))
                            if found:
                                left_file = found[0]
                                break
                        
                        for right_pattern in right_patterns:
                            found = list(right_base_dir.joinpath(subject).glob(right_pattern))
                            if found:
                                right_file = found[0]
                                break
                        
                        if left_file and right_file:
                            logging.info(f"\nProcessing subject {subject}:")
                            logging.info(f"Left_movement file: {left_file}")
                            logging.info(f"Right_movement file: {right_file}")
                            
                            # Read matrices
                            left_matrix = pd.read_csv(left_file, header=None).values
                            right_matrix = pd.read_csv(right_file, header=None).values
                            
                            # Calculate difference (Left_movement - Right_movement)
                            diff_matrix = left_matrix - right_matrix
                            all_differences.append(diff_matrix)
                            
                            # Save subject difference
                            diff_file = method_freq_dir / f"{method_freq}_Left-Right_{subject}.csv"
                            pd.DataFrame(diff_matrix).to_csv(diff_file, index=False, header=False)
                            logging.info(f"Created Left-Right difference file for subject {subject}")
                        else:
                            logging.warning(f"Missing files for subject {subject}")
                    
                    logging.info(f"\nTotal subjects processed: {len(all_differences)}")
                    
                    # Calculate and save overall difference
                    if all_differences:
                        overall_diff = np.mean(all_differences, axis=0)
                        overall_file = method_freq_dir / f"{method_freq}_Left-Right_overall.csv"
                        pd.DataFrame(overall_diff).to_csv(overall_file, index=False, header=False)
                        logging.info(f"Created overall Left-Right difference file")
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
            'V1-R', 'V3-R', 'SPOC-R', 'AG-R', 'POJ-R', 'SPL-R', 'mIPS-R', 'VIP-R'
        ]
        
        # Create heatmap
        sns.heatmap(matrix, 
                   xticklabels=region_labels, 
                   yticklabels=region_labels,
                   cmap='RdBu_r', 
                   center=0,
                   square=True,
                   cbar_kws={'label': 'Difference (Left_movement - Right_movement)'})
        
        plt.title(title, fontsize=16, pad=20)
        plt.xlabel('Target Region', fontsize=12)
        plt.ylabel('Source Region', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"Created heatmap: {output_path}")

    def process_folder(self, folder_path: Path):
        """Process a single Wilcoxon folder and create detailed analysis"""
        logging.info(f"\nProcessing folder: {folder_path.name}")
        
        # Get all CSV files in the folder
        csv_files = list(folder_path.glob("*.csv"))
        
        if not csv_files:
            logging.warning(f"No CSV files found in {folder_path.name}")
            return
        
        # Extract method_freq from folder name (e.g., "cue_gPDC_10Hz" -> "gPDC_10Hz")
        folder_name = folder_path.name
        if '_' in folder_name:
            method_freq = '_'.join(folder_name.split('_')[1:])  # Remove condition prefix
        else:
            method_freq = folder_name
        
        # Find overall file and subject files
        overall_file = None
        subject_files = []
        
        for csv_file in csv_files:
            if 'overall' in csv_file.name:
                overall_file = csv_file
            elif 'Left-Right_' in csv_file.name and 'overall' not in csv_file.name:
                subject_files.append(csv_file)
        
        if not overall_file:
            logging.warning(f"No overall file found in {folder_path.name}")
            return
        
        if not subject_files:
            logging.warning(f"No subject files found in {folder_path.name}")
            return
        
        # Read overall matrix
        overall_matrix = pd.read_csv(overall_file, header=None).values
        matrix_size = overall_matrix.shape[0]
        output_matrix = np.zeros_like(overall_matrix)
        
        # Read subject matrices
        subject_matrices = {}
        for subject_file in subject_files:
            subject_name = subject_file.stem.split('_')[-1]  # Extract subject name
            subject_matrix = pd.read_csv(subject_file, header=None).values
            subject_matrices[subject_name] = subject_matrix
        
        # Create detailed log file for this subfolder
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
                        
                        if len(values) != len(subject_matrices):
                            logging.warning(f"Cell ({i},{j}): Found {len(values)} values (expected {len(subject_matrices)})")
                        
                        # Perform Wilcoxon test against zero
                        w_stat, p_value = stats.wilcoxon(values, alternative='two-sided')
                        is_significant = p_value < 0.05
                        
                        # Log results
                        f.write(f"Cell ({i}, {j}):\n")
                        f.write(f"Subject Values: {values}\n")
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
        
        logging.info(f"Created detailed log file: {log_file}")
        
        # Save output matrix
        output_file = folder_path / f"{method_freq}_L-R_wilcoxon.csv"
        pd.DataFrame(output_matrix).to_csv(output_file, index=False, header=False)
        
        # Create heatmap for original overall matrix
        heatmap_filename = overall_file.stem + "_heatmap.png"
        heatmap_path = folder_path / heatmap_filename
        title = f"Left_movement - Right_movement Difference\n{folder_path.name}"
        self.create_heatmap(overall_matrix, heatmap_path, title)
        
        # Create heatmap for Wilcoxon filtered matrix
        wilcoxon_heatmap_file = folder_path / f"{method_freq}_L-R_wilcoxon_heatmap.png"
        self.create_heatmap(output_matrix, wilcoxon_heatmap_file, 
                          f"{method_freq} Wilcoxon Filtered Matrix")
        
        self.matrices_created += 2  # Both original and filtered heatmaps
        self.folders_processed += 1

    def run_analysis(self):
        """Run the complete Wilcoxon analysis"""
        logging.info("Starting Wilcoxon analysis...")
        
        # Step 1: Prepare data
        self.prepare_wilcoxon_data()
        
        # Step 2: Process folders and create heatmaps
        logging.info("\nCreating heatmaps...")
        for folder in self.wilcoxon_dir.iterdir():
            if folder.is_dir():
                self.process_folder(folder)
                self.folders_processed += 1
        
        logging.info("Wilcoxon analysis completed!")

    def has_sufficient_data(self, folder_path: Path, condition: str, method_freq: str) -> bool:
        """Check if there's sufficient data for analysis"""
        csv_files = list(folder_path.glob("*.csv"))
        return len(csv_files) > 0

    def print_summary(self):
        """Print summary of operations"""
        logging.info("\nOperation Summary")
        logging.info("================")
        logging.info(f"Folders processed: {self.folders_processed}")
        logging.info(f"Matrices created: {self.matrices_created}")
        logging.info(f"Folders skipped: {self.folders_skipped}")
        logging.info(f"Errors encountered: {self.errors}")
        logging.info(f"\nProcess completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def main():
    """Main function to execute the Wilcoxon analysis"""
    analyzer = MEGWilcoxonAnalyzerV2()
    analyzer.run_analysis()
    analyzer.print_summary()

if __name__ == "__main__":
    main()