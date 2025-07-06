#!/usr/bin/env python3

"""
MEG T-test Data Preparation
--------------------------
This script prepares data for T-test analysis by:
1. Creating T_Test directory structure
2. Computing L-R differences for overall and subject averages
3. Organizing results by alignment/method/frequency combinations

The script validates input data and maintains detailed logging.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
import sys
from typing import List, Tuple, Dict

class TTestDataPreparer:
    def __init__(self):
        """Initialize the data preparer"""
        self.data_dir = Path("Data")
        self.logs_dir = Path("Logs")
        
        # Get user selection for dataset folder
        self.dataset_dir = self.select_dataset_folder()
        
        # Ask about excluding iDTF
        self.exclude_idtf = self.ask_exclude_idtf()
        
        # Create T_Test directory
        self.ttest_dir = self.dataset_dir / "T_Test"
        self.ttest_dir.mkdir(exist_ok=True)
        
        # Initialize counters
        self.combinations_processed = 0
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
        log_file = self.logs_dir / f"meg_ttest_preparation_{timestamp}.txt"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        logging.info("MEG T-test Data Preparation")
        logging.info("==========================")
        logging.info(f"Selected dataset: {self.dataset_dir.name}")
        logging.info(f"Excluding iDTF: {self.exclude_idtf}")
        logging.info(f"Process started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info("---------------------\n")

    def find_average_file(self, directory: Path, pattern: str) -> Path:
        """Find average file in directory matching pattern"""
        try:
            avg_files = list(directory.glob(pattern))
            if len(avg_files) == 1:
                return avg_files[0]
            elif len(avg_files) == 0:
                logging.error(f"No average file found matching '{pattern}' in {directory}")
            else:
                logging.error(f"Multiple average files found matching '{pattern}' in {directory}")
            return None
        except Exception as e:
            logging.error(f"Error finding average file: {str(e)}")
            return None

    def find_all_combinations(self) -> List[Tuple[str, str, str]]:
        """
        Find all valid combinations of alignment/method/frequency
        Returns list of tuples (alignment, method, frequency)
        """
        combinations = []
        try:
            # Process each alignment (cue/mov)
            for align_dir in self.dataset_dir.iterdir():
                if align_dir.is_dir() and align_dir.name in ['cue', 'mov']:
                    alignment = align_dir.name
                    
                    # Look in L directory for method_freq combinations
                    l_dir = align_dir / 'L'
                    if l_dir.exists():
                        for method_freq_dir in l_dir.iterdir():
                            if method_freq_dir.is_dir():
                                method, freq = method_freq_dir.name.split('_')
                                
                                # Skip iDTF if requested
                                if self.exclude_idtf and method == 'iDTF':
                                    continue
                                
                                # Verify R directory exists
                                r_dir = align_dir / 'R' / method_freq_dir.name
                                if r_dir.exists():
                                    combinations.append((alignment, method, freq))
                                else:
                                    logging.warning(f"Missing R directory for {alignment}/{method}_{freq}")
            
            logging.info(f"Found {len(combinations)} valid combinations")
            return combinations
        
        except Exception as e:
            logging.error(f"Error finding combinations: {str(e)}")
            self.errors += 1
            return combinations

    def compute_difference(self, left_matrix: pd.DataFrame, 
                          right_matrix: pd.DataFrame) -> pd.DataFrame:
        """Compute L-R difference between matrices"""
        try:
            return left_matrix - right_matrix
        except Exception as e:
            logging.error(f"Error computing difference: {str(e)}")
            self.errors += 1
            return None

    def process_combination(self, alignment: str, method: str, freq: str):
        """Process one alignment/method/frequency combination"""
        try:
            logging.info(f"\nProcessing: {alignment}_{method}_{freq}")
            
            # Create combination directory
            comb_dir = self.ttest_dir / f"{alignment}_{method}_{freq}"
            comb_dir.mkdir(exist_ok=True)
            
            # Get paths
            method_freq = f"{method}_{freq}"
            l_dir = self.dataset_dir / alignment / 'L' / method_freq
            r_dir = self.dataset_dir / alignment / 'R' / method_freq
            
            # Process overall average
            l_avg_file = None
            r_avg_file = None
            
            # Look for L average file
            l_avg_files = list(l_dir.glob("*_average.csv"))
            l_avg_file = l_avg_files[0] if l_avg_files else None
            
            # Look for corresponding R average file
            if l_avg_file:
                r_avg_name = l_avg_file.name.replace('_L_', '_R_')
                r_avg_file = r_dir / r_avg_name
                if not r_avg_file.exists():
                    r_avg_file = None
            
            if l_avg_file and r_avg_file:
                l_avg = pd.read_csv(l_avg_file, header=None)
                r_avg = pd.read_csv(r_avg_file, header=None)
                diff = self.compute_difference(l_avg, r_avg)
                
                if diff is not None:
                    # Save overall difference
                    out_file = comb_dir / f"{alignment}_{method}_{freq}_L-R_overall.csv"
                    diff.to_csv(out_file, index=False, header=False)
                    self.matrices_created += 1
                    logging.info(f"Created overall L-R difference: {out_file}")
                    
                    # Process subject averages
                    for subject_dir in l_dir.iterdir():
                        if subject_dir.is_dir():
                            subject = subject_dir.name
                            logging.info(f"Processing subject: {subject}")
                            
                            # Find subject average files
                            l_subj_files = list(subject_dir.glob("*_average.csv"))
                            if l_subj_files:
                                l_subj_file = l_subj_files[0]
                                r_subj_file = r_dir / subject / l_subj_file.name.replace('_L_', '_R_')
                                
                                if r_subj_file.exists():
                                    l_subj = pd.read_csv(l_subj_file, header=None)
                                    r_subj = pd.read_csv(r_subj_file, header=None)
                                    subj_diff = self.compute_difference(l_subj, r_subj)
                                    
                                    if subj_diff is not None:
                                        # Save subject difference
                                        out_file = comb_dir / f"{alignment}_{method}_{freq}_L-R_{subject}.csv"
                                        subj_diff.to_csv(out_file, index=False, header=False)
                                        self.matrices_created += 1
                                        logging.info(f"Created subject L-R difference: {out_file}")
            
            self.combinations_processed += 1
            
        except Exception as e:
            logging.error(f"Error processing {alignment}_{method}_{freq}: {str(e)}")
            self.errors += 1

    def run_preparation(self):
        """Run the data preparation process"""
        logging.info("Starting T-test data preparation...")
        
        # Find all valid combinations
        combinations = self.find_all_combinations()
        
        # Process each combination
        for alignment, method, freq in combinations:
            self.process_combination(alignment, method, freq)
        
        logging.info("\nPreparation complete!")
        self.print_summary()

    def print_summary(self):
        """Print summary of operations"""
        logging.info("\nOperation Summary")
        logging.info("================")
        logging.info(f"Combinations processed: {self.combinations_processed}")
        logging.info(f"Difference matrices created: {self.matrices_created}")
        logging.info(f"Errors encountered: {self.errors}")
        logging.info(f"\nProcess completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def main():
    """Main execution function"""
    preparer = TTestDataPreparer()
    preparer.run_preparation()

if __name__ == "__main__":
    main() 