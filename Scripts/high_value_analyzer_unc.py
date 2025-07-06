#!/usr/bin/env python3

"""
MEG Signal Analysis - High Value Analyzer for UNC Files
----------------------------------------------------
This script analyzes MEG CSV files ending with '_unc' to identify values >= 1,
excluding diagonal elements. It processes files in the selected folder under Data directory.
User can choose to exclude iDTF files from analysis.

Creates a detailed report of where high values occur in the UNC files.
Output is provided both to console and a timestamped log file.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
import sys
import time

class HighValueAnalyzerUnc:
    def __init__(self):
        """Initialize the analyzer with necessary paths and data structures"""
        self.data_dir = Path("Data")
        self.logs_dir = Path("Logs")
        
        # Get user selection for dataset folder
        self.dataset_dir = self.select_dataset_folder()
        
        # Get user preference for iDTF files
        self.exclude_idtf = self.get_user_preference()
        
        # Counters for statistics
        self.high_value_count = 0
        self.files_analyzed = 0
        self.start_time = time.time()
        
        # Setup logging
        self.setup_logging()

    def select_dataset_folder(self) -> Path:
        """
        Present available folders and let user select one
        
        Returns:
            Path: Selected dataset directory path
        """
        print("\nAvailable datasets under Data folder:")
        print("=====================================")
        
        available_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
        
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

    def get_user_preference(self) -> bool:
        """
        Get user preference for excluding iDTF files
        
        Returns:
            bool: True if iDTF files should be excluded, False otherwise
        """
        while True:
            choice = input("Do you want to exclude iDTF files from analysis? (Y/N): ").strip().upper()
            if choice in ['Y', 'N']:
                return choice == 'Y'
            print("Please enter Y or N")

    def setup_logging(self):
        """Setup logging to both file and console"""
        timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
        log_file = self.logs_dir / f"high_value_analysis_unc_{timestamp}.txt"
        
        # Create logs directory if it doesn't exist
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging configuration
        logging.basicConfig(
            level=logging.INFO,
            format='%(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        logging.info("MEG Signal Analysis - High Value Analyzer for UNC Files (Values >= 1)")
        logging.info("===================================================================")
        logging.info(f"Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info(f"Selected dataset: {self.dataset_dir.name}")
        logging.info(f"iDTF files excluded: {self.exclude_idtf}")
        logging.info("-------------------------------------------------------------------\n")

    def should_process_file(self, filepath: Path) -> bool:
        """
        Check if the file should be processed based on user preferences
        
        Args:
            filepath (Path): Path to the CSV file
            
        Returns:
            bool: True if file should be processed, False otherwise
        """
        # Check if file ends with '_unc.csv'
        if not (filepath.stem.endswith('_unc') and filepath.suffix == '.csv'):
            return False
            
        # Check if file should be excluded based on iDTF preference
        if self.exclude_idtf and 'iDTF' in str(filepath):
            return False
            
        return True

    def find_high_values(self, filepath: Path):
        """
        Find values >= 1 in CSV file, excluding diagonal elements and treating NaN as zero.
        
        Args:
            filepath (Path): Path to the CSV file
        """
        try:
            # Read CSV file
            matrix = pd.read_csv(filepath, header=None).values
            
            # Replace NaN with 0
            matrix = np.nan_to_num(matrix, nan=0.0)
            
            # Create mask for diagonal elements
            diagonal_mask = ~np.eye(matrix.shape[0], dtype=bool)
            
            # Apply mask
            masked_matrix = matrix * diagonal_mask
            
            # Find positions where values >= 1
            high_positions = np.where(masked_matrix >= 1)
            
            if len(high_positions[0]) > 0:
                # Log file information
                rel_path = filepath.relative_to(self.dataset_dir)
                logging.info(f"File: {filepath.name}")
                logging.info(f"Location: {filepath.parent.relative_to(self.dataset_dir)}")
                logging.info("High values found (excluding diagonal elements):")
                
                for row, col in zip(high_positions[0], high_positions[1]):
                    value = masked_matrix[row, col]
                    logging.info(f"  Value: {value:.6f} at position ({row}, {col})")
                    self.high_value_count += 1
                
                logging.info("-" * 70)

        except Exception as e:
            logging.error(f"Error processing {filepath}: {str(e)}")
            logging.info("-" * 70)
        
        self.files_analyzed += 1

    def analyze_files(self):
        """Analyze all UNC CSV files in the selected directory"""
        if not self.dataset_dir.exists():
            logging.error(f"Directory not found: {self.dataset_dir}")
            return
        
        # Process all matching CSV files recursively
        for filepath in self.dataset_dir.rglob("*.csv"):
            if self.should_process_file(filepath):
                self.find_high_values(filepath)

    def print_summary(self):
        """Print summary of analysis results"""
        execution_time = time.time() - self.start_time
        
        logging.info("\nAnalysis Summary")
        logging.info("===============")
        logging.info(f"Total files analyzed: {self.files_analyzed}")
        if self.high_value_count > 0:
            logging.info(f"Total number of values >= 1 found: {self.high_value_count}")
        else:
            logging.info("No values >= 1 found in any file")
        logging.info(f"Execution time: {execution_time:.2f} seconds")
        logging.info(f"iDTF files were {'excluded' if self.exclude_idtf else 'included'} in the analysis")
        logging.info("\nAnalysis completed at: " + 
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

def main():
    """Main function to execute the analysis"""
    analyzer = HighValueAnalyzerUnc()
    analyzer.analyze_files()
    analyzer.print_summary()

if __name__ == "__main__":
    main() 