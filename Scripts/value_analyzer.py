#!/usr/bin/env python3

"""
MEG Signal Analysis - Value Analyzer
----------------------------------
This script analyzes all MEG measurement CSV files to find maximum values,
excluding diagonal elements. It processes files in Data/cue and Data/mov
directories, skipping average and temporary files.

Output is provided both to console and a timestamped log file.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
import sys

class ValueAnalyzer:
    def __init__(self):
        """Initialize the analyzer with necessary paths and data structures"""
        self.data_dir = Path("Data")
        self.logs_dir = Path("Logs")
        self.alignments = ['cue', 'mov']
        self.hemispheres = ['L', 'R']
        
        # Track global maximum
        self.global_max = 0
        self.global_max_file = None
        self.global_max_position = None
        
        # Setup logging
        self.setup_logging()

    def setup_logging(self):
        """Setup logging to both file and console"""
        timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
        log_file = self.logs_dir / f"value_analysis_{timestamp}.txt"
        
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
        
        logging.info("MEG Signal Analysis - Value Analyzer")
        logging.info("====================================")
        logging.info(f"Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info("------------------------------------\n")

    def should_process_file(self, filepath: Path) -> bool:
        """
        Check if the file should be processed based on naming conventions.
        
        Args:
            filepath (Path): Path to the CSV file
            
        Returns:
            bool: True if file should be processed, False otherwise
        """
        filename = filepath.name.lower()
        return (filepath.suffix == '.csv' and 
                'average' not in filename and 
                'temp' not in filename)

    def find_max_value(self, filepath: Path):
        """
        Find maximum value in CSV file, excluding diagonal elements and treating NaN as zero.
        
        Args:
            filepath (Path): Path to the CSV file
        """
        try:
            # Read CSV file
            matrix = pd.read_csv(filepath, header=None).values
            
            # Replace NaN with 0
            matrix = np.nan_to_num(matrix, nan=0.0)
            
            # Create mask for diagonal elements
            mask = ~np.eye(matrix.shape[0], dtype=bool)
            
            # Apply mask and get maximum value
            masked_matrix = matrix * mask
            max_value = np.max(masked_matrix)
            max_position = np.unravel_index(np.argmax(masked_matrix), matrix.shape)
            
            # Update global maximum if necessary
            if self.global_max_file is None or max_value > self.global_max:
                self.global_max = max_value
                self.global_max_file = filepath
                self.global_max_position = max_position
            
            # Log results
            logging.info(f"File: {filepath.relative_to(self.data_dir)}")
            logging.info(f"Maximum value: {max_value}")
            logging.info(f"Position: {max_position}")
            logging.info("-" * 50)

        except Exception as e:
            logging.error(f"Error processing {filepath}: {str(e)}")
            logging.info("-" * 50)

    def analyze_files(self):
        """Analyze all relevant CSV files in the data directory"""
        for alignment in self.alignments:
            for hemisphere in self.hemispheres:
                # Get base path for this combination
                base_path = self.data_dir / alignment / hemisphere
                
                # Skip if path doesn't exist
                if not base_path.exists():
                    continue
                
                # Process all method/frequency directories
                for method_freq_dir in base_path.iterdir():
                    if method_freq_dir.is_dir():
                        # Process all CSV files in this directory
                        for filepath in method_freq_dir.glob("*.csv"):
                            if self.should_process_file(filepath):
                                self.find_max_value(filepath)

    def print_summary(self):
        """Print summary of analysis results"""
        logging.info("\nAnalysis Summary")
        logging.info("===============")
        logging.info(f"Global maximum value: {self.global_max:.6f}")
        
        # Check if a file with maximum value was found
        if self.global_max_file is not None:
            logging.info(f"Found in file: {self.global_max_file.relative_to(self.data_dir)}")
            logging.info(f"At position: {self.global_max_position}")
        else:
            logging.info("No files were processed")
        
        logging.info("\nAnalysis completed at: " + 
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

def main():
    """Main function to execute the analysis"""
    analyzer = ValueAnalyzer()
    analyzer.analyze_files()
    analyzer.print_summary()

if __name__ == "__main__":
    main() 