#!/usr/bin/env python3

"""
MEG Signal Analysis - High Value Analyzer for Original Files
--------------------------------------------------------
This script analyzes all original MEG CSV files to identify values >= 1,
excluding diagonal elements. It processes all files in the 
MEG_20250117_group_project_cat directory and its subdirectories.

Creates a detailed report of where high values occur in the original files.
Output is provided both to console and a timestamped log file.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
import sys
import re

class HighValueAnalyzerOriginal:
    def __init__(self):
        """Initialize the analyzer with necessary paths and data structures"""
        self.data_dir = Path("MEG_20250117_group_project_cat")
        self.logs_dir = Path("Logs")
        
        # Counter for high values
        self.high_value_count = 0
        
        # Setup logging
        self.setup_logging()

    def setup_logging(self):
        """Setup logging to both file and console"""
        timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
        log_file = self.logs_dir / f"high_value_analysis_original_files_{timestamp}.txt"
        
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
        
        logging.info("MEG Signal Analysis - High Value Analyzer for Original Files (Values >= 1)")
        logging.info("======================================================================")
        logging.info(f"Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info("----------------------------------------------------------------------\n")

    def parse_filename(self, filepath: Path) -> dict:
        """
        Parse filename to extract information.
        
        Args:
            filepath (Path): Path to the CSV file
            
        Returns:
            dict: Dictionary containing parsed information
        """
        filename = filepath.stem
        parts = filename.split('_')
        
        info = {
            'filename': filename,  # Store complete filename for reference
            'relative_path': filepath.relative_to(self.data_dir)  # Store relative path
        }
        return info

    def find_high_values(self, filepath: Path):
        """
        Find values >= 1 in CSV file, excluding diagonal elements and treating NaN as zero.
        
        Args:
            filepath (Path): Path to the CSV file
        """
        try:
            # Read CSV file
            matrix = pd.read_csv(filepath, header=None).values
            
            # Replace NaN with 0 (explicit NaN handling)
            matrix = np.nan_to_num(matrix, nan=0.0)
            
            # Create mask for diagonal elements (1 for non-diagonal, 0 for diagonal)
            diagonal_mask = ~np.eye(matrix.shape[0], dtype=bool)
            
            # Apply diagonal mask to exclude diagonal elements
            masked_matrix = matrix * diagonal_mask
            
            # Find positions where values >= 1 (only in non-diagonal elements)
            high_positions = np.where(masked_matrix >= 1)
            
            if len(high_positions[0]) > 0:
                # Get file information
                file_info = self.parse_filename(filepath)
                
                logging.info(f"File: {file_info['relative_path']}")
                logging.info("High values found (excluding diagonal elements):")
                
                for row, col in zip(high_positions[0], high_positions[1]):
                    value = masked_matrix[row, col]
                    logging.info(f"  Value: {value:.6f} at position ({row}, {col})")
                    self.high_value_count += 1
                
                logging.info("-" * 50)

        except Exception as e:
            logging.error(f"Error processing {filepath}: {str(e)}")
            logging.info("-" * 50)

    def analyze_files(self):
        """Analyze all CSV files recursively in the original data directory"""
        if not self.data_dir.exists():
            logging.error(f"Directory not found: {self.data_dir}")
            return
            
        # Process all CSV files recursively
        for filepath in self.data_dir.rglob("*.csv"):
            current_dir = filepath.parent.relative_to(self.data_dir)
            logging.info(f"\nAnalyzing files in {current_dir}")
            logging.info("=" * 50)
            self.find_high_values(filepath)

    def print_summary(self):
        """Print summary of analysis results"""
        logging.info("\nAnalysis Summary")
        logging.info("===============")
        logging.info(f"Total number of values >= 1 found: {self.high_value_count}")
        logging.info("\nAnalysis completed at: " + 
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

def main():
    """Main function to execute the analysis"""
    analyzer = HighValueAnalyzerOriginal()
    analyzer.analyze_files()
    analyzer.print_summary()

if __name__ == "__main__":
    main() 