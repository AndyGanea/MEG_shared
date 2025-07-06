#!/usr/bin/env python3

"""
MEG Signal Analysis - High Value Analyzer
---------------------------------------
This script analyzes all MEG measurement CSV files to identify values >= 1,
excluding diagonal elements. It processes files in Data/cue and Data/mov
directories, skipping average and temporary files.

Creates a detailed report of where high values occur, organized by:
- Alignment (cue/mov)
- Hemisphere (L/R)
- Method (iDTF/gDTF/iPDC/gPDC)
- Frequency (10Hz/20Hz/100Hz)
- Movement type and subject

Output is provided both to console and a timestamped log file.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
import sys
import re

class HighValueAnalyzer:
    def __init__(self):
        """Initialize the analyzer with necessary paths and data structures"""
        self.data_dir = Path("Data")
        self.logs_dir = Path("Logs")
        self.alignments = ['cue', 'mov']
        self.hemispheres = ['L', 'R']
        
        # Counter for high values
        self.high_value_count = 0
        
        # Setup logging
        self.setup_logging()

    def setup_logging(self):
        """Setup logging to both file and console"""
        timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
        log_file = self.logs_dir / f"high_value_analysis_{timestamp}.txt"
        
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
        
        logging.info("MEG Signal Analysis - High Value Analyzer (Values >= 1)")
        logging.info("===================================================")
        logging.info(f"Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info("---------------------------------------------------\n")

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

    def parse_filename(self, filepath: Path) -> dict:
        """
        Parse filename to extract movement type and subject information.
        
        Args:
            filepath (Path): Path to the CSV file
            
        Returns:
            dict: Dictionary containing parsed information
        """
        filename = filepath.stem
        parts = filename.split('_')
        
        info = {
            'movement': '_'.join(parts[1:-1]),  # Everything between first and last parts
            'subject': parts[-1].split('.')[0]  # Last part before extension
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
                rel_path = filepath.relative_to(self.data_dir)
                file_info = self.parse_filename(filepath)
                
                logging.info(f"File: {rel_path}")
                logging.info(f"Movement: {file_info['movement']}")
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
                        method_freq = method_freq_dir.name
                        logging.info(f"\nAnalyzing {alignment}/{hemisphere}/{method_freq}")
                        logging.info("=" * 50)
                        
                        # Process all CSV files in this directory
                        for filepath in method_freq_dir.glob("*.csv"):
                            if self.should_process_file(filepath):
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
    analyzer = HighValueAnalyzer()
    analyzer.analyze_files()
    analyzer.print_summary()

if __name__ == "__main__":
    main() 