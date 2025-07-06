"""
MEG Signal Analysis - Matrix Averaging
------------------------------------
This script creates both:
1. Overall averages across all subjects
2. Individual subject averages from their specific measurements
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import List, Dict, Optional

class MatrixAverager:
    def __init__(self):
        self.data_dir = Path("Data")
        self.subjects = ['DOC', 'GB', 'JDC', 'JFXD', 'JZ', 'LT', 'NvA', 'RR', 'SJB', 'BG']
        self.alignments = ['cue', 'mov']
        self.targets = ['L', 'R']
        self.methods = ['iDTF', 'gDTF', 'iPDC', 'gPDC']
        self.frequencies = ['10Hz', '20Hz', '100Hz']
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('matrix_averaging.log'),
                logging.StreamHandler()
            ]
        )

    def create_all_averages(self):
        """Create both overall and subject-specific averages"""
        logging.info("Starting matrix averaging process")
        
        for alignment in self.alignments:
            for target in self.targets:
                for method in self.methods:
                    for freq in self.frequencies:
                        base_dir = self.data_dir / alignment / target / f"{method}_{freq}"
                        
                        if not base_dir.exists():
                            logging.warning(f"Directory not found: {base_dir}")
                            continue
                            
                        logging.info(f"\nProcessing: {alignment}/{target}/{method}_{freq}")
                        
                        # Create overall average
                        self.create_overall_average(base_dir, alignment, target, method, freq)
                        
                        # Create subject-specific averages
                        self.create_subject_averages(base_dir, alignment, target, method, freq)

    def create_overall_average(self, base_dir: Path, alignment: str, target: str, 
                             method: str, freq: str) -> None:
        """Create average matrix across all subjects"""
        try:
            # Get all CSV files (excluding existing averages)
            files = [f for f in base_dir.glob("**/*.csv") 
                    if not f.name.endswith('_average.csv')]
            
            if not files:
                logging.warning(f"No files found for overall average in {base_dir}")
                return
                
            # Read and stack all matrices
            matrices = []
            for file in files:
                try:
                    # Read matrix and replace NaN with 0
                    matrix = pd.read_csv(file, header=None).fillna(0).values
                    matrices.append(matrix)
                except Exception as e:
                    logging.error(f"Error reading {file}: {str(e)}")
                    continue
            
            if not matrices:
                logging.warning("No valid matrices found for averaging")
                return
                
            # Calculate average
            average_matrix = np.mean(matrices, axis=0)
            
            # Replace any remaining NaN with 0 in the average
            average_matrix = np.nan_to_num(average_matrix, nan=0.0)
            
            # Save overall average
            output_filename = f"{alignment}_{target}_{method}_{freq}_average.csv"
            output_path = base_dir / output_filename
            pd.DataFrame(average_matrix).to_csv(output_path, header=False, index=False)
            logging.info(f"Created overall average: {output_filename}")
            
        except Exception as e:
            logging.error(f"Error creating overall average: {str(e)}")

    def create_subject_averages(self, base_dir: Path, alignment: str, target: str, 
                              method: str, freq: str) -> None:
        """Create individual averages for each subject"""
        for subject in self.subjects:
            subject_dir = base_dir / subject
            
            if not subject_dir.exists():
                logging.info(f"No directory found for subject {subject}")
                continue
                
            try:
                # Get all subject's files (excluding any existing averages)
                files = [f for f in subject_dir.glob("*.csv") 
                        if not f.name.endswith('_average.csv')]
                
                if not files:
                    logging.warning(f"No files found for subject {subject}")
                    continue
                    
                # Read and stack subject's matrices
                matrices = []
                for file in files:
                    try:
                        # Read matrix and replace NaN with 0
                        matrix = pd.read_csv(file, header=None).fillna(0).values
                        matrices.append(matrix)
                    except Exception as e:
                        logging.error(f"Error reading {file}: {str(e)}")
                        continue
                
                if not matrices:
                    logging.warning(f"No valid matrices found for subject {subject}")
                    continue
                    
                # Calculate subject's average
                subject_average = np.mean(matrices, axis=0)
                
                # Replace any remaining NaN with 0 in the average
                subject_average = np.nan_to_num(subject_average, nan=0.0)
                
                # Save subject's average
                output_filename = f"{alignment}_{target}_{method}_{freq}_{subject}_average.csv"
                output_path = subject_dir / output_filename
                pd.DataFrame(subject_average).to_csv(output_path, header=False, index=False)
                logging.info(f"Created average for subject {subject}: {output_filename}")
                
                # Print summary for this subject
                logging.info(f"Subject {subject}: Averaged {len(matrices)} files")
                
            except Exception as e:
                logging.error(f"Error processing subject {subject}: {str(e)}")

def main():
    print("Matrix Averaging Program")
    print("======================")
    
    averager = MatrixAverager()
    
    response = input("Do you want to create matrix averages? (y/n): ").lower()
    if response == 'y':
        averager.create_all_averages()
        print("\nAveraging complete. Check matrix_averaging.log for details.")
    else:
        print("Program terminated by user.")

if __name__ == "__main__":
    main()
