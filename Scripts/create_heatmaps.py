"""
MEG Signal Analysis - Heatmap Generator
-------------------------------------
This script creates heatmaps for:
1. Overall averages (ignoring diagonal values)
2. Individual subject averages (ignoring diagonal values)
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from typing import List, Optional, Tuple

class HeatmapGenerator:
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
                logging.FileHandler('heatmap_generation.log'),
                logging.StreamHandler()
            ]
        )

    def create_all_heatmaps(self):
        """Create both overall and subject-specific heatmaps"""
        logging.info("Starting heatmap generation process")
        
        for alignment in self.alignments:
            for target in self.targets:
                for method in self.methods:
                    for freq in self.frequencies:
                        base_dir = self.data_dir / alignment / target / f"{method}_{freq}"
                        
                        if not base_dir.exists():
                            logging.warning(f"Directory not found: {base_dir}")
                            continue
                        
                        logging.info(f"\nProcessing: {alignment}/{target}/{method}_{freq}")
                        
                        # Create overall average heatmap
                        self.create_overall_heatmap(base_dir, alignment, target, method, freq)
                        
                        # Create subject-specific heatmaps
                        self.create_subject_heatmaps(base_dir, alignment, target, method, freq)

    def process_matrix_for_heatmap(self, matrix: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """
        Process matrix by ignoring diagonal values for scaling.
        
        Args:
            matrix: Input matrix
            
        Returns:
            Tuple of (processed matrix, vmin, vmax)
        """
        # Create a copy to avoid modifying the original
        matrix_copy = matrix.copy()
        
        # Get diagonal indices
        diag_indices = np.diag_indices_from(matrix_copy)
        
        # Store diagonal values for later
        diagonal_values = matrix_copy[diag_indices]
        
        # Temporarily set diagonal to nan to ignore in min/max calculation
        matrix_copy[diag_indices] = np.nan
        
        # Calculate min/max excluding diagonal
        vmin = np.nanmin(matrix_copy)
        vmax = np.nanmax(matrix_copy)
        
        # Restore diagonal values
        matrix_copy[diag_indices] = diagonal_values
        
        return matrix_copy, vmin, vmax

    def create_overall_heatmap(self, base_dir: Path, alignment: str, target: str, 
                             method: str, freq: str) -> None:
        """Create heatmap for overall average matrix"""
        try:
            # Load average matrix
            avg_filename = f"{alignment}_{target}_{method}_{freq}_average.csv"
            avg_path = base_dir / avg_filename
            
            if not avg_path.exists():
                logging.warning(f"Average file not found: {avg_filename}")
                return
            
            matrix = pd.read_csv(avg_path, header=None).values
            
            # Process matrix
            processed_matrix, vmin, vmax = self.process_matrix_for_heatmap(matrix)
            
            # Create heatmap
            plt.figure(figsize=(12, 10))
            sns.heatmap(processed_matrix, 
                       cmap='viridis',
                       vmin=vmin,
                       vmax=vmax,
                       mask=np.diag(np.ones(processed_matrix.shape[0])),  # Mask diagonal
                       cbar_kws={'label': 'Connectivity Value'})
            
            plt.title(f'Average Connectivity Matrix\n{alignment.upper()} {target} {method} {freq}')
            plt.xlabel('Target Region')
            plt.ylabel('Source Region')
            
            # Save heatmap with correct naming convention
            output_filename = f"{alignment}_{target}_{method}_{freq}_average_heatmap.png"
            plt.savefig(base_dir / output_filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info(f"Created overall heatmap: {output_filename}")
            
        except Exception as e:
            logging.error(f"Error creating overall heatmap: {str(e)}")

    def create_subject_heatmaps(self, base_dir: Path, alignment: str, target: str, 
                              method: str, freq: str) -> None:
        """Create heatmaps for individual subject averages"""
        for subject in self.subjects:
            subject_dir = base_dir / subject
            
            if not subject_dir.exists():
                continue
                
            try:
                # Load subject's average matrix
                avg_filename = f"{alignment}_{target}_{method}_{freq}_{subject}_average.csv"
                avg_path = subject_dir / avg_filename
                
                if not avg_path.exists():
                    logging.warning(f"Subject average not found: {avg_filename}")
                    continue
                
                matrix = pd.read_csv(avg_path, header=None).values
                
                # Process matrix
                processed_matrix, vmin, vmax = self.process_matrix_for_heatmap(matrix)
                
                # Create heatmap
                plt.figure(figsize=(12, 10))
                sns.heatmap(processed_matrix, 
                           cmap='viridis',
                           vmin=vmin,
                           vmax=vmax,
                           mask=np.diag(np.ones(processed_matrix.shape[0])),  # Mask diagonal
                           cbar_kws={'label': 'Connectivity Value'})
                
                plt.title(f'Subject {subject} Average Connectivity Matrix\n{alignment.upper()} {target} {method} {freq}')
                plt.xlabel('Target Region')
                plt.ylabel('Source Region')
                
                # Save heatmap in subject's directory
                output_filename = f"heatmap_{alignment}_{target}_{method}_{freq}_{subject}.png"
                plt.savefig(subject_dir / output_filename, dpi=300, bbox_inches='tight')
                plt.close()
                
                logging.info(f"Created heatmap for subject {subject}: {output_filename}")
                
            except Exception as e:
                logging.error(f"Error creating heatmap for subject {subject}: {str(e)}")

def main():
    print("Heatmap Generation Program")
    print("=========================")
    
    generator = HeatmapGenerator()
    
    response = input("Do you want to create heatmaps? (y/n): ").lower()
    if response == 'y':
        generator.create_all_heatmaps()
        print("\nHeatmap generation complete. Check heatmap_generation.log for details.")
    else:
        print("Program terminated by user.")

if __name__ == "__main__":
    main() 