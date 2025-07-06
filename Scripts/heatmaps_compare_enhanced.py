#!/usr/bin/env python3

"""
MEG Signal Analysis - Enhanced Heatmaps Comparison
-----------------------------------------------
This script creates comparison heatmaps for average matrices, with dynamic folder selection.
It processes average files in the selected dataset folder and creates normalized
3-panel comparisons showing left target, right target, and their difference.

Output is stored in Heatmaps/<selected folder>/Heatmaps_Comparison_normalized/<timestamp>
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from datetime import datetime
import sys
import os

class HeatmapsComparer:
    def __init__(self):
        """Initialize the comparer with necessary paths and settings"""
        self.data_dir = Path("Data")
        self.logs_dir = Path("Logs")
        
        # Get user selection for dataset folder
        self.dataset_dir = self.select_dataset_folder()
        
        # Setup output directory (without timestamp)
        self.output_dir = Path("Heatmaps") / self.dataset_dir.name / "Heatmaps_Comparison_normalized"
        
        # Initialize counters
        self.comparisons_created = 0
        self.errors = 0
        
        # Setup logging
        self.setup_logging()

    def select_dataset_folder(self) -> Path:
        """Present available folders and let user select one"""
        print("\nAvailable datasets under Data folder:")
        print("=====================================")
        
        # Get all directories under Data
        available_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
        
        # Present options to user
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

    def setup_logging(self):
        """Setup logging to both file and console"""
        timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        log_file = self.logs_dir / f"heatmaps_comparison_{timestamp}.txt"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        logging.info("MEG Signal Analysis - Enhanced Heatmaps Comparison")
        logging.info("===============================================")
        logging.info(f"Selected dataset: {self.dataset_dir.name}")
        logging.info(f"Output directory: {self.output_dir}")
        logging.info(f"Process started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info("-----------------------------------------------\n")

    def find_global_max(self):
        """Find the maximum value across all average files"""
        max_value = 0
        
        # Process each alignment (cue/mov)
        for align_dir in self.dataset_dir.iterdir():
            if align_dir.is_dir():
                # Process each target (L/R)
                for target in ['L', 'R']:
                    target_dir = align_dir / target
                    if not target_dir.is_dir():
                        continue
                    
                    # Process each method_frequency folder
                    for method_freq_dir in target_dir.iterdir():
                        if not method_freq_dir.is_dir():
                            continue
                        
                        # Read all average files (both method-level and subject-level)
                        avg_files = list(method_freq_dir.glob("*average.csv"))
                        
                        for avg_file in avg_files:
                            try:
                                matrix = pd.read_csv(avg_file, header=None).values
                                # Mask diagonal before finding max
                                np.fill_diagonal(matrix, 0)
                                max_value = max(max_value, np.max(matrix))
                            except Exception as e:
                                logging.error(f"Error reading {avg_file}: {str(e)}")
        
        return max_value

    def create_comparison_heatmap(self, left_matrix, right_matrix, title, output_path, is_subject=False, global_max=None):
        """Create a three-panel comparison heatmap with improved formatting"""
        try:
            # Create copies of matrices to avoid modifying originals
            left_matrix = left_matrix.copy()
            right_matrix = right_matrix.copy()
            
            # Mask diagonal elements
            np.fill_diagonal(left_matrix, np.nan)
            np.fill_diagonal(right_matrix, np.nan)
            
            # Calculate difference matrix
            diff_matrix = left_matrix - right_matrix
            
            # Calculate max difference for scaling
            diff_max = np.nanmax(np.abs(diff_matrix))
            
            # Create figure with three subplots
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
            
            # Create heatmaps with improved formatting
            # Left target
            sns.heatmap(left_matrix, ax=ax1, cmap='Reds', 
                       vmin=0, vmax=global_max,
                       mask=np.isnan(left_matrix), square=True,
                       xticklabels=True, yticklabels=True)
            ax1.set_title(f'Left Target\nMax: {global_max:.4f}', fontsize=12, pad=10)
            ax1.set_xlabel('Target Region', fontsize=10)
            ax1.set_ylabel('Source Region', fontsize=10)
            
            # Right target
            sns.heatmap(right_matrix, ax=ax2, cmap='Reds', 
                       vmin=0, vmax=global_max,
                       mask=np.isnan(right_matrix), square=True,
                       xticklabels=True, yticklabels=True)
            ax2.set_title(f'Right Target\nMax: {global_max:.4f}', fontsize=12, pad=10)
            ax2.set_xlabel('Target Region', fontsize=10)
            ax2.set_ylabel('Source Region', fontsize=10)
            
            # Difference plot with dynamic scale based on actual differences
            sns.heatmap(diff_matrix, ax=ax3, cmap='RdBu_r', 
                       vmin=-diff_max, vmax=diff_max, center=0,
                       mask=np.isnan(diff_matrix), square=True,
                       xticklabels=True, yticklabels=True)
            ax3.set_title(f'Difference (L-R)\nRange: ±{diff_max:.4f}', fontsize=12, pad=10)
            ax3.set_xlabel('Target Region', fontsize=10)
            ax3.set_ylabel('Source Region', fontsize=10)
            
            # Set main title
            if is_subject:
                parts = title.split('_')
                main_title = f"{parts[1]} {parts[2]}\n{parts[0].upper()} - Subject {parts[3]}"
            else:
                parts = title.split('_')
                main_title = f"{parts[1]} {parts[2]}\n{parts[0].upper()} - Group Average"
            
            plt.suptitle(main_title, fontsize=14, y=1.05)
            
            # Adjust layout and save
            plt.tight_layout()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            plt.close()
            
            logging.info(f"Created heatmap: {output_path}")
            self.comparisons_created += 1
            
        except Exception as e:
            self.errors += 1
            logging.error(f"Error creating heatmap {title}: {str(e)}")

    def process_averages(self):
        """Process average files and create comparison heatmaps"""
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find global maximum value across all average files
        global_max = self.find_global_max()
        logging.info(f"Global maximum value: {global_max}")
        
        # Process each alignment (cue/mov)
        for align_dir in self.dataset_dir.iterdir():
            if align_dir.is_dir():
                logging.info(f"\nProcessing alignment: {align_dir.name}")
                
                # Process each method_frequency combination
                for method_freq in ['gDTF_10Hz', 'gDTF_20Hz', 'gDTF_100Hz',
                                  'iPDC_10Hz', 'iPDC_20Hz', 'iPDC_100Hz',
                                  'gPDC_10Hz', 'gPDC_20Hz', 'gPDC_100Hz']:
                    
                    try:
                        # 1. Process method-level averages
                        left_avg = list((align_dir / 'L' / method_freq).glob("*average.csv"))
                        right_avg = list((align_dir / 'R' / method_freq).glob("*average.csv"))
                        
                        if left_avg and right_avg:
                            left_matrix = pd.read_csv(left_avg[0], header=None).values
                            right_matrix = pd.read_csv(right_avg[0], header=None).values
                            
                            title = f"{align_dir.name}_{method_freq}"
                            output_name = f"{title}_comparison_normalized.png"
                            self.create_comparison_heatmap(
                                left_matrix, right_matrix, title,
                                self.output_dir / output_name,
                                is_subject=False,
                                global_max=global_max
                            )
                        
                        # 2. Process subject-level averages
                        for subject in ['DOC', 'GB', 'JDC', 'JFXD', 'JZ', 'LT', 'NvA', 'RR', 'SJB', 'BG']:
                            left_subj = list((align_dir / 'L' / method_freq / subject).glob("*average.csv"))
                            right_subj = list((align_dir / 'R' / method_freq / subject).glob("*average.csv"))
                            
                            if left_subj and right_subj:
                                left_matrix = pd.read_csv(left_subj[0], header=None).values
                                right_matrix = pd.read_csv(right_subj[0], header=None).values
                                
                                title = f"{align_dir.name}_{method_freq}_{subject}"
                                output_name = f"{title}_comparison_normalized.png"
                                self.create_comparison_heatmap(
                                    left_matrix, right_matrix, title,
                                    self.output_dir / output_name,
                                    is_subject=True,
                                    global_max=global_max
                                )
                                
                    except Exception as e:
                        self.errors += 1
                        logging.error(f"Error processing {method_freq}: {str(e)}")

    def get_subjects(self, directory):
        """Get list of subjects from a directory"""
        return [d.name for d in directory.iterdir() if d.is_dir()]

    def print_summary(self):
        """Print summary of operations"""
        logging.info("\nOperation Summary")
        logging.info("================")
        logging.info(f"Total comparisons created: {self.comparisons_created}")
        logging.info(f"Errors encountered: {self.errors}")
        logging.info(f"\nOutput directory: {self.output_dir}")
        logging.info(f"Process completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def main():
    """Main function to execute the comparison"""
    comparer = HeatmapsComparer()
    comparer.process_averages()
    comparer.print_summary()

if __name__ == "__main__":
    main() 