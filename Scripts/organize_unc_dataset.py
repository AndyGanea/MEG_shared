#!/usr/bin/env python3

"""
MEG Signal Analysis - UNC Dataset Organizer
-----------------------------------------
This script organizes UNC files from MEG_20250130_group_project_cat into a structured dataset,
excluding iDTF files and following specific naming conventions.

The script:
1. Creates a structured folder hierarchy under Data/DataSet2_unc
2. Copies and renames '_unc' files (excluding iDTF) to appropriate locations
3. Creates subject-specific subfolders
4. Logs all operations
"""

import os
import shutil
from pathlib import Path
import logging
from datetime import datetime
import re

class UncDatasetOrganizer:
    def __init__(self):
        """Initialize the organizer with necessary paths and variables"""
        # Setup paths
        self.source_dir = Path("Data/MEG_20250130_group_project_cat")
        self.target_dir = Path("Data/DataSet2_unc")
        self.logs_dir = Path("Logs")
        
        # Initialize counters
        self.files_processed = 0
        self.files_copied = 0
        self.errors = 0
        
        # Define valid values
        self.alignments = ['cue', 'mov']
        self.postures = ['RT-Pronation', 'RT-Down', 'RT-Upright', 'LT-Pronation']
        self.movements = ['pro', 'anti']
        self.targets = ['L', 'R']
        self.subjects = ['DOC', 'GB', 'JDC', 'JFXD', 'JZ', 'LT', 'NvA', 'RR', 'SJB', 'BG']
        self.methods = ['gDTF', 'iPDC', 'gPDC']  # Excluding iDTF
        self.frequencies = ['10Hz', '20Hz', '100Hz']
        
        # Setup logging
        self.setup_logging()

    def setup_logging(self):
        """Setup logging to both file and console"""
        timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        log_file = self.logs_dir / f"unc_dataset_organization_{timestamp}.txt"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        logging.info("MEG Signal Analysis - UNC Dataset Organization")
        logging.info("============================================")
        logging.info(f"Organization started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info("--------------------------------------------\n")

    def create_folder_structure(self):
        """Create the required folder structure"""
        logging.info("Creating folder structure...")
        
        for alignment in self.alignments:
            for target in self.targets:
                for method in self.methods:
                    for freq in self.frequencies:
                        # Create method_frequency folder
                        method_freq_path = self.target_dir / alignment / target / f"{method}_{freq}"
                        method_freq_path.mkdir(parents=True, exist_ok=True)
                        
                        # Create subject subfolders
                        for subject in self.subjects:
                            subject_path = method_freq_path / subject
                            subject_path.mkdir(exist_ok=True)
        
        logging.info("Folder structure created successfully\n")

    def parse_original_filename(self, filepath: Path) -> dict:
        """Extract components from original filename and path structure"""
        try:
            components = {}
            
            # Get filename parts
            filename = filepath.name
            if not filename.endswith('_unc.csv'):
                return None
                
            # Skip iDTF files
            if 'iDTF' in filename:
                return None
                
            # Parse folder structure (e.g., "Align_mov/RT_Upright_pro_R")
            parent_folder = filepath.parent.name
            align_folder = filepath.parent.parent.name
            
            # Extract alignment (mov/cue)
            components['alignment'] = align_folder.split('_')[1]  # 'mov' from 'Align_mov'
            
            # Parse the movement folder (e.g., "RT_Upright_pro_R")
            folder_parts = parent_folder.split('_')
            
            # Handle posture conversion
            posture_type = folder_parts[0]  # RT or LT
            posture_name = folder_parts[1]  # Pronation, Down, or Upright
            components['posture'] = f"{posture_type}-{posture_name}"
            
            # Extract movement and target
            components['movement'] = folder_parts[2]  # 'pro' or 'anti'
            components['target'] = folder_parts[3]    # 'L' or 'R'
            
            # Parse filename (e.g., "SJB_iPDC_20Hz_alg1_crit1_maxip30_p1_unc.csv")
            name_parts = filename.replace('.csv', '').split('_')
            
            # Extract components from filename
            components['subject'] = name_parts[0]    # 'SJB'
            components['method'] = name_parts[1]     # 'iPDC'
            components['frequency'] = name_parts[2]  # '20Hz'
            
            return components
                
        except Exception as e:
            logging.error(f"Error parsing filename {filepath}: {str(e)}")
            return None

    def generate_new_filename(self, components: dict) -> str:
        """Generate new filename from components"""
        try:
            return f"{components['alignment']}_{components['posture']}_{components['target']}_" \
                   f"{components['movement']}_{components['subject']}_{components['method']}_" \
                   f"{components['frequency']}_unc.csv"
        except Exception as e:
            logging.error(f"Error generating new filename: {str(e)}")
            return None

    def process_files(self):
        """Process all files in the source directory"""
        logging.info("Processing files...")
        
        # First create the folder structure
        self.create_folder_structure()
        
        # Process all CSV files recursively
        for filepath in self.source_dir.rglob("*.csv"):
            self.files_processed += 1
            
            # Parse original filename
            components = self.parse_original_filename(filepath)
            if not components:
                continue
                
            # Generate new filename
            new_filename = self.generate_new_filename(components)
            if not new_filename:
                self.errors += 1
                continue
                
            try:
                # Define target paths
                method_freq_dir = f"{components['method']}_{components['frequency']}"
                
                # Build correct paths for both locations
                method_freq_path = self.target_dir / components['alignment'] / components['target'] / method_freq_dir
                subject_path = method_freq_path / components['subject']
                
                # Ensure directories exist
                method_freq_path.mkdir(parents=True, exist_ok=True)
                subject_path.mkdir(parents=True, exist_ok=True)
                
                # Copy to method_frequency folder
                method_freq_file = method_freq_path / new_filename
                shutil.copy2(filepath, method_freq_file)
                
                # Copy to subject folder
                subject_file = subject_path / new_filename
                shutil.copy2(filepath, subject_file)
                
                self.files_copied += 2  # Counting both copies
                logging.info(f"Processed: {filepath.name}")
                logging.info(f"Created: {method_freq_file}")
                logging.info(f"Created: {subject_file}")
                logging.info("-" * 50)
                
            except Exception as e:
                self.errors += 1
                logging.error(f"Error processing {filepath}: {str(e)}")
                logging.info("-" * 50)

    def print_summary(self):
        """Print summary of operations"""
        logging.info("\nOperation Summary")
        logging.info("================")
        logging.info(f"Total files processed: {self.files_processed}")
        logging.info(f"Total files copied: {self.files_copied}")
        logging.info(f"Errors encountered: {self.errors}")
        logging.info(f"\nTarget directory structure created at: {self.target_dir}")
        logging.info("\nOperation completed at: " + 
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

def main():
    """Main function to execute the organization"""
    organizer = UncDatasetOrganizer()
    organizer.process_files()
    organizer.print_summary()

if __name__ == "__main__":
    main() 