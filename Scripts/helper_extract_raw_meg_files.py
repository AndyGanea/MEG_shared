#!/usr/bin/env python3

"""
MEG Raw Data Extraction Helper
-----------------------------
This script extracts and organizes MEG measurement files from a raw data folder
into a structured directory hierarchy for T-Test analysis. It handles both regular
and 'unc' files, organizing them by alignment, target, method, and frequency.

Key Features:
- Processes either regular or 'unc' files (user selection)
- Organizes files from multiple subjects (DOC, GB, JDC, etc.)
- Handles multiple analysis methods (gDTF, iPDC, gPDC, iDTF)
- Supports different frequencies (10Hz, 20Hz, 100Hz)
- Creates structured directory hierarchy with dual file organization:
  - By method/frequency with all subject files
  - By subject within each method/frequency

Directory Structure Created:
DataSetX_MEG_source_folder[_unc]/
├── [alignment (cue/mov)]/
    ├── [target (L/R)]/
        ├── [method_freq]/
            ├── all subject files
            └── [subject]/
                └── subject-specific files

File Naming Convention:
{alignment}_{movement}_{target}_{condition}_{subject}_{method}_{freq}[_unc].csv
Example: cue_LT-Pronation_L_anti_DOC_gDTF_10Hz.csv
"""

import os
import shutil
from pathlib import Path
import logging
from datetime import datetime
import sys

class FileExtractor:
    def __init__(self):
        """Initialize the File Extractor with predefined configurations."""
        self.subjects = ['DOC', 'GB', 'JDC', 'JFXD', 'JZ', 'LT', 'NvA', 'RR', 'SJB', 'BG']
        self.methods = ['gDTF', 'iPDC', 'gPDC', 'iDTF']
        self.frequencies = ['10Hz', '20Hz', '100Hz']
        
        # Setup directories
        self.data_dir = Path("Data")
        self.logs_dir = Path("Logs")
        self.logs_dir.mkdir(exist_ok=True)
        
        # Get user selections
        self.source_dir = self.select_source_folder()
        self.dataset_number = self.get_dataset_number()
        self.process_unc = self.ask_process_unc()
        
        # Initialize counters
        self.files_processed = 0
        self.errors = 0
        
        # Setup logging
        self.setup_logging()
        
        # Create dataset directory using correct prefix format
        suffix = "_unc" if self.process_unc else ""
        self.dataset_name = f"DataSet{self.dataset_number}_{self.source_dir.name}{suffix}"
        self.dataset_dir = self.data_dir / self.dataset_name

    def select_source_folder(self) -> Path:
        """Let user select raw data source folder."""
        print("\nAvailable data folders:")
        print("=====================")
        
        # Look for data folders in Data directory
        self.data_dir.mkdir(exist_ok=True)
        available_dirs = [d for d in self.data_dir.iterdir() 
                         if d.is_dir()]
        
        if not available_dirs:
            raise ValueError("No folders found in Data directory")
        
        for idx, dir_path in enumerate(available_dirs, 1):
            print(f"{idx}. {dir_path.name}")
        
        while True:
            try:
                choice = int(input("\nSelect folder number: "))
                if 1 <= choice <= len(available_dirs):
                    selected_dir = available_dirs[choice - 1]
                    print(f"\nSelected: {selected_dir.name}")
                    return selected_dir
                else:
                    print("Invalid selection. Please try again.")
            except ValueError:
                print("Please enter a valid number.")

    def get_dataset_number(self) -> int:
        """Get dataset number from user input."""
        while True:
            try:
                number = int(input("\nEnter dataset number: "))
                if number > 0:
                    # Check if dataset already exists
                    existing = list(self.data_dir.glob(f"DataSet{number}_*"))
                    if existing:
                        print(f"Warning: DataSet{number} already exists:")
                        for e in existing:
                            print(f"- {e.name}")
                        if input("Continue anyway? (y/n): ").lower() != 'y':
                            continue
                    return number
                else:
                    print("Please enter a positive number.")
            except ValueError:
                print("Please enter a valid number.")

    def ask_process_unc(self) -> bool:
        """Ask user whether to process 'unc' files."""
        while True:
            response = input("\nProcess 'unc' files? (y/n): ").lower()
            if response in ['y', 'n']:
                return response == 'y'
            print("Please enter 'y' or 'n'")

    def setup_logging(self):
        """Setup logging configuration."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.logs_dir / f"meg_extraction_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        logging.info("MEG Data Extraction")
        logging.info("==================")
        logging.info(f"Source: {self.source_dir.name}")
        logging.info(f"Processing {'unc' if self.process_unc else 'regular'} files")
        logging.info("------------------")

    def create_directory_structure(self):
        """Create the directory structure for organized files."""
        try:
            # Create main dataset directory
            self.dataset_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Created dataset directory: {self.dataset_dir}")
            
            # Create alignment directories (cue/mov)
            for alignment in ['cue', 'mov']:
                align_dir = self.dataset_dir / alignment
                align_dir.mkdir(exist_ok=True)
                
                # Create L/R directories
                for target in ['L', 'R']:
                    target_dir = align_dir / target
                    target_dir.mkdir(exist_ok=True)
                    
                    # Create method_frequency directories
                    for method in self.methods:
                        for freq in self.frequencies:
                            method_freq_dir = target_dir / f"{method}_{freq}"
                            method_freq_dir.mkdir(exist_ok=True)
                            
                            # Create subject directories
                            for subject in self.subjects:
                                subject_dir = method_freq_dir / subject
                                subject_dir.mkdir(exist_ok=True)
            
            logging.info("Directory structure created successfully")
            
        except Exception as e:
            logging.error(f"Error creating directory structure: {str(e)}")
            self.errors += 1

    def process_files(self):
        """Process and organize the raw data files."""
        try:
            # Walk through all subdirectories
            for dirpath, dirnames, filenames in os.walk(self.source_dir):
                current_path = Path(dirpath)
                
                # Debug logging
                logging.info(f"\nProcessing directory: {current_path}")
                
                # Skip if not in a leaf directory (should contain 'anti' or 'pro')
                if not ('anti' in str(current_path).lower() or 'pro' in str(current_path).lower()):
                    logging.info("Skipping non-leaf directory")
                    continue
                
                # Parse path components
                path_str = str(current_path)
                dir_name = current_path.name  # Get the last part of the path
                
                alignment = 'cue' if 'cue' in path_str.lower() else 'mov'
                condition = 'anti' if 'anti' in dir_name.lower() else 'pro'
                
                # Extract target from the last character of directory name
                if dir_name.endswith('_L'):
                    target = 'L'
                elif dir_name.endswith('_R'):
                    target = 'R'
                else:
                    logging.info(f"Skipping directory - cannot determine target direction from name: {dir_name}")
                    continue
                    
                # Extract movement from directory name (e.g., "RT_Down" -> "RT-Down")
                dir_parts = dir_name.split('_')
                if len(dir_parts) >= 2:
                    movement = f"{dir_parts[0]}-{dir_parts[1]}"
                else:
                    logging.info("Skipping directory - invalid movement format")
                    continue
                    
                logging.info(f"Path components: alignment={alignment}, target={target}, "
                           f"condition={condition}, movement={movement}")
                
                # Get all CSV files in current directory
                csv_files = [f for f in filenames if f.endswith('.csv')]
                logging.info(f"CSV files found: {csv_files}")
                
                # Process each CSV file
                for filename in csv_files:
                    # Check if file matches unc selection
                    is_unc = '_unc.csv' in filename
                    if is_unc != self.process_unc:
                        logging.info(f"Skipping file (unc mismatch): {filename}")
                        continue
                    
                    try:
                        file_path = current_path / filename
                        
                        # Parse filename components
                        name_parts = filename.replace('_unc.csv', '.csv').replace('.csv', '').split('_')
                        subject = name_parts[0]
                        method = name_parts[1]
                        freq = name_parts[2]
                        
                        logging.info(f"File components: subject={subject}, method={method}, freq={freq}")
                        
                        if subject in self.subjects and method in self.methods:
                            # Create new filename
                            new_filename = f"{alignment}_{movement}_{target}_{condition}_{subject}_{method}_{freq}"
                            if self.process_unc:
                                new_filename += "_unc"
                            new_filename += ".csv"
                            
                            # Create destination paths and ensure directories exist
                            method_freq_dir = self.dataset_dir / alignment / target / f"{method}_{freq}"
                            subject_dir = method_freq_dir / subject
                            
                            # Create directories if they don't exist
                            method_freq_dir.mkdir(parents=True, exist_ok=True)
                            subject_dir.mkdir(parents=True, exist_ok=True)
                            
                            # Copy files
                            dest_file = method_freq_dir / new_filename
                            subj_dest_file = subject_dir / new_filename
                            
                            logging.info(f"Copying to:")
                            logging.info(f"- {dest_file}")
                            logging.info(f"- {subj_dest_file}")
                            
                            # Ensure source file exists and is readable
                            if not file_path.exists():
                                logging.error(f"Source file does not exist: {file_path}")
                                continue
                                
                            # Copy with explicit error handling
                            try:
                                shutil.copy2(str(file_path), str(dest_file))
                                shutil.copy2(str(file_path), str(subj_dest_file))
                                self.files_processed += 2
                                logging.info(f"Successfully copied: {filename} -> {new_filename}")
                            except Exception as e:
                                logging.error(f"Error copying file {filename}: {str(e)}")
                                self.errors += 1
                                
                        else:
                            logging.info(f"Skipping file (invalid subject/method): {filename}")
                            
                    except Exception as e:
                        logging.error(f"Error processing file {filename}: {str(e)}")
                        self.errors += 1
                        
            logging.info(f"\nTotal files processed: {self.files_processed}")
            
        except Exception as e:
            logging.error(f"Error in file processing: {str(e)}")
            self.errors += 1

    def print_summary(self):
        """Print summary of operations."""
        logging.info("\nOperation Summary")
        logging.info("================")
        logging.info(f"Files processed: {self.files_processed}")
        logging.info(f"Errors encountered: {self.errors}")
        logging.info(f"Process completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def main():
    """Main execution function."""
    extractor = FileExtractor()
    
    print("\nMEG Data Extraction")
    print("=================")
    print(f"Creating: {extractor.dataset_name}")
    print("-------------------------")
    
    # Execute steps
    extractor.create_directory_structure()
    extractor.process_files()
    extractor.print_summary()

if __name__ == "__main__":
    main() 