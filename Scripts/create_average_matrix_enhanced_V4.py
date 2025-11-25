#!/usr/bin/env python3

"""
MEG Signal Analysis - Enhanced Average Matrix Creator (Unified)
--------------------------------------------------------------
This script creates average matrices for MEG data, with dynamic folder selection.

Supported directory structures
------------------------------
1) Classic L/R structure (e.g., cue datasets):
   Data/
     DataSetXX_Align_cue/
       cue/
         L/
           <method_freq>/
             <subject>/
               *.csv
         R/
           <method_freq>/
             <subject>/
               *.csv

2) New Left_movement/Right_movement structure (e.g., mov datasets):
   Data/
     DataSetYY_Align_mov/
       mov/
         Left_movement/
           <method_freq>/
             <subject>/
               *.csv
         Right_movement/
           <method_freq>/
             <subject>/
               *.csv

What it computes
----------------
For each alignment (cue/mov) and each method_freq:

- Classic L/R structure:
  1. Overall averages for each (target, method_freq) combination (L, R)
  2. Subject-specific averages per target
  3. Condition-specific averages (Pro-only and Anti-only) per target (L/R),
     at both the overall and subject levels.

- Left_movement/Right_movement structure:
  1. Overall averages for each method_freq (across Left_movement + Right_movement)
  2. Subject-specific averages (across both movement dirs)
  3. Condition-specific averages (Pro-only and Anti-only) at
     both the overall and subject levels.

NaN values in input matrices are replaced with 0 before averaging.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
import sys
import os

class AverageMatrixCreator:
    def __init__(self):
        """Initialize the creator with necessary paths and variables"""
        self.data_dir = Path("Data")
        self.logs_dir = Path("Logs")
        
        # Get user selection for dataset folder
        self.dataset_dir = self.select_dataset_folder()
        
        # Ask about excluding iDTF
        self.exclude_idtf = self.ask_exclude_idtf()
        
        # Ask about excluding LT-Pronation files
        self.exclude_lt_pronation = self.ask_exclude_lt_pronation()
        
        # Initialize counters
        self.matrices_processed = 0
        self.averages_created = 0
        self.errors = 0
        self.total_nan_count = 0
        self.lt_pronation_files_excluded = 0  # New counter for excluded files
        
        # Setup logging
        self.setup_logging()

    # -------------------------------------------------------------------------
    # Setup / utilities
    # -------------------------------------------------------------------------

    def ask_exclude_idtf(self) -> bool:
        """Ask user whether to exclude iDTF method"""
        while True:
            response = input("\nExclude iDTF method (values may be >1)? (y/n): ").lower()
            if response in ['y', 'n']:
                return response == 'y'
            print("Please enter 'y' or 'n'")

    def ask_exclude_lt_pronation(self) -> bool:
        """Ask user whether to exclude LT-Pronation files"""
        while True:
            response = input("\nExclude LT-Pronation files? (y/n): ").lower()
            if response in ['y', 'n']:
                return response == 'y'
            print("Please enter 'y' or 'n'")

    def select_dataset_folder(self) -> Path:
        """Present available folders and let user select one"""
        print("\nAvailable datasets:")
        print("================")
        
        # Get all directories under Data that start with DataSet
        available_dirs = [d for d in self.data_dir.iterdir() 
                         if d.is_dir() and d.name.startswith("DataSet")]
        
        if not available_dirs:
            raise ValueError("No DataSet folders found in Data directory")
        
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
        log_file = self.logs_dir / f"average_matrix_creation_{timestamp}.txt"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        logging.info("MEG Signal Analysis - Enhanced Average Matrix Creation (Unified)")
        logging.info("===============================================================")
        logging.info(f"Selected dataset: {self.dataset_dir.name}")
        logging.info(f"Excluding iDTF: {self.exclude_idtf}")
        logging.info(f"Excluding LT-Pronation: {self.exclude_lt_pronation}")
        logging.info(f"Process started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info("--------------------------------------------------\n")

    def is_lt_pronation_file(self, filename: str) -> bool:
        """Check if a file contains 'LT-Pronation' in its name"""
        return "LT-Pronation" in filename

    def should_include_file(self, filename: str) -> bool:
        """Determine if a file should be included based on user preferences"""
        if self.exclude_lt_pronation and self.is_lt_pronation_file(filename):
            return False
        return True

    def get_condition_from_filename(self, filename: str):
        """
        Infer condition ('pro' or 'anti') from filename using underscore-separated structure.

        Expected pattern (example):
        mov_structure-Down_L_pro_BG_msc_cat_100Hz.csv
          0         1             2  3    4   5    6    7
        -> ['mov', 'structure-Down', 'L', 'pro', 'BG', 'msc', 'cat', '100Hz.csv']

        Returns:
            'pro', 'anti', or None if not identifiable.
        """
        name = Path(filename).name  # strip path
        parts = name.split('_')

        # Need at least: [something, something, side, condition, ...]
        if len(parts) < 4:
            return None

        cond = parts[3].lower()
        if cond in ("pro", "anti"):
            return cond
        return None

    def check_lt_pronation_files_exist(self) -> bool:
        """Check if any LT-Pronation files exist in the dataset"""
        logging.info("Checking for LT-Pronation files in dataset...")
        lt_pronation_found = False
        
        for align_dir in self.dataset_dir.iterdir():
            if align_dir.is_dir():
                for child_dir in align_dir.iterdir():
                    if child_dir.is_dir():
                        for method_freq_dir in child_dir.iterdir():
                            if method_freq_dir.is_dir():
                                # Check files in method_freq directory
                                for file in method_freq_dir.glob("*.csv"):
                                    if self.is_lt_pronation_file(file.name):
                                        lt_pronation_found = True
                                        logging.info(f"Found LT-Pronation file: {file.name}")
                                
                                # Check files in subject directories
                                for subject_dir in method_freq_dir.iterdir():
                                    if subject_dir.is_dir():
                                        for file in subject_dir.glob("*.csv"):
                                            if self.is_lt_pronation_file(file.name):
                                                lt_pronation_found = True
                                                logging.info(f"Found LT-Pronation file: {file.name}")
        
        if not lt_pronation_found:
            logging.info("No LT-Pronation files found in the dataset")
        
        return lt_pronation_found

    def create_average_matrix(self, matrices):
        """Create average matrix from a list of matrices"""
        if not matrices:
            return None
        
        # Convert all matrices to numpy arrays and handle NaN values
        numpy_matrices = []
        for m in matrices:
            # Convert to numpy array
            matrix = m.values if isinstance(m, pd.DataFrame) else m
            
            # Log NaN statistics
            nan_count = np.isnan(matrix).sum()
            if nan_count > 0:
                self.total_nan_count += nan_count
                logging.info(f"Found {nan_count} NaN values in matrix - replacing with 0")
            
            # Replace NaN with 0
            matrix = np.nan_to_num(matrix, nan=0.0)
            numpy_matrices.append(matrix)
        
        # Stack matrices and calculate mean
        stacked = np.stack(numpy_matrices)
        return np.mean(stacked, axis=0)

    def save_average_matrix(self, matrix, filepath):
        """Save average matrix to CSV file"""
        # Always overwrite existing files
        pd.DataFrame(matrix).to_csv(filepath, index=False, header=False, mode='w')
        logging.info(f"Created/Overwrote average matrix: {filepath}")
        self.averages_created += 1

    # -------------------------------------------------------------------------
    # Core processing
    # -------------------------------------------------------------------------

    def process_dataset(self):
        """Process the entire dataset structure, handling both L/R and movement layouts."""
        logging.info("Processing dataset...")
        
        # Check for LT-Pronation files if user chose to exclude them
        if self.exclude_lt_pronation:
            lt_pronation_exists = self.check_lt_pronation_files_exist()
            if not lt_pronation_exists:
                logging.warning("WARNING: You chose to exclude LT-Pronation files, but no such files were found in the dataset.")
                logging.warning("Files will still be created with '_NO-LT' suffix as requested.")
        
        # Process each alignment (e.g., 'cue', 'mov')
        for align_dir in self.dataset_dir.iterdir():
            if not align_dir.is_dir():
                continue

            alignment = align_dir.name
            child_dirs = [d for d in align_dir.iterdir() if d.is_dir()]
            child_names = {d.name for d in child_dirs}

            logging.info(f"\nProcessing alignment: {alignment}")

            # Detect structure type
            if any(name in ('L', 'R') for name in child_names):
                logging.info(f"Detected L/R structure under alignment '{alignment}'")
                self._process_alignment_lr(align_dir, alignment)
            elif any(name in ('Left_movement', 'Right_movement') for name in child_names):
                logging.info(f"Detected Left_movement/Right_movement structure under alignment '{alignment}'")
                self._process_alignment_movement(align_dir, alignment)
            else:
                logging.warning(f"Unknown directory structure under alignment '{alignment}'. "
                                f"Subdirs: {sorted(child_names)}")

    # -------------------------------------------------------------------------
    # L/R structure branch (classic)
    # -------------------------------------------------------------------------

    def _process_alignment_lr(self, align_dir: Path, alignment: str):
        """Process an alignment with classic L/R structure."""
        # Collect method_freq directories keyed by (target, method_freq)
        method_freq_dirs = {}  # key: (target, method_freq), value: list of directories

        for target_dir in align_dir.iterdir():
            if target_dir.is_dir():
                target = target_dir.name  # 'L' or 'R'
                logging.info(f"Processing target: {target}")
                
                for method_freq_dir in target_dir.iterdir():
                    if method_freq_dir.is_dir():
                        key = (target, method_freq_dir.name)  # e.g., ('R', 'gDTF_10Hz')
                        if key not in method_freq_dirs:
                            method_freq_dirs[key] = []
                        method_freq_dirs[key].append(method_freq_dir)
                        logging.info(f"Found method_freq directory: {method_freq_dir}")

        # Process each (target, method_freq)
        for (target, method_freq), dirs in method_freq_dirs.items():
            parts = method_freq.split('_')
            if len(parts) < 2:
                logging.warning(f"Unexpected method_freq name: {method_freq}")
                continue
            method = '_'.join(parts[:-1])   # e.g., 'gDTF' or 'msc_mean'
            freq   = parts[-1]              # e.g., '100Hz'
            
            # Skip iDTF if requested
            if self.exclude_idtf and method == 'iDTF':
                logging.info(f"Skipping iDTF method as requested")
                continue
            
            logging.info(f"\nProcessing {target} {method_freq}")
            
            # Debug: list files
            for dir_path in dirs:
                logging.info(f"Looking for files in: {dir_path}")
                all_files = list(dir_path.glob("*.csv"))
                logging.info(f"Found {len(all_files)} files")
                for file in all_files:
                    logging.info(f"Found file: {file.name}")
            
            # Collect matrices: ALL, Pro-only, Anti-only
            matrices_all = []
            matrices_pro = []
            matrices_anti = []
            excluded_files = []

            has_unc_all = False
            has_unc_pro = False
            has_unc_anti = False

            for dir_path in dirs:
                for file in dir_path.glob("*.csv"):
                    if 'average' in file.name:
                        continue

                    if not self.should_include_file(file.name):
                        excluded_files.append(file.name)
                        logging.info(f"Excluding LT-Pronation file: {file.name}")
                        continue
                    
                    try:
                        matrix = pd.read_csv(file, header=None)
                        self.matrices_processed += 1
                        logging.info(f"Successfully read matrix from: {file.name}")

                        if "_unc" in file.name:
                            has_unc_all = True

                        matrices_all.append(matrix)

                        cond = self.get_condition_from_filename(file.name)
                        if cond == "pro":
                            matrices_pro.append(matrix)
                            if "_unc" in file.name:
                                has_unc_pro = True
                        elif cond == "anti":
                            matrices_anti.append(matrix)
                            if "_unc" in file.name:
                                has_unc_anti = True

                    except Exception as e:
                        self.errors += 1
                        logging.error(f"Error reading {file}: {str(e)}")

            # Log exclusion summary
            if self.exclude_lt_pronation:
                if excluded_files:
                    logging.info(f"Excluded {len(excluded_files)} LT-Pronation files from averaging")
                    self.lt_pronation_files_excluded += len(excluded_files)
                else:
                    logging.info("No LT-Pronation files found in this directory")

            # ALL-trial overall average
            if matrices_all:
                avg_matrix_all = self.create_average_matrix(matrices_all)
                if avg_matrix_all is not None:
                    avg_filename = f"{alignment}_{target}_{method}_{freq}_average"
                    if has_unc_all:
                        avg_filename += "_unc"
                    if self.exclude_lt_pronation:
                        avg_filename += "_NO-LT"
                    avg_filename += ".csv"
                    
                    for dir_path in dirs:
                        avg_filepath = dir_path / avg_filename
                        self.save_average_matrix(avg_matrix_all, avg_filepath)
            else:
                logging.warning(f"No matrices found for {target} {method_freq}")

            # Pro-only overall average
            if matrices_pro:
                avg_matrix_pro = self.create_average_matrix(matrices_pro)
                if avg_matrix_pro is not None:
                    avg_filename_pro = f"{alignment}_{target}_{method}_{freq}_Pro_average"
                    if has_unc_pro:
                        avg_filename_pro += "_unc"
                    if self.exclude_lt_pronation:
                        avg_filename_pro += "_NO-LT"
                    avg_filename_pro += ".csv"

                    for dir_path in dirs:
                        avg_filepath_pro = dir_path / avg_filename_pro
                        self.save_average_matrix(avg_matrix_pro, avg_filepath_pro)
            else:
                logging.info(f"No Pro-condition matrices found for {target} {method_freq}")

            # Anti-only overall average
            if matrices_anti:
                avg_matrix_anti = self.create_average_matrix(matrices_anti)
                if avg_matrix_anti is not None:
                    avg_filename_anti = f"{alignment}_{target}_{method}_{freq}_Anti_average"
                    if has_unc_anti:
                        avg_filename_anti += "_unc"
                    if self.exclude_lt_pronation:
                        avg_filename_anti += "_NO-LT"
                    avg_filename_anti += ".csv"

                    for dir_path in dirs:
                        avg_filepath_anti = dir_path / avg_filename_anti
                        self.save_average_matrix(avg_matrix_anti, avg_filepath_anti)
            else:
                logging.info(f"No Anti-condition matrices found for {target} {method_freq}")

            # Subject-level averages
            for dir_path in dirs:
                for subject_dir in dir_path.iterdir():
                    if not subject_dir.is_dir():
                        continue

                    subject = subject_dir.name
                    logging.info(f"\nProcessing subject: {subject}")

                    matrices_all = []
                    matrices_pro = []
                    matrices_anti = []
                    excluded_files = []

                    has_unc_all_subj = False
                    has_unc_pro_subj = False
                    has_unc_anti_subj = False

                    for file in subject_dir.glob("*.csv"):
                        if 'average' in file.name:
                            continue

                        if not self.should_include_file(file.name):
                            excluded_files.append(file.name)
                            logging.info(f"Excluding LT-Pronation file: {file.name}")
                            continue
                        
                        try:
                            matrix = pd.read_csv(file, header=None)
                            self.matrices_processed += 1
                            logging.info(f"Successfully read matrix from: {file.name}")

                            if "_unc" in file.name:
                                has_unc_all_subj = True

                            matrices_all.append(matrix)

                            cond = self.get_condition_from_filename(file.name)
                            if cond == "pro":
                                matrices_pro.append(matrix)
                                if "_unc" in file.name:
                                    has_unc_pro_subj = True
                            elif cond == "anti":
                                matrices_anti.append(matrix)
                                if "_unc" in file.name:
                                    has_unc_anti_subj = True

                        except Exception as e:
                            self.errors += 1
                            logging.error(f"Error reading {file}: {str(e)}")

                    if self.exclude_lt_pronation:
                        if excluded_files:
                            logging.info(f"Excluded {len(excluded_files)} LT-Pronation files from subject {subject}")
                            self.lt_pronation_files_excluded += len(excluded_files)
                        else:
                            logging.info(f"No LT-Pronation files found for subject {subject}")

                    # ALL-trial subject average
                    if matrices_all:
                        avg_matrix_all = self.create_average_matrix(matrices_all)
                        if avg_matrix_all is not None:
                            avg_filename = f"{alignment}_{target}_{method}_{freq}_{subject}_average"
                            if has_unc_all_subj:
                                avg_filename += "_unc"
                            if self.exclude_lt_pronation:
                                avg_filename += "_NO-LT"
                            avg_filename += ".csv"
                            avg_filepath = subject_dir / avg_filename
                            self.save_average_matrix(avg_matrix_all, avg_filepath)
                    else:
                        logging.warning(f"No matrices found for subject {subject}")

                    # Pro-only subject average
                    if matrices_pro:
                        avg_matrix_pro = self.create_average_matrix(matrices_pro)
                        if avg_matrix_pro is not None:
                            avg_filename_pro = f"{alignment}_{target}_{method}_{freq}_{subject}_Pro_average"
                            if has_unc_pro_subj:
                                avg_filename_pro += "_unc"
                            if self.exclude_lt_pronation:
                                avg_filename_pro += "_NO-LT"
                            avg_filename_pro += ".csv"
                            avg_filepath_pro = subject_dir / avg_filename_pro
                            self.save_average_matrix(avg_matrix_pro, avg_filepath_pro)

                    # Anti-only subject average
                    if matrices_anti:
                        avg_matrix_anti = self.create_average_matrix(matrices_anti)
                        if avg_matrix_anti is not None:
                            avg_filename_anti = f"{alignment}_{target}_{method}_{freq}_{subject}_Anti_average"
                            if has_unc_anti_subj:
                                avg_filename_anti += "_unc"
                            if self.exclude_lt_pronation:
                                avg_filename_anti += "_NO-LT"
                            avg_filename_anti += ".csv"
                            avg_filepath_anti = subject_dir / avg_filename_anti
                            self.save_average_matrix(avg_matrix_anti, avg_filepath_anti)

    # -------------------------------------------------------------------------
    # Left_movement / Right_movement branch
    # -------------------------------------------------------------------------

    def _process_alignment_movement(self, align_dir: Path, alignment: str):
        """
        Process an alignment with Left_movement / Right_movement structure.

        Here we aggregate over both movement directions for each method_freq:
        - Overall ALL / Pro / Anti averages
        - Subject-level ALL / Pro / Anti averages
        """
        # Collect method_freq directories keyed by method_freq name
        method_freq_dirs = {}  # key: method_freq, value: list of directories

        for movement_dir in align_dir.iterdir():
            if movement_dir.is_dir():
                movement_name = movement_dir.name  # 'Left_movement' or 'Right_movement'
                logging.info(f"Processing movement: {movement_name}")
                
                for method_freq_dir in movement_dir.iterdir():
                    if method_freq_dir.is_dir():
                        key = method_freq_dir.name  # e.g., 'gDTF_10Hz'
                        if key not in method_freq_dirs:
                            method_freq_dirs[key] = []
                        method_freq_dirs[key].append(method_freq_dir)
                        logging.info(f"Found method_freq directory: {method_freq_dir}")

        # Process each method_freq across both movement dirs
        for method_freq, dirs in method_freq_dirs.items():
            parts = method_freq.split('_')
            if len(parts) < 2:
                logging.warning(f"Unexpected method_freq name: {method_freq}")
                continue
            method = '_'.join(parts[:-1])   # e.g., 'gDTF' or 'msc_mean'
            freq   = parts[-1]              # e.g., '100Hz'
            
            # Skip iDTF if requested
            if self.exclude_idtf and method == 'iDTF':
                logging.info(f"Skipping iDTF method as requested")
                continue
            
            logging.info(f"\nProcessing {method_freq} across Left_movement + Right_movement")
            
            # Debug: list files
            for dir_path in dirs:
                logging.info(f"Looking for files in: {dir_path}")
                all_files = list(dir_path.glob("*.csv"))
                logging.info(f"Found {len(all_files)} files")
                for file in all_files:
                    logging.info(f"Found file: {file.name}")
            
            # Collect matrices: ALL, Pro-only, Anti-only (across all movement dirs)
            matrices_all = []
            matrices_pro = []
            matrices_anti = []
            excluded_files = []

            has_unc_all = False
            has_unc_pro = False
            has_unc_anti = False

            for dir_path in dirs:
                for file in dir_path.glob("*.csv"):
                    if 'average' in file.name:
                        continue

                    if not self.should_include_file(file.name):
                        excluded_files.append(file.name)
                        logging.info(f"Excluding LT-Pronation file: {file.name}")
                        continue
                    
                    try:
                        matrix = pd.read_csv(file, header=None)
                        self.matrices_processed += 1
                        logging.info(f"Successfully read matrix from: {file.name}")

                        if "_unc" in file.name:
                            has_unc_all = True

                        matrices_all.append(matrix)

                        cond = self.get_condition_from_filename(file.name)
                        if cond == "pro":
                            matrices_pro.append(matrix)
                            if "_unc" in file.name:
                                has_unc_pro = True
                        elif cond == "anti":
                            matrices_anti.append(matrix)
                            if "_unc" in file.name:
                                has_unc_anti = True

                    except Exception as e:
                        self.errors += 1
                        logging.error(f"Error reading {file}: {str(e)}")

            # Log exclusion summary
            if self.exclude_lt_pronation:
                if excluded_files:
                    logging.info(f"Excluded {len(excluded_files)} LT-Pronation files from averaging")
                    self.lt_pronation_files_excluded += len(excluded_files)
                else:
                    logging.info("No LT-Pronation files found in this directory")

            # ALL-trial overall average (across both movement dirs)
            if matrices_all:
                avg_matrix_all = self.create_average_matrix(matrices_all)
                if avg_matrix_all is not None:
                    avg_filename = f"{alignment}_{method}_{freq}_average"
                    if has_unc_all:
                        avg_filename += "_unc"
                    if self.exclude_lt_pronation:
                        avg_filename += "_NO-LT"
                    avg_filename += ".csv"
                    
                    for dir_path in dirs:
                        avg_filepath = dir_path / avg_filename
                        self.save_average_matrix(avg_matrix_all, avg_filepath)
            else:
                logging.warning(f"No matrices found for {method_freq}")

            # Pro-only overall average
            if matrices_pro:
                avg_matrix_pro = self.create_average_matrix(matrices_pro)
                if avg_matrix_pro is not None:
                    avg_filename_pro = f"{alignment}_{method}_{freq}_Pro_average"
                    if has_unc_pro:
                        avg_filename_pro += "_unc"
                    if self.exclude_lt_pronation:
                        avg_filename_pro += "_NO-LT"
                    avg_filename_pro += ".csv"

                    for dir_path in dirs:
                        avg_filepath_pro = dir_path / avg_filename_pro
                        self.save_average_matrix(avg_matrix_pro, avg_filepath_pro)
            else:
                logging.info(f"No Pro-condition matrices found for {method_freq}")

            # Anti-only overall average
            if matrices_anti:
                avg_matrix_anti = self.create_average_matrix(matrices_anti)
                if avg_matrix_anti is not None:
                    avg_filename_anti = f"{alignment}_{method}_{freq}_Anti_average"
                    if has_unc_anti:
                        avg_filename_anti += "_unc"
                    if self.exclude_lt_pronation:
                        avg_filename_anti += "_NO-LT"
                    avg_filename_anti += ".csv"

                    for dir_path in dirs:
                        avg_filepath_anti = dir_path / avg_filename_anti
                        self.save_average_matrix(avg_matrix_anti, avg_filepath_anti)
            else:
                logging.info(f"No Anti-condition matrices found for {method_freq}")

            # Subject-level averages (across both movement dirs)
            # We treat subject names from *all* movement dirs consistently
            subject_dirs_by_name = {}

            for dir_path in dirs:
                for subject_dir in dir_path.iterdir():
                    if subject_dir.is_dir():
                        subj_name = subject_dir.name
                        subject_dirs_by_name.setdefault(subj_name, []).append(subject_dir)

            for subject, subj_dir_list in subject_dirs_by_name.items():
                logging.info(f"\nProcessing subject: {subject} (across movement dirs)")

                matrices_all = []
                matrices_pro = []
                matrices_anti = []
                excluded_files = []

                has_unc_all_subj = False
                has_unc_pro_subj = False
                has_unc_anti_subj = False

                for subject_dir in subj_dir_list:
                    for file in subject_dir.glob("*.csv"):
                        if 'average' in file.name:
                            continue

                        if not self.should_include_file(file.name):
                            excluded_files.append(file.name)
                            logging.info(f"Excluding LT-Pronation file: {file.name}")
                            continue
                        
                        try:
                            matrix = pd.read_csv(file, header=None)
                            self.matrices_processed += 1
                            logging.info(f"Successfully read matrix from: {file.name}")

                            if "_unc" in file.name:
                                has_unc_all_subj = True

                            matrices_all.append(matrix)

                            cond = self.get_condition_from_filename(file.name)
                            if cond == "pro":
                                matrices_pro.append(matrix)
                                if "_unc" in file.name:
                                    has_unc_pro_subj = True
                            elif cond == "anti":
                                matrices_anti.append(matrix)
                                if "_unc" in file.name:
                                    has_unc_anti_subj = True

                        except Exception as e:
                            self.errors += 1
                            logging.error(f"Error reading {file}: {str(e)}")

                if self.exclude_lt_pronation:
                    if excluded_files:
                        logging.info(f"Excluded {len(excluded_files)} LT-Pronation files from subject {subject}")
                        self.lt_pronation_files_excluded += len(excluded_files)
                    else:
                        logging.info(f"No LT-Pronation files found for subject {subject}")

                # ALL-trial subject average
                if matrices_all:
                    avg_matrix_all = self.create_average_matrix(matrices_all)
                    if avg_matrix_all is not None:
                        avg_filename = f"{alignment}_{method}_{freq}_{subject}_average"
                        if has_unc_all_subj:
                            avg_filename += "_unc"
                        if self.exclude_lt_pronation:
                            avg_filename += "_NO-LT"
                        avg_filename += ".csv"

                        # Save into each subject directory (for convenience)
                        for subject_dir in subj_dir_list:
                            avg_filepath = subject_dir / avg_filename
                            self.save_average_matrix(avg_matrix_all, avg_filepath)
                else:
                    logging.warning(f"No matrices found for subject {subject}")

                # Pro-only subject average
                if matrices_pro:
                    avg_matrix_pro = self.create_average_matrix(matrices_pro)
                    if avg_matrix_pro is not None:
                        avg_filename_pro = f"{alignment}_{method}_{freq}_{subject}_Pro_average"
                        if has_unc_pro_subj:
                            avg_filename_pro += "_unc"
                        if self.exclude_lt_pronation:
                            avg_filename_pro += "_NO-LT"
                        avg_filename_pro += ".csv"

                        for subject_dir in subj_dir_list:
                            avg_filepath_pro = subject_dir / avg_filename_pro
                            self.save_average_matrix(avg_matrix_pro, avg_filepath_pro)

                # Anti-only subject average
                if matrices_anti:
                    avg_matrix_anti = self.create_average_matrix(matrices_anti)
                    if avg_matrix_anti is not None:
                        avg_filename_anti = f"{alignment}_{method}_{freq}_{subject}_Anti_average"
                        if has_unc_anti_subj:
                            avg_filename_anti += "_unc"
                        if self.exclude_lt_pronation:
                            avg_filename_anti += "_NO-LT"
                        avg_filename_anti += ".csv"

                        for subject_dir in subj_dir_list:
                            avg_filepath_anti = subject_dir / avg_filename_anti
                            self.save_average_matrix(avg_matrix_anti, avg_filepath_anti)

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    def print_summary(self):
        """Print summary of operations"""
        logging.info("\nOperation Summary")
        logging.info("================")
        logging.info(f"Total matrices processed: {self.matrices_processed}")
        logging.info(f"Total averages created: {self.averages_created}")
        logging.info(f"Total NaN values replaced: {self.total_nan_count}")
        if self.exclude_lt_pronation:
            logging.info(f"LT-Pronation files excluded: {self.lt_pronation_files_excluded}")
        logging.info(f"Errors encountered: {self.errors}")
        logging.info(f"\nProcess completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def main():
    """Main function to execute the average creation"""
    creator = AverageMatrixCreator()
    creator.process_dataset()
    creator.print_summary()

if __name__ == "__main__":
    main()
