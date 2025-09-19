 #!/usr/bin/env python3

"""
MEG Data File Organization Helper V2
-----------------------------------
This script organizes MEG measurement files into a structured directory hierarchy
for histogram analysis. V2 introduces a new grouping logic based on movement patterns.

Key Features:
- Organizes files from multiple subjects (DOC, GB, JDC, etc.)
- Handles multiple analysis methods (gDTF, iPDC, gPDC, iDTF)
- Supports different frequencies (10Hz, 20Hz, 25Hz, 100Hz)
- Creates dual-organization structure (by method/freq and by subject)
- Maintains detailed operation logging
- Supports both cue and movement aligned data
- Processes either regular or 'unc' files based on user selection
- NEW: Groups files by movement patterns (Left_movement/Right_movement)

Directory Structure Created:
DataSetX_[source][_unc]/
├── [alignment (cue/mov)]/
    ├── Left_movement/
    │   ├── [method_freq]/
    │   │   ├── subject files (L_pro and R_anti)
    │   │   └── [subject]/
    │   │       └── subject files
    └── Right_movement/
        ├── [method_freq]/
        │   ├── subject files (R_pro and L_anti)
        │   └── [subject]/
        │       └── subject files

Grouping Logic:
- Left_movement: Contains files from folders with "pro_L" and "anti_R" patterns
- Right_movement: Contains files from folders with "pro_R" and "anti_L" patterns

Usage:
    python helper_arrange_measurement_files_for_histogram_V2.py

The script will:
1. Present available source folders for selection
2. Ask for dataset number (3-7)
3. Create directory structure
4. Copy and organize files based on movement patterns
5. Provide detailed operation summary
"""

import os
import shutil
from pathlib import Path

class FileOrganizerV2:
    def __init__(self):
        """
        Initialize the File Organizer V2 with predefined configurations.
        
        Attributes:
            subjects (list): List of subject identifiers
            movement_patterns (dict): Mapping of patterns to movement types
            methods (list): List of analysis methods
            frequencies (list): List of frequency bands
            data_dir (Path): Base data directory
            source_dir (Path): Selected source folder
            dataset_number (int): User-selected dataset number
            process_unc (bool): Whether to process 'unc' files
            alignment (str): Data alignment type (cue/mov)
            dataset_name (str): Generated dataset folder name
            dataset_dir (Path): Path to dataset directory
            alignment_dir (Path): Path to alignment directory
        """
        self.subjects = ['DOC', 'GB', 'JDC', 'JFXD', 'JZ', 'LT', 'NvA', 'RR', 'SJB', 'BG']
        
        # Define movement patterns and their grouping
        self.movement_patterns = {
            'pro_L': 'Left_movement',
            'anti_R': 'Left_movement', 
            'pro_R': 'Right_movement',
            'anti_L': 'Right_movement'
        }
        
        self.methods = ['gDTF', 'iPDC', 'gPDC', 'iDTF']
        self.frequencies = ['10Hz', '20Hz', '25Hz', '100Hz']
        
        self.data_dir = Path("Data")
        self.source_dir = self.select_source_folder()
        self.dataset_number = self.get_dataset_number()
        self.process_unc = self.ask_process_unc()
        
        # Determine alignment from source folder name
        self.alignment = 'cue' if 'cue' in self.source_dir.name.lower() else 'mov'
        
        # Create new dataset folder name with appropriate suffix
        suffix = "_unc" if self.process_unc else ""
        self.dataset_name = f"DataSet{self.dataset_number}_{self.source_dir.name}{suffix}"
        self.dataset_dir = self.data_dir / self.dataset_name
        
        # Create alignment directory path
        self.alignment_dir = self.dataset_dir / self.alignment

    def select_source_folder(self) -> Path:
        """
        Let user select source folder from available base folders in Data directory.
        
        First presents available base folders, then shows subfolders within the selected base.
        
        Returns:
            Path: Selected source directory path
            
        Raises:
            ValueError: If invalid selection is made
        """
        # First, let user select base folder
        print("\nAvailable base folders in Data directory:")
        print("=========================================")
        
        available_base_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
        
        if not available_base_dirs:
            raise ValueError("No directories found in Data folder")
        
        for idx, dir_path in enumerate(available_base_dirs, 1):
            print(f"{idx}. {dir_path.name}")
        
        while True:
            try:
                choice = int(input("\nSelect base folder number: "))
                if 1 <= choice <= len(available_base_dirs):
                    selected_base_dir = available_base_dirs[choice - 1]
                    print(f"\nSelected base folder: {selected_base_dir.name}")
                    break
                else:
                    print("Invalid selection. Please try again.")
            except ValueError:
                print("Please enter a valid number.")
        
        # Now show subfolders within the selected base folder
        print(f"\nAvailable folders in {selected_base_dir.name}:")
        print("=" * (len(selected_base_dir.name) + 25))
        
        available_dirs = [d for d in selected_base_dir.iterdir() if d.is_dir()]
        
        if not available_dirs:
            raise ValueError(f"No subdirectories found in {selected_base_dir.name}")
        
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
                return number
            except ValueError:
                print("Please enter a valid number.")

    def ask_process_unc(self) -> bool:
        """Ask user whether to process unc or regular files."""
        while True:
            response = input("\nProcess 'unc' files? (y/n): ").lower()
            if response in ['y', 'n']:
                return response == 'y'
            print("Please enter 'y' or 'n'")

    def determine_movement_type(self, folder_name: str) -> str:
        """
        Determine movement type based on folder name patterns.
        
        Args:
            folder_name (str): Name of the condition folder
            
        Returns:
            str: Movement type ('Left_movement' or 'Right_movement') or None if no pattern found
        """
        for pattern, movement_type in self.movement_patterns.items():
            if pattern in folder_name:
                return movement_type
        return None

    def create_directory_structure(self):
        """
        Create the complete directory structure for data organization.
        
        Creates a hierarchical directory structure:
        - Dataset directory
        - Alignment directory (cue/mov)
        - Movement directories (Left_movement/Right_movement)
        - Method_frequency directories
        - Subject directories
        
        Handles existing directories and provides detailed feedback.
        """
        print("\nStep 1: Create directory structure")
        response = input("Do you want to execute this step? (y/n): ").lower()
        if response != 'y':
            print("Skipping step 1")
            return

        try:
            # Create main dataset directory
            if not self.dataset_dir.exists():
                self.dataset_dir.mkdir(parents=True)
                print(f"Created dataset folder: {self.dataset_dir}")
            else:
                print(f"Dataset folder already exists: {self.dataset_dir}")

            # Create alignment directory
            if not self.alignment_dir.exists():
                self.alignment_dir.mkdir(parents=True)
                print(f"Created alignment folder: {self.alignment_dir}")
            else:
                print(f"Alignment folder already exists: {self.alignment_dir}")

            # Create movement directories (Left_movement/Right_movement)
            movement_dirs = ['Left_movement', 'Right_movement']
            for movement_dir in movement_dirs:
                movement_path = self.alignment_dir / movement_dir
                if not movement_path.exists():
                    movement_path.mkdir()
                    print(f"Created movement folder: {movement_path}")
                else:
                    print(f"Movement folder already exists: {movement_path}")
                
                # Create method_frequency directories
                for method in self.methods:
                    for freq in self.frequencies:
                        method_freq_dir = movement_path / f"{method}_{freq}"
                        if not method_freq_dir.exists():
                            method_freq_dir.mkdir()
                            print(f"Created method_freq directory: {method_freq_dir}")
                        else:
                            print(f"Method_freq directory already exists: {method_freq_dir}")
                        
                        # Create subject folders inside method_freq directory
                        for subject in self.subjects:
                            subject_dir = method_freq_dir / subject
                            if not subject_dir.exists():
                                subject_dir.mkdir()
                                print(f"Created subject directory: {subject_dir}")
                            else:
                                print(f"Subject directory already exists: {subject_dir}")

        except Exception as e:
            print(f"Error creating directory structure: {str(e)}")

    def organize_files(self):
        """
        Organize files into the created directory structure based on movement patterns.
        
        Processes files from source directory:
        1. Identifies movement type from folder name patterns
        2. Identifies method, frequency, and subject from filenames
        3. Copies files to appropriate movement/method_freq directory
        4. Creates duplicate in subject-specific directory
        5. Maintains detailed tracking of operations
        
        Provides comprehensive logging of:
        - Files processed
        - Movement patterns found
        - Method-frequency combinations found
        - Files per subject
        - Any errors encountered
        """
        print("\nStep 2: Organize files")
        response = input("Do you want to execute this step? (y/n): ").lower()
        if response != 'y':
            print("Skipping step 2")
            return

        try:
            print(f"\nSource directory: {self.source_dir}")
            
            # Initialize tracking
            method_freq_counts = {}
            subject_counts = {subject: 0 for subject in self.subjects}
            movement_counts = {'Left_movement': 0, 'Right_movement': 0}
            failed_files = []
            skipped_folders = []
            files_copied = 0
            combinations_found = set()
            
            # Process condition directories
            condition_dirs = [d for d in self.source_dir.iterdir() if d.is_dir()]
            print(f"\nFound {len(condition_dirs)} condition directories")
            
            for condition_dir in condition_dirs:
                try:
                    # Determine movement type from folder name
                    movement_type = self.determine_movement_type(condition_dir.name)
                    
                    if movement_type is None:
                        skipped_folders.append(condition_dir.name)
                        print(f"Skipping folder (no movement pattern found): {condition_dir.name}")
                        continue
                    
                    # Parse condition directory name for additional info
                    dir_parts = condition_dir.name.split('_')
                    if len(dir_parts) < 4:
                        print(f"Warning: Unexpected condition directory format: {condition_dir.name}")
                        continue
                    
                    posture = dir_parts[0]
                    movement = dir_parts[1]
                    condition = dir_parts[2]
                    target = dir_parts[3]
                    
                    # Process CSV files
                    for file_path in condition_dir.glob("*.csv"):
                        # Skip unc files when processing regular files
                        if not self.process_unc and "_unc.csv" in file_path.name:
                            continue
                            
                        try:
                            parts = file_path.stem.split('_')
                            
                            # Extract file components
                            method = next((p for p in parts if p in self.methods), None)
                            freq = next((p for p in parts if p in self.frequencies), None)
                            subject = next((p for p in parts if p in self.subjects), None)
                            
                            if all([method, freq, subject]):
                                combo = f"{method}_{freq}"
                                combinations_found.add(combo)
                                
                                # Update counters
                                method_freq_counts[combo] = method_freq_counts.get(combo, 0) + 1
                                subject_counts[subject] += 1
                                movement_counts[movement_type] += 1
                                
                                # Create new filename
                                new_filename = f"{self.alignment}_{posture}-{movement}_{target}_{condition}_{subject}_{method}_{freq}"
                                if self.process_unc and "_unc" in file_path.name:
                                    new_filename += "_unc"
                                new_filename += ".csv"
                                
                                # Copy to method_freq directory within movement folder
                                method_freq_dir = self.alignment_dir / movement_type / f"{method}_{freq}"
                                method_freq_path = method_freq_dir / new_filename
                                shutil.copy2(file_path, method_freq_path)
                                files_copied += 1
                                
                                # Copy to subject directory
                                subject_dir = method_freq_dir / subject
                                subject_path = subject_dir / new_filename
                                shutil.copy2(file_path, subject_path)
                                files_copied += 1
                                
                            else:
                                failed_files.append((file_path.name, "Could not determine method/frequency/subject"))
                                
                        except Exception as e:
                            failed_files.append((file_path.name, str(e)))
                            
                except Exception as e:
                    print(f"Error processing condition directory {condition_dir.name}: {str(e)}")

            # Print summary
            print("\nFile Organization Summary")
            print("========================")
            print(f"Total files copied: {files_copied}")
            print(f"Note: Each file is copied twice (method_freq and subject folders)")
            
            print("\nMovement Distribution:")
            for movement, count in movement_counts.items():
                print(f"- {movement}: {count} files")
            
            print("\nMethod-Frequency Combinations:")
            for combo in sorted(combinations_found):
                count = method_freq_counts[combo]
                print(f"- {combo}: {count} files")
            
            print("\nFiles per Subject:")
            for subject, count in sorted(subject_counts.items()):
                if count > 0:
                    print(f"- {subject}: {count} files")
            
            if skipped_folders:
                print("\nSkipped Folders (no movement pattern):")
                for folder in skipped_folders:
                    print(f"- {folder}")
            
            if failed_files:
                print("\nFailed Files:")
                for name, reason in failed_files:
                    print(f"- {name}: {reason}")

        except Exception as e:
            print(f"Error organizing files: {str(e)}")

def main():
    """
    Main execution function.
    
    Creates FileOrganizerV2 instance and executes the organization process:
    1. Initialize organizer
    2. Create directory structure
    3. Organize files based on movement patterns
    4. Display results
    """
    organizer = FileOrganizerV2()
    
    print("\nFile Organization Program V2")
    print("============================")
    print(f"Creating: {organizer.dataset_name}")
    print(f"Alignment: {organizer.alignment}")
    print(f"Processing: {'unc' if organizer.process_unc else 'regular'} files")
    print("Movement Grouping:")
    print("- Left_movement: pro_L and anti_R patterns")
    print("- Right_movement: pro_R and anti_L patterns")
    print("------------------------")
    
    # Execute steps
    organizer.create_directory_structure()
    organizer.organize_files()

if __name__ == "__main__":
    main()