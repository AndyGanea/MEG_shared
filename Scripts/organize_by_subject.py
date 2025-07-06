"""
MEG Signal Analysis - Subject-based File Organization
--------------------------------------------------
This script reorganizes MEG data files by subject, creating a hierarchical structure:
Data/
├── cue/
│   ├── L/
│   │   ├── gDTF_10Hz/
│   │   │   ├── DOC/
│   │   │   ├── GB/
│   │   │   └── ...
│   │   ├── gDTF_20Hz/
│   │   └── ...
│   └── R/
└── mov/
"""

import os
from pathlib import Path
import shutil
import logging
from typing import List, Set

class SubjectFileOrganizer:
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
                logging.FileHandler('subject_organization.log'),
                logging.StreamHandler()
            ]
        )

    def organize_files(self):
        """Main function to organize files by subject"""
        logging.info("Starting subject-based file organization")
        
        for alignment in self.alignments:
            for target in self.targets:
                for method in self.methods:
                    for freq in self.frequencies:
                        # Construct source directory path
                        source_dir = self.data_dir / alignment / target / f"{method}_{freq}"
                        
                        if not source_dir.exists():
                            logging.warning(f"Source directory not found: {source_dir}")
                            continue
                            
                        logging.info(f"Processing directory: {source_dir}")
                        
                        # Process each subject
                        for subject in self.subjects:
                            # Create subject directory
                            subject_dir = source_dir / subject
                            
                            # Check if there are any files for this subject
                            subject_files = self._get_subject_files(source_dir, subject)
                            
                            if subject_files:
                                # Create subject directory if files exist
                                subject_dir.mkdir(exist_ok=True)
                                
                                # Copy files to subject directory
                                self._copy_subject_files(subject_files, subject_dir)
                            else:
                                logging.info(f"No files found for subject {subject} in {source_dir}")

    def _get_subject_files(self, directory: Path, subject: str) -> Set[Path]:
        """
        Get all files for a specific subject in a directory.
        
        Args:
            directory: Source directory to search
            subject: Subject identifier
            
        Returns:
            Set of Path objects for files belonging to the subject
        """
        subject_files = set()
        
        try:
            for file_path in directory.glob("*.csv"):
                # Check if file belongs to subject
                if f"_{subject}_" in file_path.name:
                    subject_files.add(file_path)
                    
        except Exception as e:
            logging.error(f"Error searching for files in {directory}: {str(e)}")
            
        return subject_files

    def _copy_subject_files(self, files: Set[Path], target_dir: Path) -> None:
        """
        Copy files to subject directory.
        
        Args:
            files: Set of files to copy
            target_dir: Target directory for copies
        """
        for file_path in files:
            try:
                shutil.copy2(file_path, target_dir)
                logging.info(f"Copied {file_path.name} to {target_dir}")
                
            except Exception as e:
                logging.error(f"Error copying {file_path}: {str(e)}")

    def verify_organization(self) -> None:
        """Verify the organization and print summary"""
        total_subjects = 0
        total_files = 0
        
        for alignment in self.alignments:
            for target in self.targets:
                for method in self.methods:
                    for freq in self.frequencies:
                        base_dir = self.data_dir / alignment / target / f"{method}_{freq}"
                        
                        if not base_dir.exists():
                            continue
                            
                        for subject in self.subjects:
                            subject_dir = base_dir / subject
                            
                            if subject_dir.exists():
                                total_subjects += 1
                                file_count = len(list(subject_dir.glob("*.csv")))
                                total_files += file_count
                                
                                if file_count > 0:
                                    logging.info(f"Found {file_count} files for {subject} in {subject_dir}")

        logging.info(f"\nOrganization Summary:")
        logging.info(f"Total subject directories created: {total_subjects}")
        logging.info(f"Total files organized: {total_files}")

def main():
    print("Subject-based File Organization")
    print("==============================")
    
    organizer = SubjectFileOrganizer()
    
    response = input("Do you want to organize files by subject? (y/n): ").lower()
    if response == 'y':
        organizer.organize_files()
        organizer.verify_organization()
        print("\nFile organization complete. Check subject_organization.log for details.")
    else:
        print("Program terminated by user.")

if __name__ == "__main__":
    main() 