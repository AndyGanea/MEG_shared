#!/usr/bin/env python3

"""
MEG Signal Analysis - Average Files Cleanup Helper
-----------------------------------------------
This script helps clean up 'average.csv' files from a selected folder under Data.
It provides a user interface to select the target folder and removes all files
named exactly 'average.csv'.
"""

import os
from pathlib import Path
import logging
from datetime import datetime
import sys

class AverageFilesCleaner:
    def __init__(self):
        """Initialize the cleaner with necessary paths"""
        self.data_dir = Path("Data")
        self.logs_dir = Path("Logs")
        
        # Get user selection for dataset folder
        self.target_dir = self.select_folder()
        
        # Initialize counters
        self.files_found = 0
        self.files_deleted = 0
        self.errors = 0
        
        # Setup logging
        self.setup_logging()

    def select_folder(self) -> Path:
        """Present available folders and let user select one"""
        print("\nAvailable folders under Data:")
        print("============================")
        
        # Get all directories under Data
        available_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
        
        # Present options to user
        for idx, dir_path in enumerate(available_dirs, 1):
            print(f"{idx}. {dir_path.name}")
        
        while True:
            try:
                choice = int(input("\nSelect folder number: "))
                if 1 <= choice <= len(available_dirs):
                    selected_dir = available_dirs[choice - 1]
                    print(f"\nSelected: {selected_dir.name}")
                    
                    # Confirm deletion
                    confirm = input(f"\nThis will delete all 'average.csv' files in {selected_dir.name} "
                                  f"and its subfolders. Continue? (y/n): ").lower()
                    if confirm == 'y':
                        return selected_dir
                    else:
                        print("Operation cancelled.")
                        sys.exit(0)
                else:
                    print("Invalid selection. Please try again.")
            except ValueError:
                print("Please enter a valid number.")

    def setup_logging(self):
        """Setup logging to both file and console"""
        timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        log_file = self.logs_dir / f"average_files_cleanup_{timestamp}.txt"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        logging.info("MEG Signal Analysis - Average Files Cleanup")
        logging.info("=========================================")
        logging.info(f"Selected folder: {self.target_dir.name}")
        logging.info(f"Process started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info("-----------------------------------------\n")

    def delete_average_files(self):
        """Find and delete all 'average.csv' files in the selected folder structure"""
        logging.info("Scanning for 'average.csv' files...")
        
        try:
            # Walk through all subdirectories
            for root, _, files in os.walk(self.target_dir):
                for file in files:
                    if file == 'average.csv':
                        file_path = Path(root) / file
                        self.files_found += 1
                        
                        try:
                            # Delete the file
                            file_path.unlink()
                            self.files_deleted += 1
                            logging.info(f"Deleted: {file_path}")
                        except Exception as e:
                            self.errors += 1
                            logging.error(f"Error deleting {file_path}: {str(e)}")
        
        except Exception as e:
            self.errors += 1
            logging.error(f"Error scanning directory: {str(e)}")

    def print_summary(self):
        """Print summary of operations"""
        logging.info("\nOperation Summary")
        logging.info("================")
        logging.info(f"Total 'average.csv' files found: {self.files_found}")
        logging.info(f"Files successfully deleted: {self.files_deleted}")
        logging.info(f"Errors encountered: {self.errors}")
        logging.info(f"\nProcess completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def main():
    """Main function to execute the cleanup"""
    cleaner = AverageFilesCleaner()
    cleaner.delete_average_files()
    cleaner.print_summary()

if __name__ == "__main__":
    main() 