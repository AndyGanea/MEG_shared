#!/usr/bin/env python3

"""
MEG Wilcoxon Summary Script
---------------------------
This script scans Wilcoxon analysis log files and creates a summary CSV file
containing only the non-zero values that were retained in the analysis.

The script:
1. Scans the Data folder for DataSet folders
2. Lets user select a specific Wilcoxon folder
3. Analyzes all log files in subfolders
4. Extracts non-zero retained values with their statistics
5. Creates a summary CSV with columns: Frequency, Region_From, Region_To, W_Stat, P_value
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
from datetime import datetime
import logging
import sys
from typing import List, Tuple, Dict, Optional

class WilcoxonSummaryAnalyzer:
    def __init__(self):
        """Initialize the Wilcoxon summary analyzer"""
        self.data_dir = Path("Data")
        self.logs_dir = Path("Logs")
        
        # Region labels mapping (32 regions total)
        self.region_labels = [
            'V1-L', 'V3-L', 'SPOC-L', 'AG-L', 'POJ-L', 'SPL-L', 'mIPS-L', 'VIP-L', 
            'IPL-L', 'STS-L', 'S1-L', 'M1-L', 'SMA-L', 'PMd-L', 'FEF-L', 'PMv-L',
            'V1-R', 'V3-R', 'SPOC-R', 'AG-R', 'POJ-R', 'SPL-R', 'mIPS-R', 'VIP-R', 
            'IPL-R', 'STS-R', 'S1-R', 'M1-R', 'SMA-R', 'PMd-R', 'FEF-R', 'PMv-R'
        ]
        
        # Initialize results storage by method
        self.summary_data_by_method = {}
        
        # Setup logging
        self.setup_logging()

    def setup_logging(self):
        """Setup logging to both file and console"""
        timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        log_file = self.logs_dir / f"wilcoxon_summary_{timestamp}.txt"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        logging.info("MEG Wilcoxon Summary Analysis")
        logging.info("============================")
        logging.info(f"Process started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info("-----------------------------\n")

    def select_dataset_folder(self) -> Path:
        """Present available DataSet folders and let user select one"""
        print("\nAvailable DataSet folders:")
        print("=========================")
        
        available_dirs = [d for d in self.data_dir.iterdir() 
                         if d.is_dir() and d.name.startswith("DataSet")]
        
        if not available_dirs:
            print("No DataSet folders found in Data directory!")
            sys.exit(1)
        
        for idx, dir_path in enumerate(available_dirs, 1):
            print(f"{idx}. {dir_path.name}")
        
        while True:
            try:
                choice = int(input("\nSelect DataSet number: "))
                if 1 <= choice <= len(available_dirs):
                    selected_dir = available_dirs[choice - 1]
                    print(f"\nSelected: {selected_dir.name}")
                    return selected_dir
                else:
                    print("Invalid selection. Please try again.")
            except ValueError:
                print("Please enter a valid number.")

    def select_wilcoxon_folder(self, dataset_dir: Path) -> Path:
        """Present available Wilcoxon folders and let user select one"""
        print(f"\nAvailable Wilcoxon folders in {dataset_dir.name}:")
        print("=============================================")
        
        available_wilcoxon_dirs = [d for d in dataset_dir.iterdir() 
                                  if d.is_dir() and d.name.startswith("Wilcoxon")]
        
        if not available_wilcoxon_dirs:
            print("No Wilcoxon folders found in the selected dataset!")
            sys.exit(1)
        
        for idx, dir_path in enumerate(available_wilcoxon_dirs, 1):
            print(f"{idx}. {dir_path.name}")
        
        while True:
            try:
                choice = int(input("\nSelect Wilcoxon folder number: "))
                if 1 <= choice <= len(available_wilcoxon_dirs):
                    selected_dir = available_wilcoxon_dirs[choice - 1]
                    print(f"\nSelected: {selected_dir.name}")
                    return selected_dir
                else:
                    print("Invalid selection. Please try again.")
            except ValueError:
                print("Please enter a valid number.")

    def extract_method_and_frequency_from_folder_name(self, folder_name: str) -> tuple:
        """Extract method and frequency from folder name (e.g., 'Cue_gPDC_10Hz' -> ('gPDC', '10Hz'))"""
        # Look for method and frequency pattern in folder name
        method_freq_pattern = r'([a-zA-Z]+)_(\d+Hz)'
        match = re.search(method_freq_pattern, folder_name)
        if match:
            return match.group(1), match.group(2)
        else:
            logging.warning(f"Could not extract method and frequency from folder name: {folder_name}")
            return "Unknown", "Unknown"

    def parse_log_file(self, log_file_path: Path, method: str, frequency: str) -> List[Dict]:
        """Parse a single Wilcoxon log file and extract non-zero retained values (robust line-by-line parser)"""
        results = []
        try:
            with open(log_file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            i = j = None
            w_stat = p_value = overall_value = None
            result = None
            for line in lines:
                cell_match = re.match(r'Cell \((\d+), (\d+)\):', line)
                if cell_match:
                    # If previous cell info is complete, process it
                    if (
                        i is not None and j is not None and w_stat is not None and p_value is not None
                        and overall_value is not None and result is not None
                    ):
                        if overall_value != 0.0 and result == "Retained":
                            region_from = self.region_labels[j] if j < len(self.region_labels) else f"Region_{j}"
                            region_to = self.region_labels[i] if i < len(self.region_labels) else f"Region_{i}"
                            results.append({
                                'Method': method,
                                'Frequency': frequency,
                                'Region_From': region_from,
                                'Region_To': region_to,
                                'W_Stat': w_stat,
                                'P_value': p_value
                            })
                    # Start new cell
                    i = int(cell_match.group(1))
                    j = int(cell_match.group(2))
                    w_stat = p_value = overall_value = None
                    result = None
                elif 'W-statistic:' in line:
                    w_stat_match = re.search(r'W-statistic: ([\d.]+)', line)
                    if w_stat_match:
                        w_stat = float(w_stat_match.group(1))
                elif 'P-value:' in line:
                    p_value_match = re.search(r'P-value: ([\d.]+)', line)
                    if p_value_match:
                        p_value = float(p_value_match.group(1))
                elif 'Overall Matrix Value:' in line:
                    overall_match = re.search(r'Overall Matrix Value: ([\d.-]+) \((Retained|Set to zero)\)', line)
                    if overall_match:
                        overall_value = float(overall_match.group(1))
                        result = overall_match.group(2)
            # Process last cell
            if (
                i is not None and j is not None and w_stat is not None and p_value is not None
                and overall_value is not None and result is not None
            ):
                if overall_value != 0.0 and result == "Retained":
                    region_from = self.region_labels[j] if j < len(self.region_labels) else f"Region_{j}"
                    region_to = self.region_labels[i] if i < len(self.region_labels) else f"Region_{i}"
                    results.append({
                        'Method': method,
                        'Frequency': frequency,
                        'Region_From': region_from,
                        'Region_To': region_to,
                        'W_Stat': w_stat,
                        'P_value': p_value
                    })
        except Exception as e:
            logging.error(f"Error parsing log file {log_file_path}: {str(e)}")
        return results

    def process_wilcoxon_folder(self, wilcoxon_dir: Path):
        """Process all subfolders in the selected Wilcoxon directory"""
        logging.info(f"Processing Wilcoxon folder: {wilcoxon_dir.name}")
        
        # Get all subfolders (e.g., Cue_gPDC_10Hz, Mov_gDTF_20Hz, etc.)
        subfolders = [d for d in wilcoxon_dir.iterdir() if d.is_dir()]
        
        if not subfolders:
            logging.warning("No subfolders found in Wilcoxon directory!")
            return
        
        logging.info(f"Found {len(subfolders)} subfolders to process")
        
        for subfolder in subfolders:
            logging.info(f"\nProcessing subfolder: {subfolder.name}")
            
            # Extract method and frequency from folder name
            method, frequency = self.extract_method_and_frequency_from_folder_name(subfolder.name)
            
            # Find log files in the subfolder
            log_files = list(subfolder.glob("*_wilcoxon.txt"))
            
            if not log_files:
                logging.warning(f"No log files found in {subfolder.name}")
                continue
            
            logging.info(f"Found {len(log_files)} log files in {subfolder.name}")
            
            # Process each log file
            for log_file in log_files:
                logging.info(f"Processing log file: {log_file.name}")
                results = self.parse_log_file(log_file, method, frequency)
                
                if results:
                    # Initialize method in summary_data_by_method if not exists
                    if method not in self.summary_data_by_method:
                        self.summary_data_by_method[method] = []
                    
                    self.summary_data_by_method[method].extend(results)
                    logging.info(f"Extracted {len(results)} non-zero retained values from {log_file.name}")
                else:
                    logging.info(f"No non-zero retained values found in {log_file.name}")

    def create_summary_csv_by_method(self, wilcoxon_dir: Path):
        """Create separate summary CSV files for each method"""
        if not self.summary_data_by_method:
            logging.warning("No data to summarize!")
            return []
        
        output_paths = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for method, data in self.summary_data_by_method.items():
            if not data:
                logging.info(f"No data for method {method}")
                continue
            
            # Create DataFrame for this method
            df = pd.DataFrame(data)
            
            # Sort by Frequency, Region_From, Region_To
            df = df.sort_values(['Frequency', 'Region_From', 'Region_To'])
            
            # Create output filename with method and timestamp
            output_filename = f"wilcoxon_summary_{method}_{timestamp}.csv"
            output_path = wilcoxon_dir / output_filename
            
            # Save to CSV
            df.to_csv(output_path, index=False)
            
            logging.info(f"\nSummary CSV created for {method}: {output_filename}")
            logging.info(f"Total non-zero retained values for {method}: {len(df)}")
            logging.info(f"Output location: {output_path}")
            
            # Print summary statistics for this method
            logging.info(f"\nSummary for {method} by frequency:")
            freq_counts = df['Frequency'].value_counts()
            for freq, count in freq_counts.items():
                logging.info(f"  {freq}: {count} values")
            
            output_paths.append(output_path)
        
        return output_paths

    def run_analysis(self):
        """Run the complete Wilcoxon summary analysis"""
        logging.info("Starting Wilcoxon summary analysis...")
        
        # Step 1: Select dataset folder
        dataset_dir = self.select_dataset_folder()
        
        # Step 2: Select Wilcoxon folder
        wilcoxon_dir = self.select_wilcoxon_folder(dataset_dir)
        
        # Step 3: Process all subfolders and log files
        self.process_wilcoxon_folder(wilcoxon_dir)
        
        # Step 4: Create summary CSV files by method
        if self.summary_data_by_method:
            output_paths = self.create_summary_csv_by_method(wilcoxon_dir)
            logging.info(f"\nAnalysis completed successfully!")
            logging.info(f"Summary files saved:")
            for path in output_paths:
                logging.info(f"  - {path}")
        else:
            logging.warning("No data found to summarize!")
        
        logging.info(f"\nProcess completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def main():
    """Main execution function"""
    analyzer = WilcoxonSummaryAnalyzer()
    analyzer.run_analysis()

if __name__ == "__main__":
    main() 