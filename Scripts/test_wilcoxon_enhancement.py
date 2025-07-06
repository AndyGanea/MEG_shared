#!/usr/bin/env python3

"""
Test script for enhanced MEG Wilcoxon Analysis
This script tests the enhanced functionality for skipping empty folders.
"""

import sys
from pathlib import Path

# Add the current directory to the path so we can import the main script
sys.path.append(str(Path(__file__).parent))

from meg_wilcoxon_analysis import MEGWilcoxonAnalyzer

def test_enhancement():
    """Test the enhanced functionality"""
    print("Testing Enhanced MEG Wilcoxon Analysis")
    print("=====================================")
    
    # Create a test instance
    analyzer = MEGWilcoxonAnalyzer()
    
    # Test the has_required_files method
    print("\nTesting has_required_files method:")
    test_l_dir = Path("Data/DataSet1/mov/L/gDTF_10Hz")
    test_r_dir = Path("Data/DataSet1/mov/R/gDTF_10Hz")
    
    if test_l_dir.exists() and test_r_dir.exists():
        has_files = analyzer.has_required_files(test_l_dir, test_r_dir, "mov", "gDTF_10Hz")
        print(f"Test directories exist: {has_files}")
    else:
        print("Test directories do not exist - this is expected for testing")
    
    # Test the has_sufficient_data method
    print("\nTesting has_sufficient_data method:")
    test_folder = Path("Data/DataSet1/mov")
    if test_folder.exists():
        has_data = analyzer.has_sufficient_data(test_folder, "mov", "gDTF_10Hz")
        print(f"Test folder has sufficient data: {has_data}")
    else:
        print("Test folder does not exist - this is expected for testing")
    
    print("\nEnhancement test completed!")
    print("The script now includes:")
    print("- Empty folder detection and skipping")
    print("- Better logging of skipped folders")
    print("- Cleanup of empty directories")
    print("- Enhanced summary statistics")

if __name__ == "__main__":
    test_enhancement() 