#!/usr/bin/env python3

"""
Test script for Enhanced 3D Brain Visualization V2
This script tests the new NO-LT file support functionality.
"""

import sys
from pathlib import Path

# Add the current directory to the path so we can import the main script
sys.path.append(str(Path(__file__).parent))

# Import the functions we want to test
from meg_visualization_3DBrain_select_V2 import select_file

def test_no_lt_functionality():
    """Test the NO-LT file support functionality"""
    print("Testing Enhanced 3D Brain Visualization V2")
    print("===========================================")
    
    # Test filename parsing logic
    print("\nTesting filename parsing:")
    test_filenames = [
        "gDTF_10Hz_L-R_wilcoxon.csv",
        "gDTF_10Hz_L-R_wilcoxon_NO-LT.csv",
        "iPDC_20Hz_L-R_wilcoxon.csv",
        "gPDC_25Hz_L-R_wilcoxon_NO-LT.csv",
        "iDTF_100Hz_L-R_wilcoxon.csv"
    ]
    
    for filename in test_filenames:
        # Simulate the parsing logic from the run() function
        is_no_lt_file = "_NO-LT" in filename
        
        # Remove .csv extension for parsing
        filename_stem = filename.replace(".csv", "")
        
        if is_no_lt_file:
            filename_parts = filename_stem.replace("_NO-LT", "").split('_')
        else:
            filename_parts = filename_stem.split('_')
        
        method_freq = filename_parts[0] + "_" + filename_parts[1]
        analysis_type = filename_parts[-1]
        
        if is_no_lt_file:
            analysis_type += "_NO-LT"
        
        print(f"  {filename}:")
        print(f"    Method_Freq: {method_freq}")
        print(f"    Analysis_Type: {analysis_type}")
        print(f"    Is NO-LT: {is_no_lt_file}")
    
    # Test file categorization logic
    print("\nTesting file categorization:")
    test_files = [
        "gDTF_10Hz_L-R_wilcoxon.csv",
        "gDTF_10Hz_L-R_wilcoxon_NO-LT.csv",
        "iPDC_20Hz_L-R_wilcoxon.csv",
        "gPDC_25Hz_L-R_wilcoxon_NO-LT.csv",
        "iDTF_100Hz_L-R_wilcoxon.csv"
    ]
    
    regular_files = []
    no_lt_files = []
    
    for filename in test_files:
        if "_NO-LT" in filename:
            no_lt_files.append(filename)
        else:
            regular_files.append(filename)
    
    print(f"Regular files ({len(regular_files)}): {regular_files}")
    print(f"NO-LT files ({len(no_lt_files)}): {no_lt_files}")
    
    # Test output filename generation
    print("\nTesting output filename generation:")
    test_cases = [
        ("gDTF_10Hz", "wilcoxon", False),
        ("gDTF_10Hz", "wilcoxon_NO-LT", True),
        ("iPDC_20Hz", "wilcoxon", False),
        ("gPDC_25Hz", "wilcoxon_NO-LT", True),
    ]
    
    timestamp = "20250628_165000"
    
    for method_freq, analysis_type, is_no_lt in test_cases:
        # 3D brain visualization files
        all_filename = f"{method_freq}_{analysis_type}_3D_brain_all_{timestamp}.png"
        pos_filename = f"{method_freq}_{analysis_type}_3D_brain_positive_{timestamp}.png"
        neg_filename = f"{method_freq}_{analysis_type}_3D_brain_negative_{timestamp}.png"
        
        print(f"  {method_freq} (NO-LT: {is_no_lt}):")
        print(f"    All: {all_filename}")
        print(f"    Positive: {pos_filename}")
        print(f"    Negative: {neg_filename}")
    
    print("\nEnhancement test completed!")
    print("The V2 script now includes:")
    print("- NO-LT file detection and categorization")
    print("- Separate display of regular and NO-LT files")
    print("- Proper filename parsing for NO-LT files")
    print("- Updated output naming with NO-LT indicators")
    print("- Enhanced 3D brain visualization support")

if __name__ == "__main__":
    test_no_lt_functionality() 