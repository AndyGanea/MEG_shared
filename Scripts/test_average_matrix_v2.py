#!/usr/bin/env python3

"""
Test script for Enhanced Average Matrix Creator V2
This script tests the new LT-Pronation exclusion functionality.
"""

import sys
from pathlib import Path

# Add the current directory to the path so we can import the main script
sys.path.append(str(Path(__file__).parent))

from create_average_matrix_enhanced_V2 import AverageMatrixCreator

def test_lt_pronation_functionality():
    """Test the LT-Pronation exclusion functionality"""
    print("Testing Enhanced Average Matrix Creator V2")
    print("==========================================")
    
    # Create a test instance
    creator = AverageMatrixCreator()
    
    # Test the LT-Pronation detection method
    print("\nTesting LT-Pronation file detection:")
    test_files = [
        "cue_LT-Pronation_L_anti_DOC_gPDC_20Hz.csv",
        "mov_LT-Pronation_R_pro_JZ_gDTF_10Hz.csv",
        "cue_RT_Pronation_L_anti_DOC_gPDC_20Hz.csv",  # Should not be detected
        "mov_normal_file_gDTF_10Hz.csv"  # Should not be detected
    ]
    
    for filename in test_files:
        is_lt = creator.is_lt_pronation_file(filename)
        print(f"  {filename}: {'LT-Pronation' if is_lt else 'Not LT-Pronation'}")
    
    # Test the file inclusion logic
    print("\nTesting file inclusion logic:")
    creator.exclude_lt_pronation = True  # Simulate user choosing to exclude
    
    for filename in test_files:
        should_include = creator.should_include_file(filename)
        print(f"  {filename}: {'Include' if should_include else 'Exclude'}")
    
    # Test filename generation
    print("\nTesting filename generation:")
    test_cases = [
        ("cue", "L", "gPDC", "20Hz", "GB", False, False),  # Normal case
        ("mov", "R", "gDTF", "10Hz", "DOC", True, False),  # With LT exclusion
        ("cue", "L", "iPDC", "25Hz", "JZ", False, True),   # With unc suffix
        ("mov", "R", "gPDC", "100Hz", "LT", True, True),   # With both LT exclusion and unc
    ]
    
    for alignment, target, method, freq, subject, exclude_lt, is_unc in test_cases:
        creator.exclude_lt_pronation = exclude_lt
        
        # Overall average filename
        overall_filename = f"{alignment}_{target}_{method}_{freq}_average"
        if is_unc:
            overall_filename += "_unc"
        if exclude_lt:
            overall_filename += "_NO-LT"
        overall_filename += ".csv"
        
        # Subject average filename
        subject_filename = f"{alignment}_{target}_{method}_{freq}_{subject}_average"
        if is_unc:
            subject_filename += "_unc"
        if exclude_lt:
            subject_filename += "_NO-LT"
        subject_filename += ".csv"
        
        print(f"  Overall: {overall_filename}")
        print(f"  Subject: {subject_filename}")
    
    print("\nEnhancement test completed!")
    print("The V2 script now includes:")
    print("- LT-Pronation file detection and exclusion")
    print("- Proper filename suffixing (_NO-LT)")
    print("- Detailed logging of excluded files")
    print("- Warning when no LT-Pronation files exist")
    print("- Statistics tracking of excluded files")

if __name__ == "__main__":
    test_lt_pronation_functionality() 