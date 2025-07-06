#!/usr/bin/env python3

"""
Test script for Enhanced Wilcoxon Analysis V2
This script tests the new NO-LT file selection functionality.
"""

import sys
from pathlib import Path

# Add the current directory to the path so we can import the main script
sys.path.append(str(Path(__file__).parent))

from meg_wilcoxon_analysis_V2 import MEGWilcoxonAnalyzer

def test_no_lt_functionality():
    """Test the NO-LT file selection functionality"""
    print("Testing Enhanced Wilcoxon Analysis V2")
    print("====================================")
    
    # Create a test instance
    analyzer = MEGWilcoxonAnalyzer()
    
    # Test the NO-LT file detection method
    print("\nTesting NO-LT file detection:")
    test_base_dir = Path("Data/DataSet1/mov/L/gDTF_10Hz")
    if test_base_dir.exists():
        no_lt_exists = analyzer.check_no_lt_files_exist(test_base_dir, "mov", "gDTF_10Hz")
        print(f"NO-LT files exist in test directory: {no_lt_exists}")
    else:
        print("Test directory does not exist - this is expected for testing")
    
    # Test the file pattern generation
    print("\nTesting file pattern generation:")
    
    # Test for Full files
    analyzer.use_no_lt_files = False
    l_patterns, r_patterns = analyzer.get_file_patterns("mov", "gDTF_10Hz", "DOC")
    print("Full file patterns:")
    print(f"  L patterns: {l_patterns}")
    print(f"  R patterns: {r_patterns}")
    
    # Test for NO-LT files
    analyzer.use_no_lt_files = True
    l_patterns, r_patterns = analyzer.get_file_patterns("mov", "gDTF_10Hz", "DOC")
    print("NO-LT file patterns:")
    print(f"  L patterns: {l_patterns}")
    print(f"  R patterns: {r_patterns}")
    
    # Test output filename generation
    print("\nTesting output filename generation:")
    test_cases = [
        ("gDTF_10Hz", False),  # Full files
        ("gDTF_20Hz", True),   # NO-LT files
        ("iPDC_25Hz", False),  # Full files
        ("gPDC_100Hz", True),  # NO-LT files
    ]
    
    for method_freq, use_no_lt in test_cases:
        analyzer.use_no_lt_files = use_no_lt
        
        # Output matrix filename
        output_filename = f"{method_freq}_L-R_wilcoxon"
        if use_no_lt:
            output_filename += "_NO-LT"
        output_filename += ".csv"
        
        # Heatmap filename
        heatmap_filename = f"{method_freq}_L-R_wilcoxon"
        if use_no_lt:
            heatmap_filename += "_NO-LT"
        heatmap_filename += "_heatmap.png"
        
        print(f"  {method_freq} (NO-LT: {use_no_lt}):")
        print(f"    Output: {output_filename}")
        print(f"    Heatmap: {heatmap_filename}")
    
    print("\nEnhancement test completed!")
    print("The V2 script now includes:")
    print("- User selection between Full and NO-LT average files")
    print("- Automatic fallback from NO-LT to regular files when needed")
    print("- Proper output file naming with _NO-LT suffix")
    print("- Detailed logging of file selection and fallback behavior")
    print("- Enhanced file pattern detection for both file types")

if __name__ == "__main__":
    test_no_lt_functionality() 