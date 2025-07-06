# Circular Visualization V2 Enhancement Summary

## Overview
The `meg_visualization_circular_select_V2.py` script has been enhanced to support NO-LT files, allowing users to visualize both regular and NO-LT connectivity data with proper categorization and naming.

## New Feature: NO-LT File Support

### Enhanced File Selection Interface
- **File Categorization**: Automatically detects and separates regular files from NO-LT files
- **Organized Display**: Shows files in two categories:
  - **Regular Files**: Standard connectivity files (e.g., `gDTF_10Hz_L-R_wilcoxon.csv`)
  - **NO-LT Files**: Files with NO-LT suffix (e.g., `gDTF_10Hz_L-R_wilcoxon_NO-LT.csv`)
- **Clear Logging**: Reports how many files of each type were found

### File Selection Menu Example:
```
Available files:
================

Regular Files:
--------------
1. gDTF_10Hz_L-R_wilcoxon.csv
2. iPDC_20Hz_L-R_wilcoxon.csv
3. gPDC_25Hz_L-R_wilcoxon.csv

NO-LT Files:
------------
4. gDTF_10Hz_L-R_wilcoxon_NO-LT.csv
5. gPDC_25Hz_L-R_wilcoxon_NO-LT.csv

Select file number (1-5):
```

## Enhanced File Processing

### Smart Filename Parsing:
- **NO-LT Detection**: Automatically detects `_NO-LT` suffix in filenames
- **Method/Frequency Extraction**: Correctly parses method and frequency from both file types
- **Analysis Type Detection**: Properly identifies analysis type (e.g., "wilcoxon" vs "wilcoxon_NO-LT")

### Parsing Examples:
```
Input: "gDTF_10Hz_L-R_wilcoxon.csv"
Output: method_freq="gDTF_10Hz", analysis_type="wilcoxon", is_no_lt=False

Input: "gDTF_10Hz_L-R_wilcoxon_NO-LT.csv"
Output: method_freq="gDTF_10Hz", analysis_type="wilcoxon_NO-LT", is_no_lt=True
```

## Updated Output Naming

### Connectivity Reports:
- **Regular Files**: `gDTF_10Hz_wilcoxon_connectivity_report_20250628_165000.txt`
- **NO-LT Files**: `gDTF_10Hz_wilcoxon_NO-LT_connectivity_report_20250628_165000.txt`

### Visualization Files:
- **All Connections**:
  - Regular: `gDTF_10Hz_wilcoxon_circular_connectogram_all_20250628_165000.png`
  - NO-LT: `gDTF_10Hz_wilcoxon_NO-LT_circular_connectogram_all_20250628_165000.png`

- **Positive Connections**:
  - Regular: `gDTF_10Hz_wilcoxon_circular_connectogram_positive_20250628_165000.png`
  - NO-LT: `gDTF_10Hz_wilcoxon_NO-LT_circular_connectogram_positive_20250628_165000.png`

- **Negative Connections**:
  - Regular: `gDTF_10Hz_wilcoxon_circular_connectogram_negative_20250628_165000.png`
  - NO-LT: `gDTF_10Hz_wilcoxon_NO-LT_circular_connectogram_negative_20250628_165000.png`

## Enhanced Functions

### `select_file()` - Enhanced:
- **File Categorization**: Automatically separates regular and NO-LT files
- **Organized Display**: Shows files in clear categories
- **Improved Logging**: Reports file counts and selection details
- **Better User Experience**: Clear numbering and categorization

### `generate_connectivity_report()` - Enhanced:
- **Dynamic Titles**: Report titles reflect the actual analysis type
- **NO-LT Support**: Properly handles NO-LT analysis types in report headers
- **Parameter Support**: Accepts analysis_type parameter for customization

### `run()` - Enhanced:
- **Smart Parsing**: Correctly handles both regular and NO-LT filenames
- **Proper Detection**: Automatically detects file type and adjusts processing
- **Enhanced Logging**: Provides detailed information about file type and parsing
- **Consistent Naming**: Ensures all output files reflect the input file type

## Usage Examples

### Example 1: Processing Regular File
```
User selects: "gDTF_10Hz_L-R_wilcoxon.csv"
Script detects: Regular file
Output files:
- gDTF_10Hz_wilcoxon_connectivity_report_20250628_165000.txt
- gDTF_10Hz_wilcoxon_circular_connectogram_all_20250628_165000.png
```

### Example 2: Processing NO-LT File
```
User selects: "gDTF_10Hz_L-R_wilcoxon_NO-LT.csv"
Script detects: NO-LT file
Output files:
- gDTF_10Hz_wilcoxon_NO-LT_connectivity_report_20250628_165000.txt
- gDTF_10Hz_wilcoxon_NO-LT_circular_connectogram_all_20250628_165000.png
```

## Logging Enhancements

### New Log Messages:
```
Found 3 regular files and 2 NO-LT files
Selected NO-LT file: gDTF_10Hz_L-R_wilcoxon_NO-LT.csv
Detected method_freq: gDTF_10Hz
Detected analysis_type: wilcoxon_NO-LT
File type: NO-LT
```

## Testing
A test script (`test_circular_v2.py`) has been created to verify:
- NO-LT file detection and categorization
- Filename parsing for both file types
- Output filename generation with proper suffixes
- All enhancement functionality

## Benefits
- **Clear Organization**: Users can easily distinguish between regular and NO-LT files
- **Consistent Naming**: Output files clearly indicate the input file type
- **Enhanced Usability**: Better file selection interface with categorization
- **Proper Processing**: Correct handling of both file types throughout the pipeline
- **Detailed Logging**: Full transparency about file processing and categorization
- **Backward Compatibility**: Works seamlessly with existing regular files 