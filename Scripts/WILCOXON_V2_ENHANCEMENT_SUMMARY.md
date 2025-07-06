# Wilcoxon Analysis V2 Enhancement Summary

## Overview
The `meg_wilcoxon_analysis_V2.py` script has been enhanced with a new feature to select between Full and NO-LT average files for Wilcoxon analysis, providing more flexibility in data processing.

## New Feature: Average File Type Selection

### User Interface
- **New Prompt**: "Which average files to use? (1=Full, 2=NO-LT)"
- **User Choice**: Allows users to select between regular average files or NO-LT average files
- **Integration**: Works alongside existing iDTF exclusion option

### File Selection Logic
- **Option 1 (Full)**: Uses regular average files (e.g., `cue_L_gDTF_10Hz_average.csv`)
- **Option 2 (NO-LT)**: Uses NO-LT average files (e.g., `cue_L_gDTF_10Hz_average_NO-LT.csv`)
- **Fallback Behavior**: If user selects NO-LT but no such files exist, automatically falls back to regular files

### File Detection and Processing

#### NO-LT File Detection:
- **Method**: `check_no_lt_files_exist()` - Scans directories for files with `_NO-LT` suffix
- **Scope**: Checks both overall averages and subject-specific averages
- **Patterns Detected**:
  - `*_average_NO-LT.csv` (overall averages)
  - `*_*_average_NO-LT.csv` (subject averages)

#### File Pattern Generation:
- **Method**: `get_file_patterns()` - Returns appropriate file patterns based on user selection
- **Full Files**: Standard patterns without `_NO-LT` suffix
- **NO-LT Files**: Patterns with `_NO-LT` suffix

### Processing Behavior

#### Scenario 1: NO-LT files exist and user selects them
```
User selects: "Which average files to use? (1=Full, 2=NO-LT): 2"
Result: Uses NO-LT files, output files get _NO-LT suffix
```

#### Scenario 2: NO-LT files don't exist but user selects them
```
User selects: "Which average files to use? (1=Full, 2=NO-LT): 2"
Result: Falls back to regular files, logs warning, output files still get _NO-LT suffix
```

#### Scenario 3: User selects Full files
```
User selects: "Which average files to use? (1=Full, 2=NO-LT): 1"
Result: Uses regular files, no suffix added to output files
```

## Enhanced Logging and Error Handling

### New Logging Features:
- **File Type Selection**: Logs which type of average files are being used
- **NO-LT File Detection**: Reports whether NO-LT files were found
- **Fallback Logging**: Warns when falling back from NO-LT to regular files
- **File Finding**: Logs which specific files were found and used

### Logging Examples:
```
Using NO-LT files: True
NO-LT files found for gDTF_10Hz, using NO-LT averages
NO-LT files not found for gDTF_20Hz, falling back to regular averages
Found regular L file for subject DOC: mov_L_gDTF_20Hz_average.csv
```

## Output File Naming

### When using Full files:
- **Output Matrix**: `gDTF_10Hz_L-R_wilcoxon.csv`
- **Heatmap**: `gDTF_10Hz_L-R_wilcoxon_heatmap.png`

### When using NO-LT files:
- **Output Matrix**: `gDTF_10Hz_L-R_wilcoxon_NO-LT.csv`
- **Heatmap**: `gDTF_10Hz_L-R_wilcoxon_NO-LT_heatmap.png`

## New Methods Added

### `ask_average_file_type() -> bool`
- Prompts user for average file type selection
- Returns True for NO-LT (option 2), False for Full (option 1)

### `check_no_lt_files_exist(base_dir, prefix, method_freq) -> bool`
- Scans directory for NO-LT average files
- Checks both overall and subject-specific files
- Returns True if any NO-LT files are found

### `get_file_patterns(prefix, method_freq, subject=None) -> tuple`
- Returns appropriate file patterns based on user selection
- Handles both overall and subject-specific patterns
- Supports fallback logic

## Enhanced Existing Methods

### `has_required_files()`
- Updated to use `get_file_patterns()` for consistent file detection
- Supports both Full and NO-LT file types

### `has_sufficient_data()`
- Enhanced to check for NO-LT files when user selects them
- Maintains backward compatibility with regular files

### `process_folder()`
- Updated output file naming to include `_NO-LT` suffix when appropriate
- Applies to both CSV output and heatmap files

## Error Prevention and User Experience

### Robust Fallback System:
- **Automatic Detection**: Checks for NO-LT files before processing
- **Graceful Fallback**: Continues with regular files if NO-LT not found
- **Clear Logging**: Informs user about fallback behavior

### Consistent Naming:
- **Predictable Output**: Always applies appropriate suffix based on user selection
- **Clear Identification**: Output files clearly indicate which input type was used

## Usage Examples

### Example 1: Processing with NO-LT files
```
User: "Which average files to use? (1=Full, 2=NO-LT): 2"
Script: Uses NO-LT files, creates gDTF_10Hz_L-R_wilcoxon_NO-LT.csv
```

### Example 2: Fallback scenario
```
User: "Which average files to use? (1=Full, 2=NO-LT): 2"
Script: NO-LT files not found, uses regular files, still creates NO-LT output
```

### Example 3: Regular processing
```
User: "Which average files to use? (1=Full, 2=NO-LT): 1"
Script: Uses regular files, creates gDTF_10Hz_L-R_wilcoxon.csv
```

## Testing
A test script (`test_wilcoxon_v2.py`) has been created to verify:
- NO-LT file detection functionality
- File pattern generation for both types
- Output filename generation with proper suffixes
- All enhancement functionality

## Benefits
- **Data Flexibility**: Users can choose which average files to analyze
- **Clear Output**: Output files clearly indicate input file type used
- **Robust Processing**: Handles missing files gracefully with fallback
- **Detailed Logging**: Full transparency about file selection and processing
- **Backward Compatibility**: Works with existing datasets and file structures 