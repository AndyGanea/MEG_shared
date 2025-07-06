# MEG Wilcoxon Analysis Enhancement Summary

## Overview
The `meg_wilcoxon_analysis.py` script has been enhanced to properly handle empty folders and provide better error handling and logging.

## Key Enhancements

### 1. Empty Folder Detection and Skipping
- **New Method**: `has_required_files()` - Checks if required L and R files exist before processing
- **New Method**: `has_sufficient_data()` - Validates that a folder has enough data for meaningful analysis
- **Enhanced Logic**: Folders without required files are now skipped instead of causing errors

### 2. Improved Logging and Error Handling
- **Better Warnings**: Clear messages when folders are skipped due to missing data
- **Enhanced Summary**: Added counters for skipped folders and success rate calculation
- **Detailed Logging**: Shows which folders are found and processed vs. skipped

### 3. Directory Cleanup
- **Automatic Cleanup**: Empty directories created during processing are automatically removed
- **Resource Management**: Prevents accumulation of empty folders in the output directory

### 4. Enhanced Statistics
- **New Counter**: `folders_skipped` tracks how many folders were skipped
- **Success Rate**: Calculates and displays the percentage of successfully processed folders
- **Better Summary**: More detailed operation summary with all relevant metrics

## New Methods Added

### `has_required_files(l_base_dir, r_base_dir, prefix, method_freq)`
- Checks if the required L and R directories exist
- Validates that subject folders are present
- Ensures at least one subject has the required file patterns

### `has_sufficient_data(folder_path, condition, method_freq)`
- Validates that a folder has at least 3 subject files for meaningful analysis
- Checks for the presence of overall matrix files
- Returns boolean indicating if analysis should proceed

## Error Prevention
- **FileNotFoundError Prevention**: No more crashes when overall matrix files are missing
- **Empty Directory Prevention**: Automatic cleanup prevents empty output folders
- **Graceful Degradation**: Script continues processing even when some folders are empty

## Usage
The enhanced script works exactly the same as before, but now:
1. Automatically skips folders without sufficient data
2. Provides clear logging about what's being skipped and why
3. Gives a comprehensive summary including success rates
4. Cleans up empty directories automatically

## Testing
A test script (`test_wilcoxon_enhancement.py`) has been created to verify the enhancement functionality.

## Benefits
- **Robustness**: No more crashes due to missing files
- **Clarity**: Clear feedback about what's happening during processing
- **Efficiency**: Only processes folders with actual data
- **Maintainability**: Better organized code with clear separation of concerns 