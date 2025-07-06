# Average Matrix Creator V2 Enhancement Summary

## Overview
The `create_average_matrix_enhanced_V2.py` script has been enhanced with a new feature to exclude LT-Pronation files from averaging, providing more control over data processing.

## New Feature: LT-Pronation File Exclusion

### User Interface
- **New Prompt**: "Exclude LT-Pronation files? (y/n)"
- **User Choice**: Allows users to selectively exclude files containing "LT-Pronation" in their names

### File Detection Logic
- **Detection Method**: `is_lt_pronation_file(filename)` - Checks if filename contains "LT-Pronation"
- **Examples of Detected Files**:
  - `cue_LT-Pronation_L_anti_DOC_gPDC_20Hz.csv`
  - `mov_LT-Pronation_R_pro_JZ_gDTF_10Hz.csv`
- **Examples of Non-Detected Files**:
  - `cue_RT_Pronation_L_anti_DOC_gPDC_20Hz.csv` (RT-Pronation, not LT)
  - `mov_normal_file_gDTF_10Hz.csv` (no pronation reference)

### File Naming Convention
When user selects to exclude LT-Pronation files, the `_NO-LT` suffix is added to output filenames:

#### Overall Averages:
- **Normal**: `cue_L_gPDC_20Hz_average.csv`
- **With LT Exclusion**: `cue_L_gPDC_20Hz_average_NO-LT.csv`
- **With unc suffix**: `cue_L_gPDC_20Hz_average_unc_NO-LT.csv`

#### Subject Averages:
- **Normal**: `cue_L_gPDC_20Hz_GB_average.csv`
- **With LT Exclusion**: `cue_L_gPDC_20Hz_GB_average_NO-LT.csv`
- **With unc suffix**: `cue_L_gPDC_20Hz_GB_average_unc_NO-LT.csv`

### Processing Logic
1. **File Filtering**: Files containing "LT-Pronation" are excluded from averaging when requested
2. **Suffix Application**: `_NO-LT` suffix is added regardless of whether LT-Pronation files were actually found
3. **Scope**: Applied to both overall averages and subject-specific averages

## Enhanced Logging and Statistics

### New Logging Features:
- **File Exclusion Logging**: Logs each LT-Pronation file that is excluded
- **Summary Logging**: Reports total number of excluded files per directory/subject
- **No Files Found**: Logs "No LT-Pronation files found" when none exist
- **Warning System**: Warns user if they choose to exclude LT-Pronation but no such files exist

### New Statistics:
- **Counter**: `lt_pronation_files_excluded` tracks total excluded files
- **Summary Display**: Shows excluded file count in final summary (only when LT exclusion is enabled)

## New Methods Added

### `ask_exclude_lt_pronation() -> bool`
- Prompts user for LT-Pronation exclusion preference
- Returns True if user wants to exclude, False otherwise

### `is_lt_pronation_file(filename: str) -> bool`
- Checks if filename contains "LT-Pronation" string
- Returns True if LT-Pronation is found, False otherwise

### `should_include_file(filename: str) -> bool`
- Determines if file should be included based on user preferences
- Returns False if LT-Pronation exclusion is enabled and file contains "LT-Pronation"

### `check_lt_pronation_files_exist() -> bool`
- Scans entire dataset for LT-Pronation files
- Logs found files and returns True if any exist
- Used for warning system

## Error Prevention and User Experience

### Warning System:
- **No Files Warning**: Warns user if they choose to exclude LT-Pronation but none exist
- **Clear Messaging**: Explains that `_NO-LT` suffix will still be applied

### Robust Processing:
- **Graceful Handling**: Continues processing even when no LT-Pronation files are found
- **Consistent Naming**: Always applies suffix when exclusion is enabled, regardless of file presence

## Usage Examples

### Scenario 1: LT-Pronation files exist and user excludes them
```
User selects: "Exclude LT-Pronation files? (y/n): y"
Result: LT-Pronation files are excluded, output files get _NO-LT suffix
```

### Scenario 2: No LT-Pronation files exist but user excludes them
```
User selects: "Exclude LT-Pronation files? (y/n): y"
Result: Warning is shown, output files still get _NO-LT suffix
```

### Scenario 3: User includes LT-Pronation files
```
User selects: "Exclude LT-Pronation files? (y/n): n"
Result: All files are included, no suffix added to output files
```

## Testing
A test script (`test_average_matrix_v2.py`) has been created to verify:
- LT-Pronation file detection
- File inclusion/exclusion logic
- Filename generation with proper suffixes
- All enhancement functionality

## Benefits
- **Data Control**: Users can selectively exclude specific movement types
- **Clear Naming**: Output files clearly indicate what data was excluded
- **Consistent Behavior**: Predictable naming regardless of file presence
- **Detailed Logging**: Full transparency about what files were processed/excluded
- **User-Friendly**: Clear warnings and informative messages 