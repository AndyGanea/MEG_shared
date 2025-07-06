# MEG Analysis Project

## Summer Project 2025 - Getting from raw MEG data to visualizations

Step 1: Organize the raw the data
* Receive the raw data from the MEG system
* Copy the .csv files to the Data folder using the structure: Data\<Data Set Name>\Align_cue\<Individual Subject Data Folder>\<Data File>.csv
This repository contains scripts and data for analyzing Magnetoencephalography (MEG) measurements across different experimental conditions.


Step 2: Prepare data for analysis by creating a new data set structure
* Run script: helper_arrange_measurement_files_for_histogram.py
  * Select the sub-folder, under "Data" that contains the raw files; in our case was Data\MEG_20250620_cue_vols_626_to_1563_order_10
  ** Note: the provided data set contained measurements for frequency of 25 Hz; this frequency was not used before; we enhanced the line of code as such: self.frequencies = ['10Hz', '20Hz', '25Hz', '100Hz'] in the constructor of the class FileOrganizer
  * Select the sub-folder under the Data\<data raw folder>; in our case only 1 sub-folder was available: Align_cue
  * We provided a new incremenatal number (14 in our case) to be used in naming the newly created Data set ready for analysis
  * No unc files were provided, so we answer No (n) at the question to process 'unc' files
  * Result was a new folder, named: Data\DataSet14_Align_cue containing only 1 sub-folder, 'cue', then the 2 separate sub-folders, 'L' and 'R', then under each L and R folders are <method><frequency> subfolder, then subject names subfolders, and then individual measurement files.
  ** Example of a path (branch) created: 'MEG\Data\DataSet14_Align_cue\cue\L\gPDC_10Hz\BG'

Step 3: Create average matrices
* 28-June-2025: created a v2 of the script average named test_average_matrix_v2.py; in essence the v2 script can exculde from averaging the files containing the "LT-Pronation" string. For more detailed documentation for v2 of the script, refer to V2_ENHANCEMENT_SUMMARY.md file
* Run script: create_average_matrix_enhanced.py
  * Select the Data folder; in our example we selected: 'DataSet14_Align_cue'
  * Answer 'n' at the question to process DTF files because there are none in this sample
* The script creates an average matrix per <method><frequency> combination and another average file per indidual subject; example of average files we created: 'cue_L_gPDC_10Hz_average.csv' and 'cue_L_gPDC_10Hz_BG_average.csv' ('BG' = a subject)

Step 4: Run Wilcoxon analysis and create Wilcoxon files and heatmaps
* 28-June-2025: created a v2 of the script named meg_wilcoxon_analysis_V2.py; this v2 script added the option for the user to select the full average file vs the NO-LT average files for the Wilcoxon analysis. or more detailed documentation for v2 of the script, refer to WILCOXON_V2_ENHANCEMENT_SUMMARY.md file.
* Run script meg_wilcoxon_analysis.py
** Note: we added the '25Hz' method in line 'for freq in ['10Hz', '20Hz', '25Hz', '100Hz']:' under method 'def prepare_wilcoxon_data(self):'
** Note: we enhanced the script to avoid errors when some of the original subfolders contain no data files
* Select the data folder (6. DataSet14_Align_cue in our case)
* Answer 'y' to exclude the iDTF method (in our case)
* The script creates a Wilcoxon subfolder under selected data set folder and then L-R differences files, a heatmap and a log file in each folder; sample subfolder: \DataSet14_Align_cue\Wilcoxon_06222025-1455\Cue_gPDC_10Hz

Step 5 - Manually move the wilcoxon CSV files into Data\Visual folder to be analysed
* Example: file gPDC_10Hz_L-R_wilcoxon.csv moved into Data\Visual folder


Step 6 - Run the circular connectogram and 3D brain scripts and create the 2 types of diagrams: 1) circular connectivity and 3D brain diagrams
* * 28-June-2025: created a v2 of the scripts named meg_visualization_circular_select_V2.py and meg_visualization_3DBrain_select_V2.py respectively. The V2 versions will be able to detect in the Visual folders if there are NO-LT files and they will include them in the slection options. The script groups the files in 2 categories: 1) full files and 2) NO-LT files.
* Scripts run: meg_visualization_circular_select.py and meg_visualization_3DBrain_select.py

## Project Structure


## Measurement Variables

### Experimental Conditions
- **Alignment**: ['cue', 'mov']
- **Posture**: ['RT_Pronation', 'RT_Down', 'RT_Upright', 'LT_Pronation']
- **Movement**: ['pro', 'anti']
- **Target**: ['L', 'R']

### Analysis Parameters
- **Subject IDs**: ['DOC', 'GB', 'JDC', 'JFXD', 'JZ', 'LT', 'NvA', 'RR', 'SJB', 'BG']
- **Analysis Methods**: ['iDTF', 'gDTF', 'iPDC', 'gPDC']
- **Frequency Bands**: [10Hz, 20Hz, 100Hz]

## Scripts

The `Scripts` folder contains Python scripts for data analysis and visualization:
- `organize_meg_folders.py`: Organizes project files into appropriate directories
- Additional analysis scripts (list and describe your other scripts here)

## Data Organization

### Heatmaps and Plots
- The `Heatmaps_and_Plots` directory contains generated visualizations
  - `Heatmaps/`: Connectivity and analysis heatmaps
  - `Plots/`: Other data visualizations and results

### Data Storage
- `Data/`: Contains processed and raw MEG data
- `Sample_Data/`: Example datasets for testing and validation
- `MEG_20250117_group_project_cat/`: Project-specific datasets

## Usage

1. Ensure all dependencies are installed
2. Place raw MEG data in the appropriate directory
3. Run analysis scripts from the `Scripts` folder
4. Generated visualizations will be saved in `Heatmaps_and_Plots`

## Dependencies

(List your project's dependencies here)

## Contributing

(Add contribution guidelines if this is a collaborative project)

## License

(Add license information if applicable)
