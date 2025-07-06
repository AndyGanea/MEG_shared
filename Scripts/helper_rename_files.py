import os
from pathlib import Path
import shutil

class FileRenamer:
    def __init__(self):
        """Initialize the File Renamer"""
        self.subjects = ['DOC', 'GB', 'JDC', 'JFXD', 'JZ', 'LT', 'NvA', 'RR', 'SJB', 'BG']
        self.data_dir = Path("Data")
        self.dataset_dir = self.select_dataset()

    def select_dataset(self) -> Path:
        """Let user select which dataset to process"""
        print("\nAvailable datasets:")
        print("==================")
        
        available_dirs = [d for d in self.data_dir.iterdir() if d.is_dir() and d.name.startswith("DataSet")]
        
        for idx, dir_path in enumerate(available_dirs, 1):
            print(f"{idx}. {dir_path.name}")
        
        while True:
            try:
                choice = int(input("\nSelect dataset number: "))
                if 1 <= choice <= len(available_dirs):
                    selected_dir = available_dirs[choice - 1]
                    print(f"\nSelected: {selected_dir.name}")
                    return selected_dir
                else:
                    print("Invalid selection. Please try again.")
            except ValueError:
                print("Please enter a valid number.")

    def count_files(self):
        """Count files in method_frequency folders and subject subfolders"""
        print("\nCurrent File Counts:")
        print("===================")
        
        method_freq_total = 0
        subject_totals = {subject: 0 for subject in self.subjects}
        
        alignment_dirs = [d for d in self.dataset_dir.iterdir() if d.is_dir()]
        for align_dir in alignment_dirs:
            target_dirs = [d for d in align_dir.iterdir() if d.is_dir()]
            for target_dir in target_dirs:
                method_freq_dirs = [d for d in target_dir.iterdir() if d.is_dir()]
                for method_freq_dir in method_freq_dirs:
                    # Count files in method_freq directory
                    method_freq_files = list(method_freq_dir.glob("*_unc.csv"))
                    method_freq_count = len(method_freq_files)
                    method_freq_total += method_freq_count
                    
                    # Count files in subject subfolders
                    for subject in self.subjects:
                        subject_dir = method_freq_dir / subject
                        if subject_dir.exists():
                            subject_files = list(subject_dir.glob("*_unc.csv"))
                            subject_count = len(subject_files)
                            subject_totals[subject] += subject_count
        
        print(f"\nFiles in method_frequency folders: {method_freq_total}")
        print("\nFiles in subject folders:")
        for subject, count in subject_totals.items():
            print(f"{subject}: {count}")
        
        return method_freq_total, subject_totals

    def rename_files(self):
        """Rename files in both method_frequency and subject folders"""
        print("\nStep 1: Renaming files")
        
        # Track statistics
        files_processed = 0
        files_renamed = 0
        failed_files = []
        
        # First, get all method_frequency directories
        alignment_dirs = [d for d in self.dataset_dir.iterdir() if d.is_dir()]
        for align_dir in alignment_dirs:
            alignment = align_dir.name  # 'cue' or 'mov'
            target_dirs = [d for d in align_dir.iterdir() if d.is_dir()]
            for target_dir in target_dirs:
                target = target_dir.name  # 'L' or 'R'
                method_freq_dirs = [d for d in target_dir.iterdir() if d.is_dir()]
                
                for method_freq_dir in method_freq_dirs:
                    method, freq = method_freq_dir.name.split('_')
                    
                    # Process files in both method_freq directory and subject subdirectories
                    for file_path in method_freq_dir.rglob("*_unc.csv"):
                        try:
                            files_processed += 1
                            
                            # Find subject from filename
                            subject = None
                            for subj in self.subjects:
                                if subj in file_path.name:
                                    subject = subj
                                    break
                            
                            if subject:
                                # Parse condition info from the current filename
                                # Current format: BG_gDTF_10Hz_RT_Down_anti_L_unc.csv
                                parts = file_path.stem.split('_')
                                posture = None
                                movement = None
                                
                                # Find posture and movement in filename
                                for i, part in enumerate(parts):
                                    if part in ['Down', 'Pronation', 'Upright']:
                                        posture = f"{parts[i-1]}-{part}"  # e.g., LT-Pronation
                                    elif part in ['anti', 'pro']:
                                        movement = part
                                
                                if posture and movement:
                                    # Create new filename:
                                    # {alignment}_{posture}_{target}_{movement}_{subject}_{method}_{freq}_unc.csv
                                    new_name = f"{alignment}_{posture}_{target}_{movement}_{subject}_{method}_{freq}_unc.csv"
                                    
                                    # Rename the file
                                    new_path = file_path.parent / new_name
                                    if not new_path.exists():
                                        file_path.rename(new_path)
                                        files_renamed += 1
                                        print(f"Renamed: {file_path.name} -> {new_name}")
                                else:
                                    failed_files.append((file_path.name, "Could not determine posture/movement"))
                                    print(f"Warning: Could not determine posture/movement for {file_path.name}")
                            else:
                                failed_files.append((file_path.name, "Could not determine subject"))
                                print(f"Warning: Could not determine subject for {file_path.name}")
                        
                        except Exception as e:
                            failed_files.append((file_path.name, str(e)))
                            print(f"Error processing {file_path.name}: {str(e)}")
        
        # Print summary
        print("\nFile Processing Summary")
        print("=====================")
        print(f"Total files processed: {files_processed}")
        print(f"Files renamed: {files_renamed}")
        
        if failed_files:
            print("\nFailed Files:")
            print("------------")
            for name, reason in failed_files:
                print(f"- {name}: {reason}")

def main():
    renamer = FileRenamer()
    
    print("\nFile Renaming Program")
    print("====================================")
    print(f"Processing dataset: {renamer.dataset_dir.name}")
    print("------------------------------------")
    
    # Count files before
    print("\nBEFORE:")
    before_method_total, before_subject_totals = renamer.count_files()
    
    # Execute renaming
    renamer.rename_files()
    
    # Count files after
    print("\nAFTER:")
    after_method_total, after_subject_totals = renamer.count_files()
    
    # Compare
    print("\nComparison:")
    print("===========")
    print(f"Method_frequency files: {before_method_total} -> {after_method_total}")
    print("\nSubject folder files:")
    for subject in renamer.subjects:
        before = before_subject_totals[subject]
        after = after_subject_totals[subject]
        print(f"{subject}: {before} -> {after}")

if __name__ == "__main__":
    main()
