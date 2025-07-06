import os
from sample_test import MEGConnectivityAnalysis

def list_csv_files_for_subject(base_dir, subject):
    """List all CSV files in the subject's directory"""
    subject_dir = os.path.join(base_dir, subject)
    
    if not os.path.exists(subject_dir):
        return None
    
    csv_files = []
    # Only look in the specific subject directory, not in subdirectories
    for file in os.listdir(subject_dir):
        if file.endswith('.csv'):
            full_path = os.path.abspath(os.path.join(subject_dir, file))
            csv_files.append(full_path)
    return csv_files

def main():
    # Get absolute paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(current_dir, "Data"))
    
    # Ask for subject ID
    subject = input("\nEnter subject ID (e.g., BG): ").strip()
    
    # Find CSV files for the specific subject
    csv_files = list_csv_files_for_subject(data_dir, subject)
    
    if csv_files is None:
        print(f"\nError: Subject directory 'Data\\{subject}' not found!")
        return
    
    if not csv_files:
        print(f"\nNo CSV files found in Data\\{subject}")
        return
    
    print(f"\nLooking for data in: Data\\{subject}")
    
    # Display available files
    print("\nAvailable CSV files:")
    for i, file_path in enumerate(csv_files):
        print(f"{i+1}. {os.path.basename(file_path)}")  # Show only filename, not full path
    
    try:
        # Let user select files
        print("\nSelect files by number:")
        cue_idx = int(input("Enter number for cue file: ")) - 1
        mov_idx = int(input("Enter number for movement file: ")) - 1
        
        if 0 <= cue_idx < len(csv_files) and 0 <= mov_idx < len(csv_files):
            cue_file = csv_files[cue_idx]
            mov_file = csv_files[mov_idx]
            
            # Create instance
            meg = MEGConnectivityAnalysis(data_dir)
            
            # Load selected files using absolute paths
            print("\nLoading files:")
            print(f"Cue: {os.path.basename(cue_file)}")
            print(f"Movement: {os.path.basename(mov_file)}")
            
            meg.load_two_matrices(cue_file, mov_file)
            
            # Ask about strong connections
            show_strong = input("\nShow only strongest connections (top 10%)? (Y/N): ").strip().upper()
            threshold_percentile = 90 if show_strong == 'Y' else 40  # 40% is the default
            
            # Ask user what they want to do
            print("\nWhat would you like to do?")
            print("1. Create GIF animation")
            print("2. View brain connectivity")
            print("3. Both")
            
            choice = input("Enter your choice (1-3): ")
            
            if choice == '1' or choice == '3':
                print("\nCreating animation...")
                meg.create_transition_animation()
                
            if choice == '2' or choice == '3':
                print("\nDisplaying brain connectivity...")
                print("(Three figures will be shown: Cue, Movement, and Side-by-Side Comparison)")
                meg.plot_connectivity_comparison(threshold_percentile=threshold_percentile)
            
        else:
            print("Invalid file selection")
            
    except ValueError:
        print("Please enter valid numbers")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        print(f"Working directory: {os.getcwd()}")
        print(f"Data directory: {data_dir}")

if __name__ == "__main__":
    main()