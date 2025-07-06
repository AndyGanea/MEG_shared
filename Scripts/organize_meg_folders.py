import os
import shutil
from pathlib import Path

def create_folder_structure():
    # Define the base paths
    base_dir = os.getcwd()
    data_dir = os.path.join(base_dir, 'Data')
    scatter_plots_dir = os.path.join(data_dir, 'scatter_plots')
    
    # Create folders
    folders = {
        'Scripts': [''],
        'Heatmaps_and_Plots': ['Heatmaps', 'Plots', 'Connectivity_Transitions'],
        'Logs': ['']
    }
    
    # Create all folders if they don't exist
    for main_folder, subfolders in folders.items():
        main_path = os.path.join(base_dir, main_folder)
        os.makedirs(main_path, exist_ok=True)
        
        for subfolder in subfolders:
            if subfolder:
                subfolder_path = os.path.join(main_path, subfolder)
                os.makedirs(subfolder_path, exist_ok=True)

    # Move scatter plots content
    if os.path.exists(scatter_plots_dir):
        plots_dir = os.path.join(base_dir, 'Heatmaps_and_Plots', 'Plots')
        print(f"Source directory: {scatter_plots_dir}")
        print(f"Destination directory: {plots_dir}")
        
        # Create plots directory if it doesn't exist
        os.makedirs(plots_dir, exist_ok=True)
        
        # Walk through all subdirectories
        for root, dirs, files in os.walk(scatter_plots_dir):
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    # Get relative path from scatter_plots_dir
                    rel_path = os.path.relpath(root, scatter_plots_dir)
                    # Create corresponding subfolder in plots_dir
                    dst_subfolder = os.path.join(plots_dir, rel_path)
                    os.makedirs(dst_subfolder, exist_ok=True)
                    
                    src = os.path.join(root, file)
                    dst = os.path.join(dst_subfolder, file)
                    
                    try:
                        shutil.copy2(src, dst)  # First copy
                        if os.path.exists(dst):  # Verify copy succeeded
                            os.remove(src)  # Then delete original
                            print(f"Successfully moved {file} to {dst_subfolder}")
                    except Exception as e:
                        print(f"Error moving {file}: {str(e)}")
        
        # Remove empty directories
        try:
            for root, dirs, files in os.walk(scatter_plots_dir, topdown=False):
                for dir_name in dirs:
                    dir_path = os.path.join(root, dir_name)
                    try:
                        os.rmdir(dir_path)
                        print(f"Removed empty directory: {dir_path}")
                    except OSError:
                        print(f"Directory not empty or cannot be removed: {dir_path}")
            
            # Try to remove the main scatter_plots directory
            if os.path.exists(scatter_plots_dir):
                os.rmdir(scatter_plots_dir)
                print("Removed empty scatter_plots directory")
        except OSError as e:
            print(f"Could not remove some directories: {e}")

    # Function to find all Python files recursively
    def find_python_files(directory):
        python_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.py'):
                    python_files.append(os.path.join(root, file))
        return python_files

    # Move Python files from Data folder to Scripts
    python_files = find_python_files(data_dir)
    for py_file in python_files:
        filename = os.path.basename(py_file)
        dst = os.path.join(base_dir, 'Scripts', filename)
        shutil.move(py_file, dst)
        print(f"Moved {filename} to Scripts folder")

    # Move image files to appropriate folders
    image_files = [f for f in os.listdir(base_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
    for img_file in image_files:
        if 'heatmap' in img_file.lower():
            dst_folder = os.path.join(base_dir, 'Heatmaps_and_Plots', 'Heatmaps')
        elif img_file.lower().endswith('.gif'):
            dst_folder = os.path.join(base_dir, 'Heatmaps_and_Plots', 'Connectivity_Transitions')
        else:
            dst_folder = os.path.join(base_dir, 'Heatmaps_and_Plots', 'Plots')
        
        src = os.path.join(base_dir, img_file)
        dst = os.path.join(dst_folder, img_file)
        shutil.move(src, dst)

    # Move log files to Logs folder
    log_files = [f for f in os.listdir(base_dir) if f.lower().endswith('.log')]
    for log_file in log_files:
        src = os.path.join(base_dir, log_file)
        dst = os.path.join(base_dir, 'Logs', log_file)
        shutil.move(src, dst)

    # Move heatmap files from Data directory
    if os.path.exists(data_dir):
        heatmaps_dir = os.path.join(base_dir, 'Heatmaps_and_Plots', 'Heatmaps')
        print(f"\nSearching for heatmap files in: {data_dir}")
        print(f"Moving heatmaps to: {heatmaps_dir}")
        
        # Create heatmaps directory if it doesn't exist
        os.makedirs(heatmaps_dir, exist_ok=True)
        
        # Walk through all subdirectories in Data
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.png') and 'heatmap' in file.lower():
                    src = os.path.join(root, file)
                    dst = os.path.join(heatmaps_dir, file)
                    
                    try:
                        shutil.copy2(src, dst)  # First copy
                        if os.path.exists(dst):  # Verify copy succeeded
                            os.remove(src)  # Then delete original
                            print(f"Successfully moved heatmap: {file}")
                    except Exception as e:
                        print(f"Error moving heatmap {file}: {str(e)}")

    print("Folder organization complete!")

if __name__ == "__main__":
    create_folder_structure() 