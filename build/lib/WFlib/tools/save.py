import os
import shutil
import argparse

def save_archive(drift_name, save_name):
    """
    Archives specific folders related to a drift experiment into a new save location.
    
    Args:
        drift_name (str): The name of the drift experiment (source folder suffix).
        save_name (str): The name for the archive (destination folder suffix).
    """
    # Define source base directories
    source_bases = ['checkpoints', 'logs', 'plots']
    
    # Define the root archive directory
    archive_root = os.path.join('archive', save_name)
    
    # Create the archive root if it doesn't exist
    if not os.path.exists(archive_root):
        os.makedirs(archive_root)
        print(f"Created archive directory: {archive_root}")
    
    for base in source_bases:
        # Construct source path: base/{drift_name}/
        src_path = os.path.join(base, drift_name)
        
        # Construct destination path: archive/{save_name}/{base}/
        # Note: The requirement is to remove the {df} level, so contents go directly into base
        dst_path = os.path.join(archive_root, base)
        
        if os.path.exists(src_path):
            # If destination exists, remove it to ensure clean copy or handle merge
            if os.path.exists(dst_path):
                shutil.rmtree(dst_path)
            
            # Copy the directory tree
            shutil.copytree(src_path, dst_path)
            print(f"Copied {src_path} to {dst_path}")
        else:
            print(f"Warning: Source directory {src_path} does not exist. Skipping.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Archive experiment files.")
    parser.add_argument("DriftName", type=str, help="The name of the drift experiment (source).")
    parser.add_argument("SaveName", type=str, help="The name of the archive (destination).")
    
    args = parser.parse_args()
    
    save_archive(args.DriftName, args.SaveName)