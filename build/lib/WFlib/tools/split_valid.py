import numpy as np
import os

def split_valid_to_test(valid_path, test_path, test_ratio=0.2):
    if not os.path.exists(valid_path):
        print(f"File not found: {valid_path}")
        return

    # Load the data
    data = np.load(valid_path)
    # Convert NpzFile object to a dictionary to access arrays easily
    data_dict = {key: data[key] for key in data.files}
    
    # Assuming all arrays have the same length in the first dimension (batch dimension)
    # We use the first key to determine the total number of samples
    first_key = data.files[0]
    n_samples = len(data_dict[first_key])
    
    # Calculate split index
    n_test = int(n_samples * test_ratio)
    n_valid = n_samples - n_test
    
    # Shuffle indices for a random split (optional but recommended)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    test_indices = indices[:n_test]
    valid_indices = indices[n_test:]
    
    test_data = {}
    new_valid_data = {}
    
    for key, array in data_dict.items():
        test_data[key] = array[test_indices]
        new_valid_data[key] = array[valid_indices]
        
    # Save the new files
    np.savez_compressed(test_path, **test_data)
    np.savez_compressed(valid_path, **new_valid_data) # Overwrite original valid.npz
    
    print(f"Split complete.")
    print(f"Original samples: {n_samples}")
    print(f"New Valid samples: {len(valid_indices)}")
    print(f"Test samples: {len(test_indices)}")
    print(f"Saved test data to: {test_path}")
    print(f"Updated valid data at: {valid_path}")

if __name__ == "__main__":
    valid_file = "valid.npz"
    test_file = "test.npz"
    
    split_valid_to_test(valid_file, test_file, test_ratio=0.2)