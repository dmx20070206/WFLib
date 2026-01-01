import os
import json
import argparse
import matplotlib.pyplot as plt
import matplotlib

# Set font style to match the paper (Times New Roman style)
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 14

def plot_temporal_drift(model_name):
    # Base directory
    base_dir = f"logs/TemporalDrift/{model_name}"
    
    # Constants (Added 150 based on the screenshot)
    days = [14, 30, 90, 150, 270]
    
    # Lists to store the extracted scores
    standard_scores = []
    proteus_scores = []
    
    # Iterate through the days to read files
    for day in days:
        file_standard = os.path.join(base_dir, f"day{day}.json")
        file_proteus = os.path.join(base_dir, f"Proteus_day{day}.json")
        
        # Helper function to read the Accuracy from a JSON dict
        def get_accuracy(filepath):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    
                    score = 0
                    # (1) Fix error: Handle dictionary format
                    if isinstance(data, dict):
                        # Prioritize best_accuracy, fallback to Accuracy
                        if "best_accuracy" in data:
                            score = data["best_accuracy"]
                        else:
                            score = data.get("Accuracy", 0)
                    
                    # Compatible with old format (if it is a list)
                    elif isinstance(data, list) and len(data) > 0:
                        last_item = data[-1]
                        if isinstance(last_item, dict):
                            if "best_accuracy" in last_item:
                                score = last_item["best_accuracy"]
                            else:
                                score = last_item.get("Accuracy", 0)
                    else:
                        print(f"Warning: Unexpected data format in {filepath}")
                        return 0
                    
                    # Assuming score is 0-1 range, convert to percentage
                    return score * 100
                    
            except FileNotFoundError:
                print(f"Warning: File not found {filepath}")
                return 0
            except json.JSONDecodeError:
                print(f"Error: Invalid JSON in {filepath}")
                return 0

        standard_scores.append(get_accuracy(file_standard))
        proteus_scores.append(get_accuracy(file_proteus))

    # (2) Style replication
    plt.figure(figsize=(6, 5)) # Adjust aspect ratio to be close to the screenshot
    
    # Create equidistant x-axis indices so points are evenly spaced
    x_indices = range(len(days))

    # Proteus line: Red solid line, circle marker, hollow (white fill)
    plt.plot(x_indices, proteus_scores, 
             color='#d62728',       # Red
             linestyle='-',         # Solid line
             linewidth=2.5,         # Line width
             marker='o',            # Circle marker
             markersize=10,         # Marker size
             markerfacecolor='white', # Marker fill white (hollow effect)
             markeredgewidth=2,     # Marker edge width
             label='Proteus')
    
    # Standard line: Blue dashed line, square marker, hollow (white fill)
    plt.plot(x_indices, standard_scores, 
             color='#1f77b4',       # Blue
             linestyle='--',        # Dashed line
             linewidth=2.5,         # Line width
             marker='s',            # Square marker
             markersize=10,         # Marker size
             markerfacecolor='white', # Marker fill white
             markeredgewidth=2,     # Marker edge width
             label='Standard')
    
    # Axis configuration
    plt.xlabel('Time (days)', fontsize=16)
    # plt.ylabel('Accuracy (%)', fontsize=16) # No text label on Y-axis in screenshot, only numbers
    
    # Set xticks to use the indices but display the day values as labels
    plt.xticks(x_indices, days, fontsize=14)
    plt.yticks(fontsize=14)
    
    # Set Y-axis range adaptive
    all_scores = standard_scores + proteus_scores
    if all_scores:
        min_val = min(all_scores)
        max_val = max(all_scores)
        # Add 10% margin for better visualization
        margin = (max_val - min_val) * 0.1
        if margin == 0: margin = 5 # Default margin if flat line
        
        # Ensure limits are within 0-100 range
        plt.ylim(max(0, min_val - margin), min(100, max_val + margin))
    
    # Grid lines
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Title as bottom label
    plt.title(f"{model_name}", y=-0.25, fontsize=18)
    
    # Layout adjustment
    plt.tight_layout()
    
    # Save
    os.makedirs("plots/TemporalDrift", exist_ok=True)    
    output_path = f"plots/TemporalDrift/{model_name}.png"
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Temporal Drift based on model name.")
    parser.add_argument("model", type=str, help="The name of the model (folder name).")
    
    args = parser.parse_args()
    
    if os.path.exists(f"logs/TemporalDrift/{args.model}"):
        plot_temporal_drift(args.model)
    else:
        print(f"Directory logs/TemporalDrift/{args.model} not found.")