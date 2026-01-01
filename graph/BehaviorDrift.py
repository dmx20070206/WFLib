import os
import json
import pandas as pd

def read_json_metrics(file_path, f1_key='F1-score'):
    """Helper to read metrics from a JSON file safely."""
    if not os.path.exists(file_path):
        return None, None, None
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            p = data.get('Precision', 0)
            r = data.get('Recall', 0)
            f1 = data.get(f1_key, 0)
            return p, r, f1
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None, None, None

def safe_sub(val_a, val_b):
    """Safely subtract val_b from val_a (a - b). Returns None if either is None."""
    if val_a is not None and val_b is not None:
        return val_a - val_b
    return None

def format_change_value(val):
    """Format change value with +/- sign."""
    if val is None:
        return None
    return f"{val:+.4f}" if val != 0 else "0.0000"

def main():
    base_dir = 'logs/BehaviorDrift/'
    
    # List to store rows for the DataFrame
    rows = []

    # Iterate through directories in the base path
    if not os.path.exists(base_dir):
        print(f"Directory {base_dir} does not exist.")
        return

    # Get all subdirectories (model names)
    model_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    model_dirs.sort() # Sort for consistent output

    for model_name in model_dirs:
        model_path = os.path.join(base_dir, model_name)
        
        # --- Row 1: Standard Model ---
        # Block 1: Homepage (test.json)
        hp_p, hp_r, hp_f1 = read_json_metrics(
            os.path.join(model_path, 'test.json'), 
            f1_key='F1-score'
        )
        
        # Block 2: Subpages (subpage.json)
        sp_p, sp_r, sp_f1 = read_json_metrics(
            os.path.join(model_path, 'subpage.json'), 
            f1_key='F1-score'
        )

        rows.append({
            'Model': model_name,
            'HP_P': hp_p, 'HP_R': hp_r, 'HP_F1': hp_f1,
            'SP_P': sp_p, 'SP_R': sp_r, 'SP_F1': sp_f1,
            'IsChange': False
        })

        # --- Row 2: Model-Proteus ---
        # Block 1: Homepage (Proteus_test.json)
        # Note: F1 key is 'best_f1_score' here
        prot_hp_p, prot_hp_r, prot_hp_f1 = read_json_metrics(
            os.path.join(model_path, 'Proteus_test.json'), 
            f1_key='best_f1_score'
        )
        
        # Block 2: Subpages (Proteus_subpage.json)
        # Note: F1 key is 'best_f1_score' here
        prot_sp_p, prot_sp_r, prot_sp_f1 = read_json_metrics(
            os.path.join(model_path, 'Proteus_subpage.json'), 
            f1_key='best_f1_score'
        )

        rows.append({
            'Model': f"{model_name}-Proteus",
            'HP_P': prot_hp_p, 'HP_R': prot_hp_r, 'HP_F1': prot_hp_f1,
            'SP_P': prot_sp_p, 'SP_R': prot_sp_r, 'SP_F1': prot_sp_f1,
            'IsChange': False
        })
        
        # --- Row 3: Model-Change (Proteus - Standard) ---
        rows.append({
            'Model': f"{model_name}-Change",
            'HP_P': format_change_value(safe_sub(prot_hp_p, hp_p)),
            'HP_R': format_change_value(safe_sub(prot_hp_r, hp_r)),
            'HP_F1': format_change_value(safe_sub(prot_hp_f1, hp_f1)),
            'SP_P': format_change_value(safe_sub(prot_sp_p, sp_p)),
            'SP_R': format_change_value(safe_sub(prot_sp_r, sp_r)),
            'SP_F1': format_change_value(safe_sub(prot_sp_f1, sp_f1)),
            'IsChange': True
        })

    # Create MultiIndex DataFrame for better formatting
    df = pd.DataFrame(rows)
    
    # Set the Model column as index
    if not df.empty:
        df.set_index('Model', inplace=True)
        
        # Separate IsChange column before creating MultiIndex
        is_change = df['IsChange']
        df = df.drop('IsChange', axis=1)
        
        # Create a MultiIndex for columns to group by Homepage and Subpages
        columns = [
            ('Homepage', 'P'), ('Homepage', 'R'), ('Homepage', 'F1'),
            ('Subpages', 'P'), ('Subpages', 'R'), ('Subpages', 'F1')
        ]
        df.columns = pd.MultiIndex.from_tuples(columns)

        # Display the table with styling
        print("\n" + "="*80)
        print("Behavior Drift Analysis Results")
        print("="*80 + "\n")
        
        # Print with color and formatting
        for idx, (model_name, row) in enumerate(df.iterrows()):
            if idx > 0 and idx % 3 == 0:
                print("\n" + "-"*80 + "\n")  # Separator between groups
            
            if is_change.iloc[idx]:
                # Change row: bold, red, with +/- preserved
                print(f"\033[1;31m{model_name:25s}\033[0m", end=" ")
                for val in row:
                    val_str = str(val) if val is not None else "N/A"
                    print(f"\033[1;31m{val_str:>10s}\033[0m", end=" ")
                print()
            else:
                # Normal rows
                print(f"{model_name:25s}", end=" ")
                for val in row:
                    val_str = f"{val:.4f}" if val is not None else "N/A"
                    print(f"{val_str:>10s}", end=" ")
                print()
        
        print("\n" + "="*80 + "\n")
        
        # Save to CSV
        output_file = 'plots/BehaviorDrift/results.csv'
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df.to_csv(output_file)
        print(f"Table saved to {output_file}")
        
        # Also save a styled HTML version
        html_file = 'plots/BehaviorDrift/results.html'
        
        def highlight_change_rows(row):
            model_name = row.name
            if '-Change' in model_name:
                return ['color: red; font-weight: bold'] * len(row)
            return [''] * len(row)
        
        styled_df = df.style.apply(highlight_change_rows, axis=1)
        styled_df.to_html(html_file)
        print(f"Styled HTML table saved to {html_file}")
    else:
        print("No data found.")

if __name__ == "__main__":
    main()