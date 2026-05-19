import pandas as pd
import os
from pathlib import Path

def main():
    # Get the directory where the script is located dynamically
    try:
        script_dir = Path(__file__).resolve().parent
    except NameError:
        script_dir = Path(os.getcwd()).resolve()

    # Files to be merged
    file1_name = "123_devices_merged_first_features_clean.csv"
    file2_name = "RFFI_all_second_extracted_features.csv"
    output_name = "FINAL_MERGED_RFFI_DATASET.csv"

    file1_path = script_dir / file1_name
    file2_path = script_dir / file2_name
    output_path = script_dir / output_name

    print(f"Reading [{file1_name}]...")
    try:
        df1 = pd.read_csv(file1_path)
        print(f"  -> Success. Shape: {df1.shape}")
    except FileNotFoundError:
        print(f"  [ERROR] {file1_name} not found!")
        return

    print(f"Reading [{file2_name}]...")
    try:
        df2 = pd.read_csv(file2_path)
        print(f"  -> Success. Shape: {df2.shape}")
    except FileNotFoundError:
        print(f"  [ERROR] {file2_name} not found!")
        return

    # Safety check to ensure row counts match
    if len(df1) != len(df2):
        print("\n[WARNING] Row counts of the files do not match!")
        print(f"  File 1: {len(df1)} rows")
        print(f"  File 2: {len(df2)} rows")
        print("  Merging anyway, but missing rows will be filled with 'NaN' (empty).")

    # Find common columns
    common_columns = list(set(df1.columns).intersection(set(df2.columns)))
    
    # Drop common identification columns from the second dataset to avoid duplication
    columns_to_drop = [col for col in common_columns if col in ['Device Number', 'MAC_address', 'Packet_Status']]
    
    if columns_to_drop:
        print(f"\nDropping duplicate columns from '{file2_name}' to avoid mismatch: {columns_to_drop}")
        df2_cleaned = df2.drop(columns=columns_to_drop)
    else:
        df2_cleaned = df2

    print("\nMerging datasets side-by-side...")
    # Concatenate side-by-side using axis=1
    final_merged_df = pd.concat([df1, df2_cleaned], axis=1)

    print(f"Merge complete. New dataset shape: {final_merged_df.shape}")

    # Save the result as a new CSV file
    print("\nSaving new file...")
    final_merged_df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"[SUCCESS] Final dataset containing all features created:\n👉 {output_path}")

if __name__ == "__main__":
    main()