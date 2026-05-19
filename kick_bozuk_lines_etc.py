import pandas as pd
import os
from pathlib import Path

def main():
    # Get the directory where the script is located dynamically
    try:
        script_dir = Path(__file__).resolve().parent
    except NameError:
        script_dir = Path(os.getcwd()).resolve()

    file_name = "FINAL_MERGED_RFFI_DATASET.csv"
    file_path = script_dir / file_name
    
    print(f"Loading data: {file_name}...")
    
    if not file_path.exists():
        print(f"[ERROR] {file_name} could not be found in the root directory!")
        return
        
    df = pd.read_csv(file_path)

    initial_row_count = len(df)
    print(f"Total rows before cleaning: {initial_row_count}")

    # 1. Keep only packets with 'Geçerli' status (Filter out corrupted ones)
    df_cleaned = df[df['Packet_Status'] == 'Geçerli'].copy()

    final_row_count = len(df_cleaned)
    dropped_rows = initial_row_count - final_row_count

    print(f"Number of corrupted packets dropped: {dropped_rows}")
    print(f"Remaining valid rows: {final_row_count}")

    # 2. Drop unwanted columns
    columns_to_drop = ['Source_File', '78:21:84:93:59:44']
    
    # Drop only the columns that exist in the dataframe
    df_cleaned.drop(columns=columns_to_drop, errors='ignore', inplace=True)
    
    print(f"Cleaning complete. Remaining columns: {list(df_cleaned.columns)}")

    # 3. Save the cleaned dataset
    new_file_name = "FINAL_MERGED_RFFI_DATASET_CLEANED.csv"
    new_file_path = script_dir / new_file_name
    
    df_cleaned.to_csv(new_file_path, index=False, encoding='utf-8')

    print(f"\n[SUCCESS] Corrupted rows dropped, specified columns removed. Ready-to-train dataset saved to:\n👉 {new_file_path}")

if __name__ == "__main__":
    main()