import pandas as pd
import os

# --- PHASE 2: DATA PRIMARY CLEANING AND TRANSFORMATION ---

def execute_cleaning_pipeline(input_file, output_file):
    """
    Standardizes text data and removes spam/duplicates based on 
    specific project requirements to ensure data integrity.
    """
    
    # Verify source file existence
    if not os.path.exists(input_file):
        print(f"CRITICAL ERROR: Source file '{input_file}' not found.")
        return

    # Load Raw Dataset
    df = pd.read_csv(input_file)
    initial_count = len(df)
    
    print("SYSTEM: Initiating Data Transformation Pipeline...")
    print(f"STATUS: Initial Data Volume: {initial_count} records.")

    # 1. Text Normalization
    # Trimming whitespace and ensuring string format for consistency
    df['Comment_Content'] = df['Comment_Content'].astype(str).str.strip()
    
    # 2. Deduplication (Spam Filtering)
    # Logic: Remove records where BOTH 'Author' and 'Comment_Content' are identical.
    # This keeps different comments from the same author as requested.
    df_cleaned = df.drop_duplicates(subset=['Author', 'Comment_Content'], keep='first')
    
    # 3. Noise Reduction
    # Removing entries shorter than 3 characters that lack analytical value.
    df_cleaned = df_cleaned[df_cleaned['Comment_Content'].str.len() > 2]

    # Metrics Calculation
    final_count = len(df_cleaned)
    removed_records = initial_count - final_count

    # Export Processed Dataset
    df_cleaned.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print("\n" + "="*40)
    print("CLEANING PIPELINE SUMMARY")
    print("="*40)
    print(f"Total Raw Records      : {initial_count}")
    print(f"Spam/Noise Eliminated  : {removed_records}")
    print(f"Actionable Data Points : {final_count}")
    print(f"Export Status          : Success")
    print(f"Final File Destination : {output_file}")
    print("="*40)

if __name__ == "__main__":
    # Define file paths
    RAW_DATA = 'GP_Final_Research_Data.csv'
    CLEANED_DATA = 'GP_Cleaned_Data.csv'
    
    execute_cleaning_pipeline(RAW_DATA, CLEANED_DATA)