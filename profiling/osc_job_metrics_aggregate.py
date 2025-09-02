import pandas as pd
import argparse
import glob
import os

def aggregate_augmented_csvs(input_directory, output_csv_path):
    """
    Aggregates multiple CSV files ending with '_aug.csv' from a specified directory
    into a single CSV file.

    Args:
        input_directory (str): The path to the directory containing the CSV files.
        output_csv_path (str): The path to save the aggregated CSV file.
    """
    if not os.path.isdir(input_directory):
        print(f"Error: Input directory '{input_directory}' not found.")
        return

    # Construct the search pattern for files ending with _aug.csv
    file_pattern = "*cleaned_aug.csv"
    search_pattern = os.path.join(input_directory, file_pattern)
    
    csv_files = glob.glob(search_pattern)

    if not csv_files:
        print(f"No files found matching pattern '{search_pattern}'. Please check the directory.")
        return

    print(f"Found {len(csv_files)} files matching {file_pattern} to aggregate:")
    for f_path in sorted(csv_files): # Sort for consistent order if it matters
        print(f"  - {os.path.basename(f_path)}")

    all_dataframes = []
    for f_path in sorted(csv_files): # Process in sorted order
        try:
            df = pd.read_csv(f_path)
            if df.empty:
                print(f"Warning: {os.path.basename(f_path)} is empty and will be skipped.")
                continue
            # Optional: Add a column to indicate the source file, can be useful for traceability
            # df['source_pipeline_step'] = os.path.basename(f_path).replace('_aug.csv', '') 
            all_dataframes.append(df)
            print(f"Successfully read {os.path.basename(f_path)} ({len(df)} rows, {len(df.columns)} columns)")
        except pd.errors.EmptyDataError: # Should be caught by df.empty now, but good to have
            print(f"Warning: {os.path.basename(f_path)} is empty (pd.errors.EmptyDataError) and will be skipped.")
        except Exception as e:
            print(f"Error reading {os.path.basename(f_path)}: {e}. Skipping this file.")

    if not all_dataframes:
        print("No dataframes were successfully read or all found files were empty. Output file will not be created.")
        return

    # Concatenate all dataframes
    # ignore_index=True will create a new clean index (0, 1, 2...)
    # sort=False is generally a good default when you expect columns to be mostly the same,
    # preventing pandas from alphabetically sorting columns if there are discrepancies.
    try:
        aggregated_df = pd.concat(all_dataframes, ignore_index=True, sort=False)
    except Exception as e:
        print(f"Error during concatenation: {e}")
        return

    print(f"\nAggregation complete.")
    print(f"Total rows in aggregated CSV: {len(aggregated_df)}")
    print(f"Total columns in aggregated CSV: {len(aggregated_df.columns)}")
    print(f"Columns are: {aggregated_df.columns.tolist()}")

    try:
        aggregated_df.to_csv(output_csv_path, index=False)
        print(f"\nSuccessfully aggregated CSVs into: {output_csv_path} 🎉")
    except Exception as e:
        print(f"Error writing aggregated CSV to '{output_csv_path}': {e}")

def main():
    """
    Main function to parse arguments and call the aggregation function.
    """
    parser = argparse.ArgumentParser(
        description="Aggregate multiple CSV files ending with '_aug.csv' from a specified folder into a single CSV file.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "input_folder",
        help="Path to the folder containing the '*_aug.csv' files."
    )
    parser.add_argument(
        "output_file",
        help="Path for the new aggregated CSV file (e.g., aggregated_pipeline_data.csv)."
    )

    args = parser.parse_args()

    aggregate_augmented_csvs(args.input_folder, args.output_file)

if __name__ == "__main__":
    main()