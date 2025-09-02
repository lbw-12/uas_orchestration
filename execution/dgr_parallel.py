import argparse
import os
import time
import shutil
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import traceback
from dgr import georeferencing, get_crs_zone, get_location, collect_info_specific_path

def process_single_file(args):
    file_name, df, location, crs, input_dir, output_dir, sensor, elevation_map_path, geoid_height = args
    try:
        # Read the CSV file and get the row for this file
        #df = pd.read_csv(df_path)
        # georeferencing expects the DataFrame, but we only need the row for this file
        georeferencing(
            file_name,
            df,
            location,
            crs,
            input_dir,
            output_dir,
            sensor,
            "silent",
            elevation_map_path,
            geoid_height
        )
        return (file_name, True, None)
    except Exception as e:
        return (file_name, False, traceback.format_exc())

def process_files_parallel(input_folder, output_folder, elevation_folder, geoid_height, num_workers=1):
    print(f'input_folder: {input_folder}')
    print(f'output_folder: {output_folder}')
    print(f'num_workers: {num_workers}')
    args_to_pass = collect_info_specific_path(input_folder, output_folder, elevation_folder)
    if not args_to_pass:
        print("Failed to collect necessary information")
        return

    df_csv, input_dir, output_dir, sensor, elevation_map_path = args_to_pass
    df = df_csv
    crs = get_crs_zone(df)
    location = get_location(input_dir)

    jpg_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith('.jpg') or f.lower().endswith('.tif')])
    total_files = len(jpg_files)
    print(f"Found {total_files} images to process")

    if os.path.exists(output_dir):
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                if not file.startswith('._') and not file.startswith("output.txt") and (file.endswith('.tif') or file.lower().endswith('.jpg')):
                    os.remove(os.path.join(root, file))
            for dir in dirs:
                shutil.rmtree(os.path.join(root, dir))
    else:
        os.makedirs(output_dir)

    start_time = time.time()
    processed_count = 0
    error_count = 0

    # Prepare argument tuples for each file
    arg_tuples = [
        (file_name, df, location, crs, input_dir, output_dir, sensor, elevation_map_path, geoid_height)
        for file_name in jpg_files
    ]

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = executor.map(process_single_file, arg_tuples)
        for file_name, success, tb in results:
            if success:
                processed_count += 1
                print(f"Successfully processed {file_name} ({processed_count}/{total_files})")
            else:
                error_count += 1
                print(f"Error processing {file_name}:")
                print(tb)

    end_time = time.time()
    print(f"\nProcessing Summary:")
    print(f"Total files found: {total_files}")
    print(f"Successfully processed: {processed_count}")
    print(f"Failed to process: {error_count}")
    print(f"Total time consumed: {end_time - start_time:.2f} seconds")

    output_files = [f for f in os.listdir(output_dir) if f.lower().endswith('.tif')]
    print(f"\nVerification:")
    print(f"Expected output files: {processed_count}")
    print(f"Actual output files: {len(output_files)}")

def main():
    parser = argparse.ArgumentParser(description='Parallel DGR image processing using ProcessPoolExecutor.')
    parser.add_argument('--input_folder', required=True, help='Input folder containing images')
    parser.add_argument('--output_folder', required=True, help='Output folder for processed images')
    parser.add_argument('--elevation_folder', required=True, help='Folder for elevation maps')
    parser.add_argument('--geoid_height', required=True, type=float, help='Geoid height')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of worker processes')
    args = parser.parse_args()

    print(f'input_folder: {args.input_folder}')
    print(f'output_folder: {args.output_folder}')
    print(f'elevation_folder: {args.elevation_folder}')
    print(f'geoid_height: {args.geoid_height}')
    print(f'num_workers: {args.num_workers}')

    process_files_parallel(args.input_folder, args.output_folder, args.elevation_folder, args.geoid_height, args.num_workers)

if __name__ == "__main__":
    main() 