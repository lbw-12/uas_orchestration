import os
import argparse
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from ir import process_image

def process_single_file(args):
    target_image_path, reference_image_path, output_dir = args
    try:
        result = process_image(target_image_path, reference_image_path, output_dir)
        return (os.path.basename(target_image_path), result, None)
    except Exception as e:
        import traceback
        return (os.path.basename(target_image_path), False, traceback.format_exc())

def main():
    parser = argparse.ArgumentParser(description="Parallel IR image registration using ProcessPoolExecutor.")
    parser.add_argument("input_dir", type=str, help="Directory containing target images")
    parser.add_argument("output_dir", type=str, help="Directory to save processed images")
    parser.add_argument("orthomosaic_image_path", type=str, help="Path to the orthomosaic image")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of worker processes")
    args = parser.parse_args()

    if os.path.exists(args.output_dir):
        for root, dirs, files in os.walk(args.output_dir):
            for file in files:
                if not file.startswith('._'):
                    os.remove(os.path.join(root, file))
            for dir in dirs:
                shutil.rmtree(os.path.join(root, dir))
    else:
        os.makedirs(args.output_dir)

    tif_files = sorted([f for f in os.listdir(args.input_dir) if f.endswith('.tif') and not f.startswith('._')])
    total_files = len(tif_files)
    print(f"Found {total_files} images to process")

    arg_tuples = [
        (os.path.join(args.input_dir, file), args.orthomosaic_image_path, args.output_dir)
        for file in tif_files
    ]

    start_time = time.time()
    processed_count = 0
    error_count = 0

    print(f"Using {min(args.num_workers, 20)} workers")

    with ProcessPoolExecutor(max_workers=min(args.num_workers, 20)) as executor:
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

if __name__ == "__main__":
    main() 