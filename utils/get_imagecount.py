import os
import argparse
import sys

def count_images_in_folder(path):
    """
    Counts the number of .tif, .tiff, .jpg, or .jpeg image files in a given folder.

    Args:
        folder_path (str): The path to the folder to scan.
    """
    if not os.path.isdir(path):
        print(f"Error: Folder not found at '{path}'")
        return

    image_extensions = ('.tif', '.tiff', '.jpg', '.jpeg')
    image_count = 0
    
    for entry_name in os.listdir(path):
        # Check if the entry name ends with any of the image extensions (case-insensitive)
        if entry_name.lower().endswith(image_extensions):
            # Optionally, confirm it's a file (os.listdir can also return directories)
            full_path = os.path.join(path, entry_name)
            if os.path.isfile(full_path):
                image_count += 1
                # print(f"Found image: {entry_name}") # Uncomment to list found images

    return image_count



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Count .tif/.tiff and .jpg/.jpeg images in a specified folder."
    )
    parser.add_argument(
        "folder_path",
        type=str,
        help="The path to the folder to scan for images."
    )

    args = parser.parse_args()

    path = args.folder_path

    image_count = count_images_in_folder(path)

    if image_count is not None:
        print(f'{image_count}')