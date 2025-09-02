#!/bin/bash

# Specify the source and destination directories
source_dir="/fs/ess/PAS2699/nitrogen/data/uas/2025/flights/20250603-Douglass/20250603_Douglass_Jeremiah_Altum233 Flight 01/01_Images/20250603_Douglass_Jeremiah_Altum233 Flight 01/OUTPUT"
destination_dir="/fs/ess/PAS2699/nitrogen/data/uas/2025/flights/20250603-Douglass/20250603_Douglass_Lincoln_Altum233 Flight 02/01_Images/20250603_Douglass_Lincoln_Altum233 Flight 02/OUTPUT"

# Create the destination directory if it doesn't exist
mkdir -p "$destination_dir"

# Loop through all files starting with "{number}_" in the source directory
for file in "$source_dir"/3_*; do
  # Check if any files match the pattern
  if [ -f "$file" ]; then
    # Get the base name of the file
    base_name=$(basename "$file")
    # Move and rename the file to the destination directory
    mv "$file" "$destination_dir/${base_name#3_}"
  fi
done

echo "Files moved and renamed successfully! 🎉"