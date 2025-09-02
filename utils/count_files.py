import os
import sys

filepath = '/fs/ess/PAS2699/nitrogen/data/uas/2025/plot_patches/'
#filepath = '/fs/ess/PAS2699/nitrogen/data/uas/2025/plottiles/plot_tiles_rgb_ir/'

print_dirwithfiles = True



for root, dirs, files in sorted(os.walk(filepath)):
    if files and print_dirwithfiles:
        print(f"--- Directory: {root} ---Found {len(files)} files.")
        #print(f"Found {len(files)} files.")
        # Print each filename
        #for filename in files:
        #    print(f"  - {filename}")
        
        # Optional: uncomment the line below to stop after the first directory
        # break 
    elif not files:
        print(f"No files in {root}")