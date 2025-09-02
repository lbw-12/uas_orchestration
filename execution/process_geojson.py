import json
import random
import sys
import os
import geopandas as gpd
import pandas as pd
import pdb
import argparse
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

import orchestrate_funcs as om_funcs


def add_attributes(gdf, output_filepath=None, farm_field=None, date=None, gs_json=None, canopy_json=None, ndvi_json=None):

    gs_dict = {
        'VE':  0, 'VC': 0, 'V1': 1, 'V2': 2, 'V3': 3, 'V4': 4, 'V5': 5, 'V6': 6, 'V7': 7, 'V8': 8, 'V9': 9,
        'V10': 10, 'V11': 10, 'V12': 10,
        'V13': 11, 'V14': 11, 'V15': 11, 'V16': 11, 'V17': 11, 'V18': 11, 'VT': 11,
        'R1': 12, 'R2': 13, 'R3': 14, 'R4': 15, 'R5': 16, 'R6': 17, 'R7': 18, 'R8': 19
    }

    gdf = gdf.copy()
    print(f'gdf crs is : {gdf.crs}')

    gdf = gdf.to_crs("EPSG:4326")

    print(f'gdf crs is : {gdf.crs}')
    print(f"gdf, columns: {gdf.columns}")
    plot = [col for col in gdf.columns if 'plot' in col.lower()]
    #iterate through the GeoDataFrame
    for index, row in gdf.iterrows():
        rgb_file_name = farm_field + '_om_' + row['crop'] + '_rgb_' + str(row[plot[0]]) + '_' + str(date)
        ms_file_name = farm_field + '_om_' + row['crop'] + '_multispectral_' + str(row[plot[0]]) + '_' + str(date) + '.tif'

        if rgb_file_name not in gs_json:
            print(f"Warning: {rgb_file_name} not found in gs_json. Skipping growth stage assignment for this row.")
            gdf.at[index, 'growth_stage_numeric'] = None
            gdf.at[index, 'original_growth_stage'] = None
        else:
            gdf.at[index, 'growth_stage_numeric'] = gs_json[rgb_file_name]
            gdf.at[index, 'original_growth_stage'] = next((k for k, v in gs_dict.items() if v == gdf.at[index, 'growth_stage_numeric']), None)
        
        if rgb_file_name not in canopy_json:
            print(f"Warning: {rgb_file_name} not found in canopy_json. Skipping canopy cover assignment for this row.")
            gdf.at[index, 'canopy_cover'] = None
        else:
            gdf.at[index, 'canopy_cover'] = canopy_json[rgb_file_name + '.tif']

        if ms_file_name not in ndvi_json:
            print(f"Warning: {ms_file_name} not found in ndvi_json. Skipping NDVI assignment for this row.")
            gdf.at[index, 'ndvi'] = None
        else:
            gdf.at[index, 'ndvi'] = ndvi_json[ms_file_name]

        #print(f"Row {index} original_growth_stage: {gdf.at[index, 'original_growth_stage']}, growth_stage_numeric: {gdf.at[index, 'growth_stage_numeric']}, canopy_cover: {gdf.at[index, 'canopy_cover']}, ndvi: {gdf.at[index, 'ndvi']}")
    if output_filepath:
        try:
            # 1. Convert the GeoDataFrame to a GeoJSON string
            geojson_str = gdf.to_json(indent=4)

            """# 2. Parse the GeoJSON string into a Python object (dictionary/list)
            geojson_obj = json.loads(geojson_str)"""

            # Write to file
            with open(output_filepath, 'w', encoding='utf-8') as f:
                f.write(geojson_str)

            """# 3. Write the Python object to a file with indentation
            with open(output_filepath, 'w', encoding='utf-8') as f:
                json.dump(geojson_obj, f, indent=4) # Use indent=4 for 4 spaces"""

            print(f"\nSuccessfully processed and added attributes to {len(gdf)} features.")
            print(f"Readable modified GeoJSON saved to: {output_filepath}")
        except Exception as e:
            print(f"Error: Could not write to output file {output_filepath}")
            print(f"Error details: {str(e)}")
 
def load_json(json_file):

    try:
        with open(json_file, 'r') as f:
            # The json.load() function reads the file and deserializes the JSON
            # content directly into a Python dictionary (or list if the root is an array).
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: The file '{json_file}' was not found.")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Could not decode JSON from '{json_file}'. Details: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

def main(inference_folder, location, date, output_geojson, year):    

    if inference_folder:
        # get list of json files in the inference_folder/
        json_files = [f for f in os.listdir(inference_folder) if f.endswith('.json') and location in f and date in f]
        # now group files based on "growth_stage", "canopy_cover", and "spectral_reflectance"
        gs_json, canopy_json, sr_json = [], [], []
        for json_file in json_files:
            if 'gs' in json_file:
                gs_json.append(os.path.join(inference_folder, json_file))
            elif 'cc' in json_file:
                canopy_json.append(os.path.join(inference_folder, json_file))
            elif 'sr' in json_file:
                sr_json.append(os.path.join(inference_folder, json_file))
        
        # Load the JSON files and combine into a single dictionary
        combined_gs, combined_canopy, combined_sr = {}, {}, {}
        if gs_json:
            for file in gs_json:
                gs_json_data = load_json(file)
                if gs_json_data:
                    combined_gs.update(gs_json_data)
        if canopy_json:
            for file in canopy_json:
                canopy_json_data = load_json(file)
                if canopy_json_data:
                    combined_canopy.update(canopy_json_data)
        if sr_json:
            for file in sr_json:
                ndvi_json = load_json(file)
                if ndvi_json:
                    combined_sr.update(ndvi_json)
                
    
    gs_dict = {
        'VE':  0, 'VC': 0, 'V1': 1, 'V2': 2, 'V3': 3, 'V4': 4, 'V5': 5, 'V6': 6, 'V7': 7, 'V8': 8, 'V9': 9,
        'V10': 10, 'V11': 10, 'V12': 10,
        'V13': 11, 'V14': 11, 'V15': 11, 'V16': 11, 'V17': 11, 'V18': 11, 'VT': 11,
        'R1': 12, 'R2': 13, 'R3': 14, 'R4': 15, 'R5': 16, 'R6': 17, 'R7': 18, 'R8': 19
    }

    # List of possible alphanumeric growth stages to pick from
    possible_alpha_growth_stages = list(gs_dict.keys())
    crops = ['corn','soy', 'cc', 'sc']



    #Load config file
    config = om_funcs.load_config(os.path.join('/fs/ess/PAS2699/nitrogen/data/uas/',year,'config/uas_config.yaml'))
    base_folder = config['base_folder']
    maptiles_folder = config['maptiles_folder']
    plot_shapefiles = config['plot_shapefiles']

    print(f'base folder: {base_folder}')

    plots = {}
    plots_combined = None
    for crop in crops:
        if crop in plot_shapefiles[location]:
            plots_path = os.path.join(base_folder,plot_shapefiles[location][crop])
            if os.path.exists(plots_path):
                plots[crop] = gpd.read_file(plots_path)
                print(f"{crop} shapefile is in {plots[crop].crs}")
                plots[crop]['crop'] = crop
    # Combine the GeoDataFrames'
    if not plots:
        plots_combined = None
    elif len(plots) == 1:
            plots_combined = plots[list(plots.keys())[0]]
    else:
            plots_combined = gpd.GeoDataFrame(pd.concat([plots[crop] for crop in plots.keys()], ignore_index=True))

    print(f'date: {date}')
    path_geojson = os.path.join(maptiles_folder,location,date, f'{location}_{date}.geojson') 
    print(f'path_geojson: {output_geojson}')
    if plots_combined is not None and not plots_combined.empty:
        add_attributes(plots_combined, path_geojson, location, date, 
                        gs_json=combined_gs, canopy_json=combined_canopy, ndvi_json=combined_sr)


if __name__ == "__main__":
    #arg parser
    
    parser = argparse.ArgumentParser(description="Process GeoJSON files with growth stage attributes.")
    parser.add_argument('--inference_folder', type=str, default='',
                        help='Path to the JSON file containing inference data.')
    parser.add_argument('--location', type=str, default='',
                        help='Location of the farm field.')
    parser.add_argument('--date', type=str, default='',
                        help='Date of the farm field.')
    parser.add_argument('--output_json', type=str, default='/fs/ess/PAS2699/nitrogen/data/uas/published/osu_public/',
                        help='Path to the output folder.')
    args = parser.parse_args()                    

    year = args.date[:4]
    main(args.inference_folder, args.location, args.date, args.output_json, year)
    