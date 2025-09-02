import json
import random
import yaml
import sys
import os
import geopandas as gpd
import pandas as pd
import pdb
import argparse
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

import orchestrate_funcs as om_funcs
import re


def load_phenotypes(gdf, config, field, date, year, gs_json, cc_json, sr_json, pub_folder):
    """
    Merges attributes from JSON files into GeoDataFrames and saves the result as a GeoJSON file.

    For each crop in the input gdf dictionary, this function maps growth stage,
    canopy cover, and NDVI data from the corresponding JSON objects onto the
    geodataframe based on a 'plot' identifier. It then combines all crop
    dataframes and saves a single GeoJSON file for the given field and date.

    Args:
        gdf (dict): A dictionary of GeoDataFrames, where keys are crop names (str)
                    and values are GeoDataFrames containing plot geometries.
                    Each GeoDataFrame must have a 'plot' column.
        config (dict): A configuration dictionary. Expected to contain 'geojson_folder'
                       for the output directory.
        field (str): The field identifier.
        date (str): The date of the data observation.
        year (int or str): The year of the data observation.
        gs_json (dict or None): A nested dictionary with growth stage data.
                                Structure: [field][crop][source][date][plot] -> {'numeric': X, 'stage': Y}
        cc_json (dict or None): A nested dictionary with canopy cover data.
                                Structure: [field][crop][source][date][plot] -> value
        sr_json (dict or None): A nested dictionary with NDVI data.
                                Structure: [field][crop][source][date][plot] -> value
    """
    processed_gdfs = []

    # Iterate over each crop type (e.g., 'corn', 'soybean') from the input
    for crop in gdf.keys():
        print(f"Processing attributes for crop: {crop}")
        # Work on a copy to avoid modifying the original DataFrame in the parent scope
        target_gdf = gdf[crop].copy()
        target_gdf['plot'] = target_gdf['plot'].astype(str)

        # --- Helper function to safely extract data from the nested JSON objects ---
        def extract_data_map(json_obj, json_name,data_key=None):
            """
            Extracts plot-level data from a JSON object into a dictionary for mapping.
            It gracefully handles missing keys at any level of the nesting.
            """
            if not json_obj:
                return {}  # Return an empty map if the JSON object is None

            # Safely navigate the nested dictionary structure using .get() to avoid errors
            field_data = json_obj.get(field)
            if not field_data:
                print(f"Warning: In '{json_name}', could not find top-level key for field: '{field}'")
                return {}

            crop_data = field_data.get(crop)
            if not crop_data:
                print(f"Warning: In '{json_name}', could not find data for crop: '{crop}' (under field: '{field}')")
                return {}

            sources = list(crop_data.keys())
            if not sources:
                print(f"Warning: In '{json_name}', no 'source' key found for path: '{field}/{crop}'")
                return {}
            
            source = sources[0] # Assumption: the first source is the desired one.
            source_data = crop_data.get(source)

            data_for_date = source_data.get(date)
            if not data_for_date:
                print(f"Warning: In '{json_name}', could not find data for date: '{date}' (under path: '{field}/{crop}/{source}')")
                return {}

            # If we get here, data_for_date is a dictionary of plots.
            try:
                if data_key:
                    # For growth stage, which has nested 'stage' and 'numeric' values
                    return {str(plot): plot_data.get(data_key) for plot, plot_data in data_for_date.items()}
                else:
                    # For canopy cover and NDVI, where the value is direct
                    return {str(plot): value for plot, value in data_for_date.items()}

            except (AttributeError, TypeError):
                # This might happen if data_for_date.items() fails, e.g. it's not a dict.
                print(f"Warning: In '{json_name}', data for path '{field}/{crop}/{source}/{date}' is not in the expected format (expected a dictionary of plots).")
                return {}

        # --- Extract data for each attribute and map it to the GeoDataFrame ---



        # 1. Growth Stage ('stage' and 'numeric')
        stage_map = extract_data_map(gs_json, 'gs_json', 'stage')
        numeric_map = extract_data_map(gs_json, 'gs_json', 'numeric')

        target_gdf['original_growth_stage'] = target_gdf['plot'].map(stage_map)
        target_gdf['growth_stage_numeric'] = target_gdf['plot'].map(numeric_map)

        # 2. Canopy Cover ('cc')
        cc_map = extract_data_map(cc_json, 'cc_json')
        target_gdf['canopy_cover'] = target_gdf['plot'].map(cc_map)

        # 3. Spectral Reflectance / NDVI ('ndvi')
        ndvi_map = extract_data_map(sr_json, 'sr_json')
        target_gdf['ndvi'] = target_gdf['plot'].map(ndvi_map)

        # --- Add metadata columns for context ---
        target_gdf['field'] = field
        target_gdf['date'] = date
        target_gdf['year'] = year
        target_gdf['crop'] = crop

        processed_gdfs.append(target_gdf)

        # Inside load_phenotypes(), after creating the maps
        print(f"--- Debugging {field}/{crop}/{date} ---")
        if not target_gdf.empty:
            print(f"Shapefile 'plot' type: {type(target_gdf['plot'].iloc[0])}")
            print(f"Sample shapefile plot IDs: {target_gdf['plot'].head(3).tolist()}")

        if cc_map: # Check one of the maps
            sample_key = next(iter(cc_map.keys()))
            print(f"JSON map key type: {type(sample_key)}")
            print(f"Sample JSON map keys: {list(cc_map.keys())[:3]}")


    # --- Combine all processed GeoDataFrames and save to a single file ---
    if not processed_gdfs:
        print(f"No data was processed for {field} on {date}. Skipping GeoJSON creation.")
        return

    # Concatenate all individual crop GeoDataFrames (e.g., corn, soy) into one
    final_gdf = pd.concat(processed_gdfs, ignore_index=True)

    # Define the output path for the GeoJSON file
    output_filename = f"{field}_{date}.geojson"
    output_folder = os.path.join(pub_folder, field, date)
    output_path = os.path.join(output_folder, output_filename)

    # Save the final, merged GeoDataFrame to a GeoJSON file.
    # The .to_file method handles NaN values by converting them to 'null' in the output.
    if os.path.exists(output_folder):
        geojson_str = final_gdf.to_json(indent=4)
        with open(output_path, 'w') as f:
            f.write(geojson_str)
        print(f'generated {output_path}')
    else:
        print(f'{output_folder} does not exist')
    print(f"Successfully created GeoJSON: {output_path}")
        

def generate_geojson(config, inference_folder, year):

    # Get all publishing folders
    pub_folder_dict = config['publishing_folder']
    
    pub_folders = []
    for field, folder in pub_folder_dict.items():
        if folder not in pub_folders:
            pub_folders.append(folder)

    shapefiles_dict = config['plot_shapefiles']

    for pub_folder in pub_folders:
        try:
            json_file = Path(pub_folder).parent / 'folders.json'
            with open(json_file, 'r') as f:
                folders = json.load(f)
        except FileNotFoundError:
            print(f'{json_file} not found')
            continue
        
        for field in folders[year]:
            for date in folders[year][field]: 
                print(f'Processing {field} {date} {year}')
                gdf = {}
                if field in shapefiles_dict:
                    for crop in shapefiles_dict[field]:
                        # Load the shapefile
                        shapefile_path = os.path.join(config['base_folder'], shapefiles_dict[field][crop])
                        print(f'shapefile_path: {shapefile_path}')
                        if os.path.exists(shapefile_path):
                            gdf[crop] = gpd.read_file(shapefile_path)
                            print(f'length of shapefile: {len(gdf[crop])}')
                            print(f"Processing {field} {crop} {year}")
                        else:
                            print(f'shapefile_path does not exist: {shapefile_path}')
                    # Get the json files for the date
                    json_files = [f for f in os.listdir(inference_folder) if f.endswith('.json') and field in f and date in f]
                    print(f'json_files: {json_files}')
                    # now group files based on "growth_stage", "canopy_cover", and "spectral_reflectance"
                    gs_json, cc_json, sr_json = None, None, None
                    for json_file in json_files:
                        if 'inf_gs' in json_file and not gs_json:
                            gs_json = json.load(open(os.path.join(inference_folder, json_file)))
                        elif 'inf_cc' in json_file and not cc_json:
                            cc_json = json.load(open(os.path.join(inference_folder, json_file)))
                        elif 'inf_sr' in json_file and not sr_json:
                            sr_json = json.load(open(os.path.join(inference_folder, json_file)))
                        elif 'inf_gs' in json_file and gs_json:
                            print(f'{json_file} cannot be added because of a duplicate')
                        elif 'inf_cc' in json_file and cc_json:
                            print(f'{json_file} cannot be added because of a duplicate')
                        elif 'inf_sr' in json_file and sr_json:
                            print(f'{json_file} cannot be added because of a duplicate')
                        else:
                            print(f'{json_file} cannot be added because it is not a valid json file')
                    # Add attributes to the shapefile
                    load_phenotypes(gdf, config, field, date, year, gs_json, cc_json, sr_json, pub_folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process GeoJSON files with growth stage attributes.")
    parser.add_argument('--config_file', type=str, default='',
                        help='Path to the config files.')
    args = parser.parse_args()                    

    with open(args.config_file, 'r') as f:
        config = yaml.safe_load(f)

    # Get the year from the parent directory of the config file

    config_parent_dir = Path(args.config_file).parents[1].name

    match = re.search(r'\d{4}', config_parent_dir)
    if match:
        year = match.group(0)
    else:
        print(f'year not found in {config_parent_dir}')
        exit()

    print(f'year: {year}')
    inference_folder = os.path.join(config['base_folder'], config['uas_pipeline']['step9']['output_folder'])
    print(f'inference_folder: {inference_folder}')

    generate_geojson(config, inference_folder, year)    