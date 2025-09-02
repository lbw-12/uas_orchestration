import os
import json
import argparse
import xml.etree.ElementTree as ET
from pyproj import Transformer

def find_extent(xml_file):
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"Error parsing XML file: {e}")
        return None
    except FileNotFoundError:
        print(f"Error: File not found at {xml_file}")
        return None
    
    bounding_box = root.find('BoundingBox')
    if bounding_box is not None:
        minx = bounding_box.get('minx')
        miny = bounding_box.get('miny')
        maxx = bounding_box.get('maxx')
        maxy = bounding_box.get('maxy')
    else:
        return None

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

    # Transform the bottom-left corner
    proj_min_x, proj_min_y = transformer.transform(minx, miny)

    # Transform the top-right corner
    proj_max_x, proj_max_y = transformer.transform(maxx, maxy)

    # The OpenLayers extent array is [minx, miny, maxx, maxy]
    extent = [proj_min_x, proj_min_y, proj_max_x, proj_max_y]

    return extent

def main(scan_dir):
    folders_dict = {}

    for year in sorted(os.listdir(scan_dir)):
        if not year.startswith('20') or not year.isdigit() or len(year) != 4:
            continue

        folders_dict[year] = {}
        #print(f'folder: {year}')
        for farm_field in sorted(os.listdir(os.path.join(scan_dir, year))):
            #print(f'farm_field: {farm_field}')
            farm_field_path = os.path.join(scan_dir, year, farm_field)
            all_entries = os.listdir(farm_field_path)
            subdirectories = [entry for entry in all_entries if os.path.isdir(os.path.join(farm_field_path, entry))]

            if len(subdirectories) > 0:
                folders_dict[year][farm_field] = {}
            for date in sorted(os.listdir(os.path.join(scan_dir, year, farm_field))):
                tilemapresource_file = os.path.join(scan_dir, year, farm_field, date, "tilemapresource.xml")
                extent = find_extent(tilemapresource_file)

                #print(f'      date: {date}')
                folders_dict[year][farm_field][date] = {}
                folders_dict[year][farm_field][date]['extent'] = extent

                date_folder = os.path.join(scan_dir, year, farm_field, date)
                geojson_files = [f for f in os.listdir(date_folder) if f.endswith('.geojson')]
                #print(f'length of geojson_files: {geojson_files}')
                if len(geojson_files) == 0:
                    #print(f'Warning, No geojson file in this location: {date_folder}')
                    folders_dict[year][farm_field][date]['geojson'] = {}
                elif len(geojson_files) > 1:
                    #print(f'Warning, multiple geojson files in this location: {date_folder}')
                    folders_dict[year][farm_field][date]['geojson'] = {}
                if len(geojson_files) == 1:
                    folders_dict[year][farm_field][date]['geojson'] = os.path.join(date_folder, geojson_files[0])
                    #print(f"geojson file: {folders_dict[year][farm_field][date]['geojson']}")

    json_file = os.path.join(scan_dir, 'folders.json')
    with open(json_file, 'w') as f:
        json.dump(folders_dict, f, indent=4)
    print(f"folders.json saved to {json_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scan a directory and create folders.json summary.")
    parser.add_argument('--scan_dir', required=True, help='Directory to scan')
    args = parser.parse_args()
    main(args.scan_dir) 