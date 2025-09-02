# main.py
import os
import yaml
import random
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import pandas as pd
from collections import defaultdict
import gc
from torch.utils.data import Dataset
from timm.data import resolve_data_config
from torchvision import transforms
from torch.utils.data import DataLoader
from timm.data.transforms_factory import create_transform
from tqdm import tqdm
import json
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from models.growth_stage_prediction.growth_stage_prediction_model import GrowthStagePredictionModel
from models.growth_stage_prediction.trainer import Trainer
from models.growth_stage_prediction.utils import set_seed, check_data_leakage
import argparse
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from collections import defaultdict
import pdb

loc_list = ['northwest', 'wooster', 'western']
crop_list = ['corn', 'soy']
year_list = ['2023', '2024', '2025']

def load_config(config_path: str):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

class GrowthStageDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder_path = image_folder
        self.image_folder = os.listdir(image_folder)
        self.transform = transform if transform else transforms.Compose([
            transforms.ToTensor()
        ])
        print(f"transform: {self.transform}")

    def __len__(self):
        return len(self.image_folder)

    def __getitem__(self, idx):
        image_path = self.image_folder[idx]
        full_image_path = os.path.join(self.image_folder_path, image_path)
        image = Image.open(full_image_path).convert("RGB")
        image = self.transform(image)
        return image, image_path



def run(input_images, model_path, config, output_json, field, plot_image_source, date):

    model = GrowthStagePredictionModel(config['model']['name'], config['model']['num_classes'])
    configuration = resolve_data_config({}, model=model.backbone)
    transform = create_transform(**configuration)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])
    criterion = torch.nn.MSELoss()
    
    test_dataset = GrowthStageDataset(input_images, transform)

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        pin_memory=True
    )
    trainer = Trainer(model, None, None, test_loader, criterion, optimizer, device, config, None)
    print(f"Test model: {model_path}")
    results = trainer.inference(model_path)
    gs_dict = {}

    gs_dict['corn'] = {
    'VE':  0, 'V1': 1, 'V2': 2, 'V3': 3, 'V4': 4, 'V5': 5, 'V6': 6, 'V7': 7, 'V8': 8, 'V9': 9,
    'V10': 10, 'V11': 10, 'V12': 10,
    'V13': 11, 'V14': 11, 'V15': 11, 'V16': 11, 'V17': 11, 'V18': 11, 'VT': 11,
    'R1': 12, 'R2': 13, 'R3': 14, 'R4': 15, 'R5': 16, 'R6': 17}

    gs_dict['cc'] = {
    'VE':  0, 'V1': 1, 'V2': 2, 'V3': 3, 'V4': 4, 'V5': 5, 'V6': 6, 'V7': 7, 'V8': 8, 'V9': 9,
    'V10': 10, 'V11': 10, 'V12': 10,
    'V13': 11, 'V14': 11, 'V15': 11, 'V16': 11, 'V17': 11, 'V18': 11, 'VT': 11,
    'R1': 12, 'R2': 13, 'R3': 14, 'R4': 15, 'R5': 16, 'R6': 17}

    gs_dict['sc'] = {
    'VE':  0, 'V1': 1, 'V2': 2, 'V3': 3, 'V4': 4, 'V5': 5, 'V6': 6, 'V7': 7, 'V8': 8, 'V9': 9,
    'V10': 10, 'V11': 10, 'V12': 10,
    'V13': 11, 'V14': 11, 'V15': 11, 'V16': 11, 'V17': 11, 'V18': 11, 'VT': 11,
    'R1': 12, 'R2': 13, 'R3': 14, 'R4': 15, 'R5': 16, 'R6': 17}




    gs_dict['soy'] = {
    'VE':  0, 'VC': 0, 'V1': 1, 'V2': 2, 'V3': 3, 'V4': 4, 'V5': 5, 'V6': 6, 'V7': 7, 'V8': 8, 'V9': 9,
    'V10': 10, 'V11': 10, 'V12': 10,
    'V13': 11, 'V14': 11, 'V15': 11, 'V16': 11, 'V17': 11, 'V18': 11, 'VT': 11,
    'R1': 12, 'R2': 13, 'R3': 14, 'R4': 15, 'R5': 16, 'R6': 17, 'R7': 18, 'R8': 19}


    stage_name_maps = {} # This will hold a map for 'corn' and a map for 'soy'

    # Loop through the top-level gs_dict ('corn' and 'soy')
    for crop, crop_gs_dict in gs_dict.items():
        # Reverse the specific dictionary for the current crop
        reversed_gs_dict = defaultdict(list)
        for stage, num_val in crop_gs_dict.items():
            reversed_gs_dict[num_val].append(stage)

        # Create the clean map for the current crop
        current_crop_map = {}
        for num_val, stages in reversed_gs_dict.items():
            if len(stages) == 1:
                current_crop_map[num_val] = stages[0]
            else:
                current_crop_map[num_val] = f"{stages[0]}-{stages[-1]}"
        
        # Add the newly created map to our dictionary of maps
        stage_name_maps[crop] = current_crop_map

    
    predictions_by_crop = defaultdict(dict)
    print("Validating plot numbers and grouping by crop...")

    for image_name, pred in zip(results["image_names"], results["rounded_preds"]):
        # --- A. Extract Crop Name ---
        crop_name = None
        if '_corn_' in image_name:
            crop_name = 'corn'
        elif '_soy_' in image_name:
            crop_name = 'soy'
        elif '_cc_' in image_name:
            crop_name = 'cc'
        elif '_sc_' in image_name:
            crop_name = 'sc'

        # --- B. Extract and Validate Plot Number ---
        plot_str = image_name.split('_')[-2]

        # --- C. Add to the correct group if valid ---
        if crop_name and plot_str.isdigit() and len(plot_str) == 3:
            # Add the plot and its prediction to the correct crop's dictionary
            predictions_by_crop[crop_name][plot_str] = pred
        else:
            # If crop or plot format is invalid, print a warning.
            print(f"  [!] Warning: Skipping invalid format in filename '{image_name}'.")

    # 2. Sort the plots within each crop group and build the final structure.
    final_results = {field: {}} # Start with the top-level field key

    for crop, plots_dict in predictions_by_crop.items():
        print(f"Sorting {len(plots_dict)} plots for crop: '{crop}'")

        # Sort the plots for the current crop numerically
        sorted_plots = sorted(plots_dict.items(), key=lambda item: int(item[0]))
        ordered_plots_dict = dict(sorted_plots)

        # --- NEW: Part 2 - Create the final plot dictionary with stage info ---
        current_stage_map = stage_name_maps.get(crop, {})
        plots_with_stage_info = {}
        for plot_num, numeric_pred in ordered_plots_dict.items():
        # Look up the stage name string from our map. Default to 'Unknown'.
            stage_name = current_stage_map.get(numeric_pred, "Unknown")
        
            # Create the final object for this plot
            plots_with_stage_info[plot_num] = {
                "numeric": int(numeric_pred),
                "stage": stage_name
            }

        # Only add the crop to final results if it has valid plot data.
        if plots_with_stage_info:
            # Build the nested structure for this specific crop
            final_results[field][crop] = {
                plot_image_source: {
                    date: plots_with_stage_info
                }
            }
            print(f"Found and stored {len(plots_with_stage_info)} plots for '{crop}'.")
        else:
            print(f"Warning: No plots with results found for '{crop}'. Skipping.")


    # Check if the final results dictionary for the field contains any crop data.
    if final_results[field]:
        print(f"Structuring results for {len(final_results[field])} crop(s) with data.")
        # Save the final structured dictionary to JSON
        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        with open(output_json, 'w') as f:
            json.dump(final_results, f, indent=4)
        print(f"Combined results saved to: {output_json}")
    else:
        # If final_results[field] is empty, no crops had valid data.
        print("Error: No valid plot predictions found. JSON file will not be created.")


def main():
    #get args
    
    parser = argparse.ArgumentParser(description="Run Growth Stage Prediction Inference")
    parser.add_argument('--input_dir', type=str, default='patches/', help='Path to the config file')
    parser.add_argument('--output_dir', type=str, default='results/growth_stage_predictions.json', help='Path to save the results')
    parser.add_argument('--model_path', type=str, default='models/growth_stage_model.pth', help='Path to the trained model')
    parser.add_argument('--config', type=str, default='growth_stage_configs/config.yaml', help='Path to the config file')
    parser.add_argument('--field', type=str, default=None, help='Field ID')
    parser.add_argument('--plotimage_source', type=str, default=None, help='Plot image source')
    parser.add_argument('--date', type=str, default=None, help='Date')
    print("Parsing arguments...")
    args = parser.parse_args()

    year_list = ['2023', '2024', '2025']
    # get year from input_dir
    input_dir = args.input_dir
    year = None
    for y in year_list:
        if y in input_dir:
            year = y
            break
    config = load_config(args.config)

    field = args.field
    plot_image_source = args.plotimage_source
    date = args.date
    #run(input_images=args.input_dir, output_path=args.output_dir, model_path=args.model_path, config=config, output_json=args.output_dir + '/inference_growth_stage_' + year + '.json')
    run(input_images=args.input_dir, model_path=args.model_path, config=config, output_json=args.output_dir, field=field, plot_image_source=plot_image_source, date=date)

if __name__ == "__main__":
    main()