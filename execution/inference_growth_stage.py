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
import pdb
from timm.data import resolve_data_config
from torchvision import transforms
from torch.utils.data import DataLoader
from timm.data.transforms_factory import create_transform
from tqdm import tqdm
import json
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from models.growth_stage_prediction.growth_stage_prediction_model import GrowthStagePredictionModel
from models.growth_stage_prediction.trainer import Trainer
from models.growth_stage_prediction.utils import set_seed, check_data_leakage
import argparse
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        #pdb.set_trace()
        image_path = self.image_folder[idx]
        full_image_path = os.path.join(self.image_folder_path, image_path)
        image = Image.open(full_image_path).convert("RGB")
        image = self.transform(image)
        return image, image_path



def run(input_images, model_path, config, output_json):


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

    results_dict = {
        path: int(pred)
        for path, pred in zip(results["image_names"], results["rounded_preds"])
    }

    print(f"Combined results: {len(results_dict)} entries")
    #save as json 
    
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(results_dict, f, indent=4)


def main():
    #get args
    
    parser = argparse.ArgumentParser(description="Run Growth Stage Prediction Inference")
    parser.add_argument('--input_dir', type=str, default='patches/', help='Path to the config file')
    parser.add_argument('--output_dir', type=str, default='results/growth_stage_predictions.json', help='Path to save the results')
    parser.add_argument('--model_path', type=str, default='models/growth_stage_model.pth', help='Path to the trained model')
    parser.add_argument('--config', type=str, default='growth_stage_configs/config.yaml', help='Path to the config file')
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

    #run(input_images=args.input_dir, output_path=args.output_dir, model_path=args.model_path, config=config, output_json=args.output_dir + '/inference_growth_stage_' + year + '.json')
    run(input_images=args.input_dir, model_path=args.model_path, config=config, output_json=args.output_dir)

if __name__ == "__main__":
    main()