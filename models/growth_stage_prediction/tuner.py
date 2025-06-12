import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from growth_stage_prediction_model import GrowthStagePredictionModel
from trainer import Trainer
from utils import set_seed
from torchvision import transforms
from torch.utils.data import Dataset
import yaml
import os
import pandas as pd
import random
import numpy as np
from sklearn.model_selection import KFold
from PIL import Image
import pdb
from torch.utils.data.distributed import DistributedSampler
from typing import Dict


loc_list = ['northwest', 'wooster', 'western']
crop_list = ['corn', 'soy']
year_list = ['2023', '2024', '2025']

def get_planting_date(pd_filename):
    df = pd.read_csv(pd_filename, index_col=0)
    df_melted = df.reset_index().melt(id_vars="index", var_name="Location", value_name="Date")
    df_melted.rename(columns={"index": "PD"}, inplace=True)
    df_melted["Date"] = pd.to_datetime(df_melted["Date"])
    df_melted["Key"] = df_melted["PD"] + "_" + df_melted["Location"]
    df_sorted = df_melted.sort_values("Date")
    sorted_dict = dict(zip(df_sorted["Key"], df_sorted["Date"]))
    print(sorted_dict)
    all_keys = list(sorted_dict.keys())
    random.seed(42)
    random.shuffle(all_keys)

    test_size = 3
    test_keys = all_keys[:test_size]
    trainval_keys = all_keys[test_size:]  

    print(f"Test Set (held-out): {test_keys}")

    kf = KFold(n_splits=2, shuffle=True, random_state=42)

    folds = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(trainval_keys)):
        train_keys = [trainval_keys[i] for i in train_idx]
        val_keys = [trainval_keys[i] for i in val_idx]
        folds.append((train_keys, val_keys))
        print(f"\nFold {fold+1}:")
        print(f"  Train: {train_keys}")
        print(f"  Val:   {val_keys}")
        
    test_keys = [k.lower() for k in test_keys]
    folds = [([k.lower() for k in train], [k.lower() for k in val]) for train, val in folds]
    
    return test_keys, folds

def get_plot_location_splits(plot_pd_file, test_keys, folds, crop):
    df_plots = pd.read_csv(plot_pd_file, index_col=0)
    df_plots["location"] = df_plots["location"].str.lower()
    df_plots["pd"] = df_plots["pd"].str.lower()
    df_plots["key"] = df_plots["pd"] + "_" + df_plots["location"]
    def to_plotloc(df):
        return [f"{plot}_{loc}_{crop}" for plot, loc in zip(df.index, df["location"])]
    test_df = df_plots[df_plots["key"].isin(test_keys)]
    test_plot_locs = to_plotloc(test_df)

    cv_splits = {}
    for i, (train_keys, val_keys) in enumerate(folds):
        train_df = df_plots[df_plots["key"].isin(train_keys)]
        val_df = df_plots[df_plots["key"].isin(val_keys)]

        fold_data = {
            "train": to_plotloc(train_df),
            "val": to_plotloc(val_df)
        }
        cv_splits[f"fold_{i+1}"] = fold_data

    result = {
        "test_keys": test_keys,
        "test": test_plot_locs,
        "cv_splits": cv_splits
    }
    return result

def merge_crop_splits(corn_dict, soy_dict):
    merged = {
        "test_keys": sorted(set(corn_dict["test_keys"] + soy_dict["test_keys"])),
        "test": sorted(set(corn_dict["test"] + soy_dict["test"])),
        "cv_splits": {}
    }

    for fold in corn_dict["cv_splits"]:
        merged["cv_splits"][fold] = {
            "train": sorted(set(corn_dict["cv_splits"][fold]["train"] + soy_dict["cv_splits"][fold]["train"])),
            "val": sorted(set(corn_dict["cv_splits"][fold]["val"] + soy_dict["cv_splits"][fold]["val"]))
        }

    return merged

def parse_date(col, year):
    parts = col.split('.', maxsplit=1)
    base_col = parts[0]
    suffix = f".{parts[1]}" if len(parts) > 1 else ""

    try:
       
        full_date = f"{base_col}-{year}"
        parsed_date = pd.to_datetime(full_date, format="%d-%b-%Y").date()
        return f"{parsed_date}{suffix}"  
    except:
        try:
            parsed_date = pd.to_datetime(base_col).date()
            return f"{parsed_date}{suffix}"
        except:
            return col  


def interpolate_growth_stages_full_range(df, growth_stage_dict):
    df = df.replace({".": np.nan, "": np.nan})
    df = df.fillna(-1)
    df_numeric = df.applymap(lambda x: growth_stage_dict.get(x, np.nan))
    df_numeric.columns = pd.to_datetime(df_numeric.columns)
    full_range = pd.date_range(start=df_numeric.columns.min(), end=df_numeric.columns.max())
    interpolated = (
        df_numeric
        .transpose()
        .reindex(full_range)
        .interpolate(limit_direction="forward")
        .transpose()
    )
    interpolated.columns = interpolated.columns.strftime("%Y-%m-%d")

    return interpolated

def get_growth_stage(image_folder, growthstage_csvfolder, growth_stages):
    
    #assigning growth stages to the corresponding plots and location
    before_planting_count = 0
    growth_stage_dict = {}
    for growthstage_csv in growthstage_csvfolder:
        print(f"Processing CSV: {growthstage_csv}")
        lower_name = growthstage_csv.lower()
        loc = next((l for l in loc_list if l in lower_name), None)
        crop = next((c for c in crop_list if c in lower_name), None)
        year = next((y for y in year_list if y in growthstage_csv), None)
        df = pd.read_csv(growthstage_csv)
        df.columns = [parse_date(col, year=year) for col in df.columns]
        df.columns = df.columns.astype(str).str.strip()
        date_col = next((col for col in df.columns if col.lower() == "date"), None)
        df = df.set_index(date_col)
        print(df)
        keep_mask = df.iloc[0].astype(str).str.lower() == "stage"
        df_stage_only = df.loc[:, keep_mask]
        df_stage_only = df_stage_only.reset_index()
        df_stage_only = df_stage_only.drop(index=0)
        df_stage_only.columns = df_stage_only.columns.astype(str).str.strip()
        df_stage_only.columns.values[0] = 'plot_no'
        df_stage_only = df_stage_only.set_index("plot_no")
        df_stage_only.columns = df_stage_only.columns.astype(str)
        parsed_cols = pd.to_datetime(df_stage_only.columns, errors='coerce')
        df_stage_only = df_stage_only.loc[:, parsed_cols.notna()]
        print(f"Parsed columns: {df_stage_only.columns}")
        df_stage_only = interpolate_growth_stages_full_range(df_stage_only, growth_stages)
        df_stage_only = df_stage_only.reset_index()

        for patch in os.listdir(image_folder):
            
            if loc in patch.lower() and crop in patch.lower() and year in patch.lower():
                #print(f"Processing patch: {patch}")
                
                plot_no = patch.split('_')[-3]
                date = patch.split('_')[-2]
                date = pd.to_datetime(date, format="%Y%m%d").date()
                #stage_dates = [pd.to_datetime(col).date() for col in stage_cols if col != 'plot_no']
                #valid_dates = [d for d in stage_dates if d <= date]
                #closest_date = max(valid_dates)

                growth_stage = df_stage_only.loc[df_stage_only['plot_no'] == plot_no, str(date)].values[0]
                if pd.notna(growth_stage):
                    growth_stage_dict[patch]= growth_stage
                else:
                    before_planting_count += 1
                
    #pdb.set_trace()       
    return growth_stage_dict

def get_image_names(image_folder, plot_split_keys):
    image_names = []
    for plot_key in plot_split_keys:
        plot_key = plot_key.lower()
        files = os.listdir(image_folder)
        for file in files:
            if file.startswith(plot_key):
                image_names.append(os.path.join(image_folder, file))
    return image_names
# Same Dataset class
class GrowthStageDataset(Dataset):
    def __init__(self, image_folder, plot_split_keys, growthstage, transform=None):
        self.image_folder = image_folder
        self.plot_split_keys = plot_split_keys
        self.growthstage = growthstage
        self.transform = transform if transform else transforms.ToTensor()

    def __len__(self):
        return len(self.plot_split_keys)

    def __getitem__(self, idx):
        plot_key = self.plot_split_keys[idx]
        image_path = os.path.join(self.image_folder, plot_key)
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        growth_stage = self.growthstage[idx]
        return image, growth_stage

def load_config(config_path: str):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def objective(trial, global_data: Dict):
    config = global_data["config"]
    merged_split = global_data["merged_split"]
    patches_growth_stage = global_data["patches_growth_stage"]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Sample hyperparameters
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-3)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])

    # Initialize model
    model = GrowthStagePredictionModel(config['model']['name'], config['model']['num_classes']).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    # Prepare transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # For simplicity, tune on first fold only
    fold = list(merged_split['cv_splits'].keys())[0]
    split = merged_split['cv_splits'][fold]

    dataset = {"train_img": [], "val_img": [], "train_gs": [], "val_gs": []}
    train_keys = split['train']
    val_keys = split['val']

    for patch in os.listdir(config['dataset']['image_folder']):
        loc = next((l for l in ['northwest', 'wooster', 'western'] if l in patch), None)
        crop = next((c for c in ['corn', 'soy'] if c in patch), None)
        plot = patch.split('_')[-3]

        train_matches = [s for s in train_keys if all(sub in s for sub in [loc, crop, plot])]
        val_matches = [s for s in val_keys if all(sub in s for sub in [loc, crop, plot])]

        if train_matches and patch in patches_growth_stage:
            dataset['train_img'].append(patch)
            dataset['train_gs'].append(patches_growth_stage[patch])
        elif val_matches and patch in patches_growth_stage:
            dataset['val_img'].append(patch)
            dataset['val_gs'].append(patches_growth_stage[patch])

    # Subsample for tuning
    train_dataset = GrowthStageDataset(config['dataset']['image_folder'], dataset['train_img'][:10000], dataset['train_gs'][:10000], transform)
    val_dataset = GrowthStageDataset(config['dataset']['image_folder'], dataset['val_img'][:10000], dataset['val_gs'][:10000], transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    trainer = Trainer(model, train_loader, val_loader, None, criterion, optimizer, device, config)
    best_val_mae, _ = trainer.train(fold=0, num_epochs=5)

    return best_val_mae


def main():
    config = load_config("configs/config.yaml")
    set_seed(config['general']['seed'])

    print("[INFO] Preprocessing data once for tuning...")
    test_pds, train_val_pds = get_planting_date(config['dataset']['planting_date_csv'])
    corn_split = get_plot_location_splits(config['dataset']['corn_plot_csv'], test_pds, train_val_pds, "corn")
    soy_split = get_plot_location_splits(config['dataset']['soy_plot_csv'], test_pds, train_val_pds, "soy")
    merged_split = merge_crop_splits(corn_split, soy_split)
    patches_growth_stage = get_growth_stage(
        config['dataset']['image_folder'],
        config['dataset']['growth_stage_csvs'],
        growth_stages={
            'VE': 0, 'V1': 1, 'V2': 2, 'V3': 3, 'V4': 4, 'V5': 5, 'V6': 6, 'V7': 7, 'V8': 8, 'V9': 9,
            'V10': 10, 'V11': 10, 'V12': 10,
            'V13': 11, 'V14': 11, 'V15': 11, 'V16': 11, 'V17': 11, 'V18': 11, 'VT': 11,
            'R1': 12, 'R2': 13, 'R3': 14, 'R4': 15, 'R5': 16, 'R6': 17
        }
    )

    global_data = {
        "config": config,
        "merged_split": merged_split,
        "patches_growth_stage": patches_growth_stage
    }

    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, global_data), n_trials=20)

    print("Best hyperparameters:", study.best_params)
    print("Best Validation MAE:", study.best_value)

    os.makedirs("optuna_results", exist_ok=True)
    study.trials_dataframe().to_csv("optuna_results/optuna_tuning_results.csv")


if __name__ == "__main__":
    main()
