# main.py
import os
import yaml
import random
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms
from growth_stage_prediction_model import GrowthStagePredictionModel
from trainer import Trainer
from utils import set_seed, check_data_leakage
from sklearn.model_selection import KFold
from PIL import Image
import pandas as pd
from collections import defaultdict
import gc
from torch.utils.data import Dataset
import pdb
from timm.data import resolve_data_config
from torchvision import transforms
from timm.data.transforms_factory import create_transform
from tqdm import tqdm

loc_list = ['northwest', 'wooster', 'western']
crop_list = ['corn', 'soy']
year_list = ['2023', '2024', '2025']

def load_config(config_path: str):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

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
    random.seed(config['general']['seed'])
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
    print(test_keys, folds)
    
    return test_keys, folds

def get_split_on_planting_date(pd_filename, num_folds=None):
    df = pd.read_csv(pd_filename, index_col=0)
    df_melted = df.reset_index().melt(id_vars="index", var_name="Location", value_name="Date")
    df_melted.rename(columns={"index": "PD"}, inplace=True)
    df_melted["Date"] = pd.to_datetime(df_melted["Date"])
    df_melted["Key"] = df_melted["PD"] + "_" + df_melted["Location"]
    df_sorted = df_melted.sort_values("Date")
    sorted_dict = dict(zip(df_sorted["Key"], df_sorted["Date"]))
    #print(sorted_dict)
    all_keys = list(sorted_dict.keys())
    random.seed(42)
    random.shuffle(all_keys)
    print(all_keys)
    combos = all_keys
    pd_groups = defaultdict(list)
    for combo in combos:
        pdt = combo.split('_')[0].lower()
        pd_groups[pdt].append(combo.lower())
    print(pd_groups)
    keys = list(pd_groups.keys())
    random.seed(42)
    random.shuffle(keys)
    test_pd = keys[0]

    test_pd = test_pd.lower()
    test_keys = pd_groups[test_pd]
    print(f"Test Set (held-out): {test_keys}")
    
    # Remaining PDs for train-val folds
    remaining_pds = keys[1:]

    kf = KFold(n_splits=min(num_folds, len(remaining_pds)), shuffle=True, random_state=42)
    folds = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(remaining_pds)):
        train_keys = [pd_groups[remaining_pds[i]] for i in train_idx]
        val_keys = [pd_groups[remaining_pds[i]] for i in val_idx]
        # Flatten the lists of keys
        train_keys = [item for sublist in train_keys for item in sublist]
        val_keys = [item for sublist in val_keys for item in sublist]
        folds.append((train_keys, val_keys))
        print(f"\nFold {fold+1}:")
        print(f"  Train: {train_keys}")
        print(f"  Val:   {val_keys}")
    
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
    df_numeric = df.applymap(lambda x: growth_stage_dict.get(x, np.nan))
    print(f"Converted growth stages to numeric: {df_numeric.isna().sum().sum()} NaNs")
    
    df_numeric.columns = pd.to_datetime(df_numeric.columns)
    full_range = pd.date_range(start=df_numeric.columns.min(), end=df_numeric.columns.max())
    interpolated = (
        df_numeric
        .transpose()
        .reindex(full_range)
        .interpolate(limit_direction="forward")
        .fillna(method="bfill")               
        .fillna(method="ffill") 
        .transpose()
    )
    interpolated = interpolated.round().clip(lower=0, upper=19)
    interpolated.columns = interpolated.columns.strftime("%Y-%m-%d")
    print(f"Interpolated nans: {interpolated.isna().sum().sum()} NaNs")
    print(f"Interpolated DataFrame:\n{interpolated}")
    return interpolated

def get_growth_stage(image_folder, growthstage_csvfolder, growth_stages):
    
    print(f"Image folder: {len(os.listdir(image_folder))} images")
    count = 0
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
                
                plot_no = patch.split('_')[-3]
                date = patch.split('_')[-2]
                date = pd.to_datetime(date, format="%Y%m%d").date()

                growth_stage = df_stage_only.loc[df_stage_only['plot_no'] == plot_no, str(date)].values[0]
                if pd.notna(growth_stage):
                    count += 1
                    growth_stage_dict[patch]= growth_stage
                else:
                    before_planting_count += 1
    print(f"Total patches processed: {count}")          
           
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

class GrowthStageDataset(Dataset):
    def __init__(self, image_folder, plot_split_keys, growthstage, transform=None):
        assert len(plot_split_keys) == len(growthstage), "Mismatch between plot_split_keys and growthstage length"
        self.image_folder = image_folder
        self.plot_split_keys = plot_split_keys
        self.growthstage = growthstage
        self.transform = transform if transform else transforms.Compose([
            transforms.ToTensor()
        ])
        print(f"transform: {self.transform}")

    def __len__(self):
        return len(self.plot_split_keys)

    def __getitem__(self, idx):
        plot_key = self.plot_split_keys[idx]
        image_path = os.path.join(self.image_folder, plot_key)
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        growth_stage = self.growthstage[idx]
        return image, growth_stage, image_path

def setup(rank, world_size, use_ddp):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group(
        backend="nccl" if torch.cuda.is_available() else "gloo",
        rank=rank,
        world_size=world_size
    )
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def run(rank, world_size, config):
    
    set_seed(config['general']['seed'])

    setup(rank, world_size, config['general']['use_ddp'])

    
    # Prepare data
    test_pds, train_val_pds = get_split_on_planting_date(config['dataset']['planting_date_csv'], num_folds=config['training']['num_folds'])
    corn_split = get_plot_location_splits(config['dataset']['corn_plot_csv'], test_pds, train_val_pds, "corn")
    soy_split = get_plot_location_splits(config['dataset']['soy_plot_csv'], test_pds, train_val_pds, "soy")
    merged_split = merge_crop_splits(corn_split, soy_split)
    print(f"Test Set (held-out): {merged_split['test_keys']}")
    patches_growth_stage = get_growth_stage(
        config['dataset']['image_folder'],
        config['dataset']['growth_stage_csvs'],
        growth_stages={
            'VE': 0, 'V1': 1, 'V2': 2, 'V3': 3, 'V4': 4, 'V5': 5, 'V6': 6, 'V7': 7, 'V8': 8, 'V9': 9,
            'V10': 10, 'V11': 10, 'V12': 10,
            'V13': 11, 'V14': 11, 'V15': 11, 'V16': 11, 'V17': 11, 'V18': 11, 'VT': 11,
            'R1': 12, 'R2': 13, 'R3': 14, 'R4': 15, 'R5': 16, 'R6': 17, 'R7': 18, 'R8': 19
        }
    )

    best_fold_mse = float("inf")

    for fold, split in merged_split['cv_splits'].items():

        # initialize model
        model = GrowthStagePredictionModel(config['model']['name'], config['model']['num_classes']).to(rank)
        configuration = resolve_data_config({}, model=model.backbone)
        transform = create_transform(**configuration)
        print(f"Model {model} loaded with {configuration}")
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])
        criterion = torch.nn.MSELoss()
        
        print(f"Rank {rank} -> Starting Fold {fold}...")

        dataset = {"train_img": [], "val_img": [], "test_img": [], "train_gs": [], "val_gs": [], "test_gs": []}

        train_keys = merged_split['cv_splits'][fold]['train']
        val_keys = merged_split['cv_splits'][fold]['val']
        test_keys = merged_split['test']
        
        for patch in os.listdir(config['dataset']['image_folder']):
            loc = next((l for l in ['northwest', 'wooster', 'western'] if l in patch), None)
            crop = next((c for c in ['corn', 'soy'] if c in patch), None)
            plot = patch.split('_')[-3]

            if all([loc, crop, plot]):
                full_key = f"{plot}_{loc}_{crop}"
                train_matches = [s for s in train_keys if s == full_key]
                val_matches = [s for s in val_keys if s == full_key]
                test_matches = [s for s in test_keys if s == full_key]


            if train_matches and patch in patches_growth_stage.keys():
                dataset['train_img'].append(patch)
                dataset['train_gs'].append(patches_growth_stage[patch])
            elif val_matches and patch in patches_growth_stage.keys():
                dataset['val_img'].append(patch)
                dataset['val_gs'].append(patches_growth_stage[patch])
            elif test_matches and patch in patches_growth_stage.keys():
                dataset['test_img'].append(patch)
                dataset['test_gs'].append(patches_growth_stage[patch])
        print(f"len(train_img): {len(dataset['train_img'])}, len(val_img): {len(dataset['val_img'])}, len(test_img): {len(dataset['test_img'])}")

        if rank == 0:
            check_data_leakage(dataset['train_img'], dataset['val_img'], dataset['test_img'])

        train_dataset = GrowthStageDataset(config['dataset']['image_folder'], dataset['train_img'], dataset['train_gs'], transform)
        val_dataset = GrowthStageDataset(config['dataset']['image_folder'], dataset['val_img'], dataset['val_gs'], transform)
        test_dataset = GrowthStageDataset(config['dataset']['image_folder'], dataset['test_img'], dataset['test_gs'], transform)

        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)

        train_loader = DataLoader(
            train_dataset,
            batch_size=config['training']['batch_size'],  # batch size PER GPU
            sampler=train_sampler,
            num_workers=4,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['training']['batch_size'],  # batch size PER GPU
            sampler=val_sampler,
            num_workers=4,
            pin_memory=True
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=config['training']['batch_size'],  # batch size PER GPU
            sampler=test_sampler,
            num_workers=4,
            pin_memory=True
        )
        from collections import defaultdict
        image_patch_groups = defaultdict(list)

        for images, labels, paths in tqdm(train_loader, desc="Combining Train Data", disable=rank != 0):
            for path in paths:
                # Example path: "western_bftb_om_corn_rgb_419_20240711_6.tif"
                filename = path.split("/")[-1]  # in case there's a full path
                base_name = "_".join(filename.split("_")[:-1])  # drop the last part (patch number)
                image_patch_groups[base_name].append(path)

        # Count how many unique whole images are present
        num_whole_images = len(image_patch_groups)

        print(f"Number of unique whole images: {num_whole_images}")

        for images, labels, paths in tqdm(val_loader, desc="Combining Val Data", disable=rank != 0):
            for path in paths:
                filename = path.split("/")[-1]
                base_name = "_".join(filename.split("_")[:-1])
                image_patch_groups[base_name].append(path)
        for images, labels, paths in tqdm(test_loader, desc="Combining Test Data", disable=rank != 0):
            for path in paths:
                filename = path.split("/")[-1]
                base_name = "_".join(filename.split("_")[:-1])
                image_patch_groups[base_name].append(path)
        # Count how many unique whole images are present after combining
        num_whole_images_combined = len(image_patch_groups)
        print(f"Number of unique whole images after combining: {num_whole_images_combined}")
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Val dataset size: {len(val_dataset)}")
        print(f"Test dataset size: {len(test_dataset)}")
        

        trainer = Trainer(model, train_loader, val_loader, test_loader, criterion, optimizer, rank, config, rank)
        if config['training']['train']:
            print(f"Training on fold {fold}...")
            best_val_mse, best_epoch = trainer.train(fold, num_epochs=config['training']['num_epochs'])

            if best_val_mse < best_fold_mse and rank == 0:
                best_fold_mse = best_val_mse
                best_fold = fold

    if rank == 0 and config['training']['test']:
        
        print(f"Testing on best fold {best_fold}...")
        results = trainer.test(best_fold)
        print(f"Test MAE: {results['MAE']:.4f}")
        print(f"Test RMSE: {results['RMSE']:.4f}")
        print(f"Test R2: {results['R2']:.4f}")
        print(f"Test MSE: {results['MSE']:.4f}")


    #combining train val test datasets
    combined_dict = {}

    true, paths = [], []
    for images, labels, path in tqdm(train_loader, desc="Combining Train Data", disable=rank != 0):
        true.extend(labels.detach().cpu().numpy())
        paths.extend(path)
    
    for images, labels, path in tqdm(val_loader, desc="Combining Val Data", disable=rank != 0):
        true.extend(labels.detach().cpu().numpy())
        paths.extend(path)

    #group patches to get image level values
    from collections import defaultdict
    import numpy as np

    # Group patch predictions by image prefix
    grouped_trues = defaultdict(list)

    for t, path in zip(true, paths):
        # Remove patch suffix: e.g., "image123_0.jpg" → "image123"
        image_prefix = os.path.basename(path).rsplit('_', 1)[0]
        grouped_trues[image_prefix].append(t)

    # Aggregate predictions and true values (mean or median)
    image_level_trues = []
    for key in grouped_trues:
        print(f"Processing image: {key} with {len(grouped_trues[key])} patches")
        patch_trues = grouped_trues[key]
        
        # Use mean (or replace with np.median for median)
        image_true = np.mean(patch_trues)

        image_level_trues.append(image_true)

    # Round the true values to the nearest integer
    rounded_trues = [round(t) for t in image_level_trues]
        
    #combine train, val, and test rounded gs and path

    rounded_true_dict = {
        path: int(true)
        for path, true in zip(grouped_trues.keys(), rounded_trues)
    }

    results_dict = {
        path: int(pred)
        for path, pred in zip(results["image_names"], results["rounded_preds"])
    }

    combined_dict = {**rounded_true_dict, **results_dict}

    print(f"Combined results: {len(combined_dict)} entries")
    #save as json 
    import json
    output_path = os.path.join(config['general']['model_save_dir'], f"combined_results_fold_{best_fold}.json")
    with open(output_path, 'w') as f:
        json.dump(combined_dict, f, indent=4)
    cleanup()

def main():
    config = load_config("configs/config.yaml")
    world_size = torch.cuda.device_count() if config['general']['use_ddp'] else 1

    if config['general']['use_ddp']:
        mp.spawn(run, args=(world_size, config), nprocs=world_size, join=True)
    else:
        run(rank=0, world_size=1, config=config)

if __name__ == "__main__":
    main()
