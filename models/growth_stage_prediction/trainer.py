# trainer.py
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Any, Dict
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
from collections import defaultdict
import numpy as np


class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, criterion, optimizer, device, config, rank=0):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.rank = rank
        self.config = config
        self.patience = config['training']['patience']
        self.model_save_dir = config['general']['model_save_dir']
        self.writer = SummaryWriter(log_dir=os.path.join(config['general']['log_dir'], f"fold_{rank}"))

    def _train_epoch(self) -> float:
        self.model.train()
        running_loss = 0.0
        pred, true = [], []

        for inputs, labels, path in tqdm(self.train_loader, desc="Training", disable=self.rank != 0):
            inputs, labels = inputs.to(self.device), labels.to(self.device).float()

            self.optimizer.zero_grad()
            outputs = self.model(inputs).squeeze(-1)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            pred.extend(outputs.detach().cpu().numpy())
            true.extend(labels.detach().cpu().numpy())

        epoch_loss = running_loss / len(self.train_loader)
        mae = mean_absolute_error(true, pred)
        mse = mean_squared_error(true, pred)
        return epoch_loss, mse

    def _validate_epoch(self) -> float:
        self.model.eval()
        val_loss = 0.0
        pred, true = [], []

        with torch.no_grad():
            for inputs, labels, path in tqdm(self.val_loader, desc="Validating", disable=self.rank != 0):
                inputs, labels = inputs.to(self.device), labels.to(self.device).float()

                outputs = self.model(inputs).squeeze(-1)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()

                pred.extend(outputs.detach().cpu().numpy())
                true.extend(labels.detach().cpu().numpy())

        epoch_loss = val_loss / len(self.val_loader)
        mae = mean_absolute_error(true, pred)
        mse = mean_squared_error(true, pred)
        return epoch_loss, mse

    def train(self, fold, num_epochs: int) -> None:
        best_val_mse = float("inf")
        best_epoch = 0
        epochs_no_improve = 0

        for epoch in range(num_epochs):
            train_loss, train_mse = self._train_epoch()
            val_loss, val_mse = self._validate_epoch()

            if self.rank == 0:
                print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} MSE: {train_mse:.4f} - Val Loss: {val_loss:.4f} MSE: {val_mse:.4f}")
                self.writer.add_scalar('Loss/train', train_loss, epoch)
                self.writer.add_scalar('Loss/val', val_loss, epoch)
                self.writer.add_scalar('MSE/train', train_mse, epoch)
                self.writer.add_scalar('MSE/val', val_mse, epoch)

            # Early stopping logic
            if val_mse < best_val_mse:
                best_val_mse = val_mse
                best_epoch = epoch
                epochs_no_improve = 0
                if self.rank == 0:
                    #create directory if it doesn't exist
                    os.makedirs(self.model_save_dir, exist_ok=True)
                    torch.save(self.model.state_dict(), os.path.join(self.model_save_dir, f"best_model_fold{fold}.pth"))
                    print(f"Best model saved at epoch {epoch+1} with MSE: {best_val_mse:.4f}")
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= self.patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        self.writer.close()
        return best_val_mse, best_epoch

    def test(self, best_fold) -> Dict[str, Any]:
        model_path = os.path.join(self.model_save_dir, f"best_model_fold{best_fold}.pth")
        print(f"Loading best model from {model_path} for testing...")
        self.model.load_state_dict(torch.load(model_path, map_location=f"cuda:{self.rank}"))
        self.model.to(self.device)
        self.model.eval()

        pred, true, paths = [], [], []

        with torch.no_grad():
            for images, labels, path in tqdm(self.test_loader, desc="Testing", disable=self.rank != 0):
                images, labels = images.to(self.device), labels.to(self.device).float()
                outputs = self.model(images).squeeze(-1)

                pred.extend(outputs.cpu().numpy())
                true.extend(labels.cpu().numpy())
                paths.extend(path)
                #print(f"Test predictions: {outputs.cpu().numpy()}, True labels: {labels.cpu().numpy()}")
        mae = mean_absolute_error(true, pred)
        mse = mean_squared_error(true, pred)
        r2 = r2_score(true, pred)

        #save predictions to a file
        model_pred = {
            "predictions": pred,
            "true_labels": true,
            "paths": paths
        }
        #save csv file
        import pandas as pd
        df = pd.DataFrame(model_pred)
        df.to_csv(os.path.join(self.model_save_dir, f"predictions_fold{best_fold}.csv"), index=False)
        print(f"Test predictions saved to {os.path.join(self.model_save_dir, f'predictions_fold{best_fold}.csv')}")

        from collections import defaultdict
        import numpy as np

        # Group patch predictions by image prefix
        grouped_preds = defaultdict(list)
        grouped_trues = defaultdict(list)

        for p, t, path in zip(pred, true, paths):
            # Remove patch suffix: e.g., "image123_0.jpg" → "image123"
            image_prefix = os.path.basename(path).rsplit('_', 1)[0]
            grouped_preds[image_prefix].append(p)
            grouped_trues[image_prefix].append(t)

        # Aggregate predictions and true values (mean or median)
        image_level_preds = []
        image_level_trues = []

        for key in grouped_preds:
            print(f"Processing image: {key} with {len(grouped_preds[key])} patches")
            patch_preds = grouped_preds[key]
            patch_trues = grouped_trues[key]
            
            # Use mean (or replace with np.median for median)
            image_pred = np.mean(patch_preds)
            image_true = np.mean(patch_trues)  # All should be the same, but we take mean just in case

            image_level_preds.append(image_pred)
            image_level_trues.append(image_true)

        # Evaluate at image level
        image_mae = mean_absolute_error(image_level_trues, image_level_preds)
        image_mse = mean_squared_error(image_level_trues, image_level_preds)
        image_r2 = r2_score(image_level_trues, image_level_preds)

        print(f"\nImage-Level Evaluation:")
        print(f"MAE: {image_mae:.4f}, MSE: {image_mse:.4f}, R2: {image_r2:.4f}")

        import json

        # Create dictionary with image-level predictions
        image_predictions_dict = {
            image_name: float(np.mean(grouped_preds[image_name]))  # or np.median(...) if preferred
            for image_name in grouped_preds
        }

        """
        output_json_path = os.path.join(self.model_save_dir, f"image_level_predictions_fold{best_fold}.json")
        with open(output_json_path, 'w') as f:
            json.dump(image_predictions_dict, f, indent=4)

        print(f"Image-level predictions saved to: {output_json_path}")
        """
        rounded_preds = [round(p) for p in image_level_preds]
        rounded_trues = [round(t) for t in image_level_trues]

        # Compute metrics
        rounded_mae = mean_absolute_error(rounded_trues, rounded_preds)
        rounded_mse = mean_squared_error(rounded_trues, rounded_preds)

        print(f"\nRounded Image-Level Evaluation (Python built-in round):")
        print(f"MAE: {rounded_mae:.4f}, MSE: {rounded_mse:.4f}")


        return {
            "MAE": mae,
            "MSE": mse,
            "RMSE": np.sqrt(mse),
            "R2": r2,
            "pred": pred,
            "true": true,
            "rounded_preds": rounded_preds,
            "rounded_trues": rounded_trues,
            "image_names": list(grouped_preds.keys())
        }

    def inference(self, model_path) -> Dict[str, Any]:
        model_path = os.path.join(model_path)
        print(f"Loading best model from {model_path} for testing...")
        self.model.load_state_dict(torch.load(model_path), strict=False)
        self.model.to(self.device)
        self.model.eval()

        pred, true, paths = [], [], []

        with torch.no_grad():
            for images, path in tqdm(self.test_loader, desc="Testing"):
                images = images.to(self.device)
                outputs = self.model(images).squeeze(-1)
                pred.extend(outputs.cpu().numpy())
                paths.extend(path)
        grouped_preds = defaultdict(list)
      
        for p, path in zip(pred, paths):

            image_prefix = os.path.basename(path).rsplit('_', 1)[0]
            grouped_preds[image_prefix].append(p)
        image_level_preds = []
       

        for key in grouped_preds:
            
            patch_preds = grouped_preds[key]
            image_pred = np.mean(patch_preds)
            image_level_preds.append(image_pred)
        image_predictions_dict = {
            image_name: float(np.mean(grouped_preds[image_name]))  
            for image_name in grouped_preds
        }

        rounded_preds = [round(p) for p in image_level_preds]
   

        return {
            "pred": pred,
            "rounded_preds": rounded_preds,
            "image_names": list(grouped_preds.keys())
        }