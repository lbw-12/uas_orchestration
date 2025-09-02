import torch
import torch.nn as nn
import timm
import torchvision.models as models
import torch.nn.functional as F
from timm.data import resolve_data_config

class GrowthStagePredictionModel(nn.Module):
    def __init__(self, model_name: str, num_classes: int):
        super(GrowthStagePredictionModel, self).__init__()
        self.backbone = timm.create_model(model_name, pretrained=True)
        in_features = self.backbone.get_classifier().in_features
        self.backbone.reset_classifier(0)  

        for i, block in enumerate(self.backbone.blocks):
            if i < 6:
                for param in block.parameters():
                    param.requires_grad = False

        self.head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes)
        )
        

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)