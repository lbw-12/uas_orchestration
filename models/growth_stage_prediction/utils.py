# utils.py
import os
import random
import numpy as np
import torch

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def check_data_leakage(train_list, val_list, test_list):
    train_set = set(train_list)
    val_set = set(val_list)
    test_set = set(test_list)
    
    assert len(train_set.intersection(val_set)) == 0, "Data Leakage between Train and Val!"
    assert len(train_set.intersection(test_set)) == 0, "Data Leakage between Train and Test!"
    assert len(val_set.intersection(test_set)) == 0, "Data Leakage between Val and Test!"
    print("✅ No data leakage detected between Train, Val, and Test splits.")
