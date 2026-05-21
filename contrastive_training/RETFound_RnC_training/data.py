import pandas as pd
import os
import numpy as np
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from PIL import Image
from util.utils import TwoCropTransform

class CustomDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['Path']  
        age = self.dataframe.iloc[idx]['age']  
        
        try:
            image = Image.open(img_path)
            if self.transform:
                image = self.transform(image)
            return image, age, img_path
            
        except Exception as e:
            print(f"Error loading image at index {idx}: {e}")
            return None, None, None

def load_data(args, system='hpc', train_size=0.9, val_size=0.1, random_state=42, take_first_raws = None, quality_reject_threshold=0.8):
    
    path_table_file = args.path_table_file
    path_table = pd.read_csv(path_table_file)

    if take_first_raws is not None:
        path_table = path_table.iloc[:take_first_raws]
    
    path_table_train_val = path_table[path_table['fold']!=args.leave_fold_out]
    path_table_test = path_table[path_table['fold']==args.leave_fold_out]

    grouped_df = path_table_train_val.groupby('eid')
    eids = list(grouped_df.groups.keys())
    train_eids, val_eids = train_test_split(eids, test_size=val_size, random_state=random_state)
    
    train_df = path_table_train_val[path_table_train_val['eid'].isin(train_eids)]
    val_df = path_table_train_val[path_table_train_val['eid'].isin(val_eids)]
    test_df = path_table_test
    
    return train_df, val_df, test_df

def create_loaders(train_df, val_df, test_df, batch_size=128, train_transform=None, val_transform=None, num_workers=2):
    
    train_dataset = CustomDataset(train_df, transform=train_transform)
    eval_train_dataset = CustomDataset(train_df, transform=val_transform)
    val_dataset = CustomDataset(val_df, transform=val_transform)
    test_dataset = CustomDataset(test_df, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)
    eval_train_loader = DataLoader(eval_train_dataset, batch_size=batch_size)
    eval_val_loader = DataLoader(val_dataset, batch_size=batch_size)
    eval_test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader, eval_train_loader, eval_val_loader, eval_test_loader

def create_RnC_loaders(train_df, val_df, test_df, batch_size=128, train_transform=None, val_transform=None, num_workers=2):
    # Following logic from https://github.com/kaiwenzha/Rank-N-Contrast/blob/main/utils.py

    train_dataset = CustomDataset(train_df, transform=TwoCropTransform(train_transform))
    eval_train_dataset = CustomDataset(train_df, transform=val_transform)
    val_dataset = CustomDataset(val_df, transform=val_transform)
    test_dataset = CustomDataset(test_df, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True) # will be empty if len(df) < batch_size!
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)
    eval_train_loader = DataLoader(eval_train_dataset, batch_size=batch_size)
    eval_val_loader = DataLoader(val_dataset, batch_size=batch_size)
    eval_test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader, eval_train_loader, eval_val_loader, eval_test_loader

