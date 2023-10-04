import os
import torch as th
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch.utils.data as tud

class ImageClassificationDataset(tud.Dataset):
    def __int__(self, data_path):
        self.dataframe = pd.read_csv(data_path, index_col=0)    
        
    def __len__(self):
        return len(self.dataframe.index)
        
        
    def __getitem__(self, index):
        
        item = self.dataframe.loc[index]

        input_data = ""
        target_data = ""

        return {
            "image": input_data,
            "label": target_data
        }

class ImageClassificationDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size,
        n_workers,
        pin_memory,
        train_set_path,
        validation_set_path,

    ):
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.pin_memory = pin_memory
        self.train_set_path = train_set_path
        self.validation_set_path = validation_set_path

    def setup(self, stage):
        self.train_set = ImageClassificationDataset(self.train_set_path)
        self.test_set = ImageClassificationDataset(self.validation_set_path)

    def train_dataloader(self):
        return tud.DataLoader(self.train_set,
                            batch_size=self.batch_size,
                            pin_memory=self.pin_memory,
                            shuffle=True,
                            num_workers=self.n_workers)

    def val_dataloader(self):
        return tud.DataLoader(self.val_set,
                            batch_size=1,
                            pin_memory=self.pin_memory,
                            shuffle=False,
                            num_workers=self.n_workers)