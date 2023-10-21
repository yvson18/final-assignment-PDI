import os
import torch as th
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch.utils.data as tud
import torchvision.transforms as transforms
from PIL import Image

def load_image(image_path):
    image = Image.open(image_path)
    # Transforme a imagem em um tensor
    transform = transforms.Compose([transforms.ToTensor()])
    image_tensor = transform(image)
    return image_tensor

class ImageClassificationDataset(tud.Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.dataframe = pd.read_csv(data_path)    
        
    def __len__(self):
        return len(self.dataframe.index)
        
        
    def __getitem__(self, index):
        item = self.dataframe.loc[index]

        input_data = load_image(item["path"])
        target_data = th.from_numpy(np.array([int(item["label"])]))

        return input_data, target_data

class ImageClassificationDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size,
        n_workers,
        pin_memory,
        train_set_path,
        validation_set_path,
        test_set_path

    ):
        super().__init__()

        self.batch_size = batch_size
        self.n_workers = n_workers
        self.pin_memory = pin_memory
        self.train_set_path = train_set_path
        self.validation_set_path = validation_set_path
        self.test_set_path = test_set_path

    def setup(self, stage):
        self.train_set = ImageClassificationDataset(self.train_set_path)
        self.val_set = ImageClassificationDataset(self.validation_set_path)
        self.test_set = ImageClassificationDataset(self.test_set_path)

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
    
    def test_dataloader(self):
        return tud.DataLoader(self.test_set,
                              batch_size=1,
                              pin_memory=self.pin_memory,
                              shuffle=False,
                              num_workers=self.n_workers)
    

# img = ImageClassificationDataset("data/train.csv").__getitem__(0)
# print(img[0].shape)

# dataset = pd.read_csv("data/test.csv")

# tam = len(dataset.index)

# #check if all the images have 3 channels if not remove them from the folder

# for i in range(tam):
#     img = Image.open(dataset["path"][i])
#     if img.mode != "RGB":
#         os.remove(dataset["path"][i])
#         #dataset.drop(i, inplace=True)
        