import torch.nn as nn
import torch as th
from torch.nn import functional as F
import pytorch_lightning as pl
import torchvision.transforms as transforms
from PIL import Image
import torchmetrics
from datasetHandler import ImageClassificationDataModule
from pytorch_lightning import Trainer
from pytorch_lightning import callbacks

device = th.device("cuda" if th.cuda.is_available() else "cpu")

th.set_float32_matmul_precision('medium')

class Residual(pl.LightningModule):
    def __init__(self, in_channel, out_channel, use_1x1Conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, stride=strides)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)

        if use_1x1Conv:
            self.conv3 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=strides)
        else:
            self.conv3 = None

    def forward(self, X):
        out = F.relu(self.bn1(self.conv1(X)))
        out = self.bn2(self.conv2(out))
        if self.conv3:
            X = self.conv3(X)
        out += X
        return F.relu(out)

def residualBlock(in_channel, out_channel, num_residuals, first_block=False):
    blks = []
    for i in range(num_residuals):
        if i==0 and not first_block:
            blks.append(Residual(in_channel, out_channel, use_1x1Conv=True,
            strides=2))
        else:
            blks.append(Residual(out_channel, out_channel))
    
    return blks

class ResNet(pl.LightningModule):
    def __init__(self, input_channel, num_classes, learning_rate,
                dropout,
                weight_decay,
                seed):
        super().__init__()

        #  Reproducibility
        th.manual_seed(seed)
        self.b1 = nn.Sequential(
            nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.b2 = nn.Sequential(*residualBlock(64, 64, 2, first_block=True))
        self.b3 = nn.Sequential(*residualBlock(64, 128, 2))
        self.b4 = nn.Sequential(*residualBlock(128, 256, 2))
        self.b5 = nn.Sequential(*residualBlock(256, 512, 2))
        self.finalLayer = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),nn.Linear(512, 10))

        self.b1.apply(self.init_weights)
        self.b2.apply(self.init_weights)
        self.b3.apply(self.init_weights)
        self.b4.apply(self.init_weights)
        self.b5.apply(self.init_weights)
        self.finalLayer.apply(self.init_weights)

        # Metrics
        self.train_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

        
        self.learning_rate = learning_rate
        # Define proportion or neurons to dropout
        self.dropout = nn.Dropout(dropout)
        self.weight_decay = weight_decay
        self.criterion = nn.CrossEntropyLoss()

    def init_weights(self, layer):
        if type(layer) == nn.Conv2d:
            nn.init.kaiming_normal_(layer.weight, mode='fan_out')
        if type(layer) == nn.Linear:
            nn.init.normal_(layer.weight, std=1e-3)
        if type(layer) == nn.BatchNorm2d:
            nn.init.constant_(layer.weight, 1)
            nn.init.constant_(layer.bias, 0)
        


    def forward(self, X):
        out = self.b1(X)
        out = self.b2(out)
        out = self.b3(out)
        out = self.b4(out)
        out = self.dropout(out)
        out = self.b5(out)
        out = self.dropout(out)
        out = self.finalLayer(out)

        return out
    
    def configure_optimizers(self):
        return th.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
    
    def training_step(self, batch, batch_idx):
        
        logs_dict = {'step': float(self.current_epoch)}
        # load data
        images, labels = batch
        
        # Forward pass
        preds = self(images)
        # metrics cal
            
        loss  = self.criterion(preds, labels.squeeze(1))
         
        self.train_accuracy(th.argmax(preds, dim=1), labels.squeeze(1))
        
        logs_dict['train_loss'] = loss
        logs_dict["train_acc"] =  self.train_accuracy
        
        self.log_dict(logs_dict, on_step=False, on_epoch=True, prog_bar=True)

        return logs_dict['train_loss']
    
    def validation_step(self, batch, batch_idx):

        logs_dict = {'step': float(self.current_epoch)}
        # load data
        images, labels = batch
        # Forward pass
        preds = self(images)
        # metrics cal
       
        loss  = self.criterion(preds, labels.squeeze(1))

        self.val_accuracy(th.argmax(preds, dim=1), labels.squeeze(1))

        logs_dict['val_loss'] = loss
        logs_dict["val_acc"] =  self.val_accuracy

        self.log_dict(logs_dict, on_step=False, on_epoch=True, prog_bar=True)
 
        return logs_dict

def load_image(image_path):
    image = Image.open(image_path)
    # Transforme a imagem em um tensor
    transform = transforms.Compose([transforms.ToTensor()])
    image_tensor = transform(image)
    return image_tensor


if __name__ == "__main__":

    # img = load_image("data/dogs/dog.1.jpg")

    # net = ResNet(3,2)

    # img = img.unsqueeze(0) 

    # result = net.forward(img)

    # print(result)

    net = ResNet(
        input_channel=3, 
        num_classes = 10,
        seed = 42,
        learning_rate=0.0001,
        weight_decay=0.005,
        dropout=0.4)

    datamodule = ImageClassificationDataModule(
        batch_size=40,
        n_workers=15,
        pin_memory=True,
        train_set_path= "data/train.csv",
        validation_set_path= "data/valid.csv",
        test_set_path="data/test.csv"
    )

    checkpointng = callbacks.ModelCheckpoint("checkpoints", "best_weight_acc", monitor="val_acc", mode="max", save_top_k=1, save_last= True, verbose=True)

    trainer = Trainer(max_epochs=100, devices=[0], accelerator="gpu",
                      callbacks=[checkpointng])

    trainer.fit(net, datamodule)