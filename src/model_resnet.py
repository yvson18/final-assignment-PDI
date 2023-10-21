import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import torchmetrics
import pytorch_lightning as pl
import torch.nn.functional as F
from pytorch_lightning import callbacks

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.set_float32_matmul_precision('medium')

class block(pl.LightningModule):
    def __init__(
        self, in_channels, intermediate_channels, identity_downsample=None, stride=1
    ):
        super().__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(
            in_channels,
            intermediate_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(intermediate_channels)
        self.conv2 = nn.Conv2d(
            intermediate_channels,
            intermediate_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(intermediate_channels)
        self.conv3 = nn.Conv2d(
            intermediate_channels,
            intermediate_channels * self.expansion,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(intermediate_channels * self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        self.stride = stride

    def forward(self, x):
        identity = x.clone()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x


class ResNet(pl.LightningModule):
    def __init__(self,
                block, 
                layers, 
                image_channels, 
                num_classes,
                learning_rate,
                dropout,
                weight_decay,
                seed
                ):
        
        super(ResNet, self).__init__()
        #  Reproducibility
        torch.manual_seed(seed)

        self.in_channels = 64
        self.conv1 = nn.Conv2d(
            image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Essentially the entire ResNet architecture are in these 4 lines below
        self.layer1 = self._make_layer(
            block, layers[0], intermediate_channels=64, stride=1
        )
        self.layer2 = self._make_layer(
            block, layers[1], intermediate_channels=128, stride=2
        )
        self.layer3 = self._make_layer(
            block, layers[2], intermediate_channels=256, stride=2
        )
        self.layer4 = self._make_layer(
            block, layers[3], intermediate_channels=512, stride=2
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, 10)

        # Metrics
        self.train_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

        
        self.learning_rate = learning_rate
        # Define proportion or neurons to dropout
        self.dropout = nn.Dropout(dropout)
        self.weight_decay = weight_decay
        self.criterion = nn.CrossEntropyLoss()


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x
    
    def _make_layer(self, block, num_residual_blocks, intermediate_channels, stride):
        identity_downsample = None
        layers = []

        # Either if we half the input space for ex, 56x56 -> 28x28 (stride=2), or channels changes
        # we need to adapt the Identity (skip connection) so it will be able to be added
        # to the layer that's ahead
        if stride != 1 or self.in_channels != intermediate_channels * 4:
            identity_downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    intermediate_channels * 4,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(intermediate_channels * 4),
            )

        layers.append(
            block(self.in_channels, intermediate_channels, identity_downsample, stride)
        )

        # The expansion size is always 4 for ResNet 50,101,152
        self.in_channels = intermediate_channels * 4

        # For example for first resnet layer: 256 will be mapped to 64 as intermediate layer,
        # then finally back to 256. Hence no identity downsample is needed, since stride = 1,
        # and also same amount of channels.
        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, intermediate_channels))

        return nn.Sequential(*layers)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
    
    def training_step(self, batch, batch_idx):
        
        logs_dict = {'step': float(self.current_epoch)}
        # load data
        images, labels = batch
        
        # Forward pass
        preds = self(images)
        # metrics cal
            
        loss  = self.criterion(preds, labels.squeeze(1))
         
        self.train_accuracy(torch.argmax(preds, dim=1), labels.squeeze(1))
        
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

        self.val_accuracy(torch.argmax(preds, dim=1), labels.squeeze(1))

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

from datasetHandler import ImageClassificationDataModule
from pytorch_lightning import Trainer



if __name__ == "__main__":
    
    # img = load_image("data/dogs/dog.1.jpg")
    # net = ResNet(block, [3, 4, 6, 3], image_channels=3, num_classes=1)
    
    # # Add an extra dimension for batch
    # img = img.unsqueeze(0) 

    # net.forward(img)

    # print(net.forward(img))

    net = ResNet(block, [3, 4, 6, 3], image_channels=3, num_classes = 10,
        seed = 42,
        learning_rate=0.0005,
        weight_decay=0.005,
        dropout=0.15 )

    datamodule = ImageClassificationDataModule(
        batch_size=18,
        n_workers=10,
        pin_memory=True,
        train_set_path= "data/train.csv",
        validation_set_path= "data/valid.csv",
        test_set_path="data/test.csv"
    )

    checkpointng = callbacks.ModelCheckpoint("checkpoints", "best_weight_acc", monitor="val_acc", mode="max", save_top_k=1, save_last= True, verbose=True)

    trainer = Trainer(max_epochs=150, devices=[0], accelerator="gpu",
                      callbacks=[checkpointng])

    trainer.fit(net, datamodule)