import torch as th
import torch.nn as nn
import torchmetrics
import pytorch_lightning as pl
import torch.nn.functional as F


class ImageClassifier(pl.LightningModule):
    def __init__(self, 
                input_size, 
                kernel_size, 
                stride_kernel_size, 
                max_pool_size, 
                stride_max_pool_size,
                num_classes,
                learning_rate,
                seed
                ):
        super(ImageClassifier, self).__init__()

        #  Reproducibility
        th.manual_seed(seed)
        
        # Architeture
        # Conv layers
        self.conv1 = nn.Conv2d(3, 16, (3,3), 1)  # 3 channels in, 16 channels out, kernel 3x3 e stride 1
        self.conv2 = nn.Conv2d(16, 32, (3,3), 1) # 16 channels in, 32 channels out, kernel 3x3 e stride 1
        self.conv3 = nn.Conv2d(32, 16, (3,3), 1) # 32 channels in, 16 channels out, kernel 3x3 e stride 1

        # Pooling
        self.pool = nn.MaxPool2d((2,2), 2) # Kernel size 2 x 2 stride 1
        
        # Flatten size calculation

        #Convolution layer 1
        hconv1 =int((input_size[0] - kernel_size[0] + stride_kernel_size[0]) / stride_kernel_size[0])
        wconv1 =int((input_size[1] - kernel_size[1] + stride_kernel_size[1]) / stride_kernel_size[1])

        hmaxpool = int((hconv1 - max_pool_size[0] + stride_max_pool_size[0]) / stride_max_pool_size[0])
        wmaxpool = int((wconv1 - max_pool_size[1] + stride_max_pool_size[1]) / stride_max_pool_size[1])

        #Convolution layer 2
        hconv2 =int((hmaxpool - kernel_size[0] + stride_kernel_size[0]) / stride_kernel_size[0])
        wconv2 =int((wmaxpool - kernel_size[1] + stride_kernel_size[1]) / stride_kernel_size[1])

        hmaxpool =int((hconv2 - max_pool_size[0] + stride_max_pool_size[0]) / stride_max_pool_size[0])
        wmaxpool =int((wconv2 - max_pool_size[1] + stride_max_pool_size[1]) / stride_max_pool_size[1])

        #Convolution layer 3
        hconv3 =int((hmaxpool - kernel_size[0] + stride_kernel_size[0]) / stride_kernel_size[0])
        wconv3 =int((wmaxpool - kernel_size[1] + stride_kernel_size[1]) / stride_kernel_size[1])

        hmaxpool =int((hconv3 - max_pool_size[0] + stride_max_pool_size[0]) / stride_max_pool_size[0])
        wmaxpool =int((wconv3 - max_pool_size[1] + stride_max_pool_size[1]) / stride_max_pool_size[1])
                
        flat_size = 16 * hmaxpool * wmaxpool
        
        # Dense Layers
        self.dense1 = nn.Linear(flat_size, 256)
        self.dense2 = nn.Linear(256, 1)

        # Metrics
        self.train_accuracy = torchmetrics.Accuracy(task="binary", num_classes=num_classes)
        self.val_accuracy = torchmetrics.Accuracy(task="binary", num_classes=num_classes)

        self.learning_rate = learning_rate


    def forward(self , x):
        x = self.pool(F.relu(self.conv1(x)))
        
        x = self.pool(F.relu(self.conv2(x)))
        
        x = self.pool(F.relu(self.conv3(x)))
      
        x = x.view(x.size(0), -1)
        x = F.relu(self.dense1(x))
        x = F.sigmoid(self.dense2(x))

        return x

    def configure_optimizers(self):
        return th.optim.Adam(self.parameters(), lr=self.learning_rate)
    
    def training_step(self, batch, batch_idx):
        
        logs_dict = {'step': self.current_epoch}
        # load data
        images, labels = batch
       
        # Forward pass
        preds = self(images)
        # metrics cal
        loss = F.binary_cross_entropy(preds.reshape(-1), labels.reshape(-1).float())
        self.train_accuracy(preds.reshape(-1), labels.reshape(-1).float())
        
        logs_dict['train_loss'] = loss
        logs_dict["train_acc"] =  self.train_accuracy
        
        self.log_dict(logs_dict, on_step=False, on_epoch=True, prog_bar=True)

        return logs_dict['train_loss']
    
    def validation_step(self, batch, batch_idx):

        logs_dict = {'step': self.current_epoch}
        # load data
        images, labels = batch
        # Forward pass
        preds = self(images)
        # metrics cal
        loss = F.binary_cross_entropy(preds.reshape(-1), labels.reshape(-1).float())
        self.val_accuracy(preds.reshape(-1), labels.reshape(-1))

        logs_dict['val_loss'] = loss
        logs_dict["val_acc"] =  self.val_accuracy

        self.log_dict(logs_dict, on_step=False, on_epoch=True, prog_bar=True)
        
        return logs_dict
  