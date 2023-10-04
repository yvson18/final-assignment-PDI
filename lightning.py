import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchmetrics
import pytorch_lightning as pl
import torch.nn.functional as F
from pytorch_lightning import Trainer

# Hyper-parameters
input_size = 784  # 28x28
hidden_size = 500
num_classes = 10
num_epochs = 10
batch_size = 100
learning_rate = 0.001

# Fully connected neural network with one hidden layer
class LitNeuralNet(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LitNeuralNet, self).__init__()
        
        #Architeture
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

        # Activation
        self.activation = nn.Softmax()

        # Metrics
        self.train_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=10)
        self.val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=10)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        # no activation and no softmax at the end
        return out

    def training_step(self, batch, batch_idx):
        
        logs_dict = {'step': self.current_epoch}
        # load data
        images, labels = batch
        images = images.reshape(-1, 28 * 28)
        # Forward pass
        outputs = self(images)
        probs = self.activation(outputs)
        
        # metrics cal
        pred = torch.argmax(probs, dim=1)
        loss = F.cross_entropy(outputs, labels)
        self.train_accuracy(pred, labels)
        
        logs_dict['train_loss'] = loss
        logs_dict["train_acc"] =  self.train_accuracy
        

        self.log_dict(logs_dict, on_step=False, on_epoch=True, prog_bar=True)

        return logs_dict['train_loss']
    
    def train_dataloader(self):
        # MNIST dataset
        train_dataset = torchvision.datasets.MNIST(
            root="./data", train=True, transform=transforms.ToTensor(), download=True
        )
        # Data loader
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=batch_size, num_workers=4, shuffle=False
        )
        return train_loader

    def val_dataloader(self):
        test_dataset = torchvision.datasets.MNIST(
            root="./data", train=False, transform=transforms.ToTensor()
        )

        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset, batch_size=batch_size, num_workers=4, shuffle=False
        )
        return test_loader
    
    def validation_step(self, batch, batch_idx):

        logs_dict = {'step': self.current_epoch}
        # load data
        images, labels = batch
        images = images.reshape(-1, 28 * 28)
        
        # Forward pass
        outputs = self(images)
        probs = self.activation(outputs)
        
        # metrics cal
        pred = torch.argmax(probs, dim=1)
        loss = F.cross_entropy(outputs, labels)

        self.val_accuracy(pred, labels)

        logs_dict['val_loss'] = loss
        logs_dict["val_acc"] =  self.val_accuracy

        self.log_dict(logs_dict, on_step=False, on_epoch=True, prog_bar=True)
        
        return logs_dict
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=learning_rate)
    
if __name__ == "__main__":
    trainer = Trainer(max_epochs=num_epochs)
    model = LitNeuralNet(input_size, hidden_size, num_classes)
    trainer.fit(model)