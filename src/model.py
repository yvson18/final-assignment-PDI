import torch as th
import torch.nn as nn
import torchmetrics
import pytorch_lightning as pl
import torch.nn.functional as F

device = th.device("cuda" if th.cuda.is_available() else "cpu")

th.set_float32_matmul_precision('medium')

class ImageClassifier(pl.LightningModule):
    def __init__(self, 
                input_size, 
                kernel_size, 
                stride_kernel_size, 
                max_pool_size, 
                stride_max_pool_size,
                num_classes,
                learning_rate,
                dropout,
                weight_decay,
                seed
                ):
        super(ImageClassifier, self).__init__()

        #  Reproducibility
        th.manual_seed(seed)
        
        # Architeture
        # Conv layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size,stride_kernel_size)  # 3 channels in, 16 channels out, kernel 3x3 e stride 1
        self.conv2 = nn.Conv2d(16, 32, kernel_size,stride_kernel_size) # 16 channels in, 32 channels out, kernel 3x3 e stride 1
        self.conv3 = nn.Conv2d(32, 16, kernel_size,stride_kernel_size) # 32 channels in, 16 channels out, kernel 3x3 e stride 1

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
        self.dense2 = nn.Linear(256, 10)

        # Metrics
        self.train_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

        
        self.learning_rate = learning_rate
        # Define proportion or neurons to dropout
        #self.dropout = nn.Dropout(dropout)
        self.weight_decay = weight_decay
        self.criterion = nn.CrossEntropyLoss()



    def forward(self , x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        #x = self.dropout(x)
        x = self.pool(F.relu(self.conv3(x)))
      
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.dense1(x))
        x = self.dropout(x)
        x = F.sigmoid(self.dense2(x))

        return x

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


# import torchvision.transforms as transforms
# from PIL import Image

# def load_image(image_path):
#     image = Image.open(image_path)
#     # Transforme a imagem em um tensor
#     transform = transforms.Compose([transforms.ToTensor()])
#     image_tensor = transform(image)
#     return image_tensor

# img = load_image("data/dogs/dog.1.jpg")

# imgClass = ImageClassifier(

#         input_size = (256,256), 
#         kernel_size = (3,3), 
#         stride_kernel_size = (1,1), 
#         max_pool_size = (2,2), 
#         stride_max_pool_size = (2,2),
#         num_classes = 2,
#         seed = 42,
#         learning_rate=0.0005,
#         weight_decay=0.005,
#         dropout=0.15

# )

# imgClass.forward(img)