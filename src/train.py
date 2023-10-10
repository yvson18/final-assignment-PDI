from pytorch_lightning import Trainer
from src.datasetHandler import ImageClassificationDataModule
from src.model import ImageClassifier


if __name__ == "__main__":

    datamodule = ImageClassificationDataModule(
        batch_size=16,
        n_workers=10,
        pin_memory=True,
        train_set_path= "train.csv",
        validation_set_path= "valid.csv",
        test_set_path="test.csv"
    )

    imgClass = ImageClassifier(

        input_size = (256,256), 
        kernel_size = (3,3), 
        stride_kernel_size = (1,1), 
        max_pool_size = (2,2), 
        stride_max_pool_size = (2,2),
        num_classes = 2,
        seed = 42,
        learning_rate=0.0005
    )

    trainer = Trainer(max_epochs=100)

    trainer.fit(imgClass, datamodule)
