from pytorch_lightning import Trainer
from datasetHandler import ImageClassificationDataModule
from model_bin import ImageClassifier
from pytorch_lightning import callbacks

if __name__ == "__main__":

    datamodule = ImageClassificationDataModule(
        batch_size=1,
        n_workers=10,
        pin_memory=True,
        train_set_path= "data/train.csv",
        validation_set_path= "data/valid.csv",
        test_set_path="data/test.csv"
    )

    imgClass = ImageClassifier(

        input_size = (1360,1360), 
        kernel_size = (3,3), 
        stride_kernel_size = (1,1), 
        max_pool_size = (2,2), 
        stride_max_pool_size = (2,2),
        num_classes = 1,
        seed = 42,
        learning_rate=0.0005,
        weight_decay=0.005,
        dropout=0.15

    )

    checkpointng = callbacks.ModelCheckpoint("checkpoints", "best_weight_acc", monitor="val_acc", mode="max", save_top_k=1, save_last= True, verbose=True)

    trainer = Trainer(max_epochs=100, devices=[0], accelerator="gpu",
                      callbacks=[checkpointng])

    trainer.fit(imgClass, datamodule)
