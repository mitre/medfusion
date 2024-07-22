from pathlib import Path
from datetime import datetime

import torch
from torch.utils.data import ConcatDataset
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import os


from medical_diffusion.data.datamodules import SimpleDataModule
from medical_diffusion.data.datasets import (
    AIROGSDataset,
    MSIvsMSS_2_Dataset,
    CheXpert_2_Dataset,
    SimpleDataset2D,
    RFMID_Dataset,
    OCT_2_Dataset,
    IDRID_Dataset,
    DicomDataset,
    DicomDataset3D
)
from medical_diffusion.models.embedders.latent_embedders import (
    VQVAE,
    VQGAN,
    VAE,
    VAEGAN,
)
from torch.cuda.amp import GradScaler, autocast
from pytorch_lightning.loggers import TensorBoardLogger


import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")

if __name__ == "__main__":

    # --------------- Settings --------------------
    current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    path_run_dir = Path.cwd() / "runs" / str(current_time)
    path_run_dir.mkdir(parents=True, exist_ok=True)
    gpus = [0] if torch.cuda.is_available() else None

    # ------------ Load Data ----------------
    # ds_1 = AIROGSDataset( #  256x256
    #     crawler_ext='jpg',
    #     augment_horizontal_flip=True,
    #     augment_vertical_flip=True,
    #     # path_root='/home/gustav/Documents/datasets/AIROGS/dataset',
    #     path_root='/mnt/hdd/datasets/eye/AIROGS/data_256x256',
    # )

    # ds_2 = MSIvsMSS_2_Dataset( #  512x512
    #     # image_resize=256,
    #     crawler_ext='jpg',
    #     augment_horizontal_flip=True,
    #     augment_vertical_flip=True,
    #     # path_root='/home/gustav/Documents/datasets/Kather_2/train'
    #     path_root='/mnt/hdd/datasets/pathology/kather_msi_mss_2/train/'
    # )

    # ds_3 = CheXpert_2_Dataset( #  256x256
    #     # image_resize=128,
    #     augment_horizontal_flip=False,
    #     augment_vertical_flip=False,
    #     # path_root = '/home/gustav/Documents/datasets/CheXpert/preprocessed_tianyu'
    #     path_root = '/mnt/hdd/datasets/chest/CheXpert/ChecXpert-v10/preprocessed_tianyu'
    # )
    # ds_4 = OCT_2_Dataset("/projects/NEI/pranay/Eyes/Datasets/OCT/zipped_data/OCT_Train_512",crawler_ext='jpeg')
    # ds_4_val = OCT_2_Dataset("/projects/NEI/pranay/Eyes/Datasets/OCT/zipped_data/OCT_Test_512",crawler_ext='jpeg')

    # ds_4 = RFMID_Dataset("/projects/NEI/pranay/Eyes/Datasets/A. RFMiD_All_Classes_Dataset/1. Original Images Processed 5/a. Training Set",
    #                      "/projects/NEI/pranay/Eyes/Datasets/A. RFMiD_All_Classes_Dataset/2. Groundtruths/a. RFMiD_Training_Labels_mod.csv", crawler_ext="png")
    # ds_4_val = RFMID_Dataset("/projects/NEI/pranay/Eyes/Datasets/A. RFMiD_All_Classes_Dataset/1. Original Images Processed 5/b. Validation Set",
    #                      "/projects/NEI/pranay/Eyes/Datasets/A. RFMiD_All_Classes_Dataset/2. Groundtruths/b. RFMiD_Validation_Labels_mod.csv", crawler_ext="png")
    # ds_4 = deeplake.load("hub://activeloop/diabetic-retinopathy-detection-train")
    # ds_4_val = deeplake.load("hub://activeloop/diabetic-retinopathy-detection-val")
    
    path_to_data = "/projects/NEI/pranay/Eyes/Datasets/OCT Samples"


    # create the datasets
    dataset_type1 = DicomDataset3D(path_to_data, image_resize=(32, 224, 128))
    # dataset_type2 = DicomDataset(list_of_dicom_files_type2)
    # # create the dataloaders
    # loader_type1 = DataLoader(dataset_type1, batch_size=10, num_workers=2)
    # loader_type2 = DataLoader(dataset_type2, batch_size=10, num_workers=2)

    # ds = ConcatDataset([ds_1, ds_2, ds_3])

    dm = SimpleDataModule(
        ds_train=dataset_type1, ds_val=dataset_type1, batch_size=1, num_workers=5, pin_memory=True
    )

    # ------------ Initialize Model ------------
    model = VAE(
        in_channels=1,
        out_channels=1,
        emb_channels=8,
        spatial_dims=3,
        hid_chs=[16, 32, 64],
        kernel_sizes=[3, 3, 3],
        strides=[1, 2, 2],
        deep_supervision=1,
        # dropout=0.5,
        use_attention="none",
        loss=torch.nn.MSELoss,
        # optimizer_kwargs={'lr':1e-3},
        # lr_scheduler = torch.optim.lr_scheduler.StepLR,
        # lr_scheduler_kwargs = {
        #     'step_size': 30,  # Number of epochs after which to reduce the LR
        #     'gamma': 0.1  # Factor to reduce LR by
        #     },
        # embedding_loss_weight=1e-6,
        sample_every_n_steps=100,
    )

    # model.load_pretrained(Path('/projects/NEI/pranay/Eyes/medfusion/runs/2024_03_08_142013/last.ckpt'), strict=True)

    # model = VAEGAN(
    #     in_channels=3,
    #     out_channels=3,
    #     emb_channels=8,
    #     spatial_dims=2,
    #     hid_chs =    [ 64, 128, 256,  512],
    #     deep_supervision=1,
    #     use_attention= 'none',
    #     start_gan_train_step=-1,
    #     embedding_loss_weight=1e-6
    # )

    # model.vqvae.load_pretrained(Path.cwd()/'runs/2022_11_25_082209_chest_vae/last.ckpt')
    # model.load_pretrained(Path.cwd()/'runs/2022_11_25_232957_patho_vaegan/last.ckpt')

    # model = VQVAE(
    #     in_channels=3,
    #     out_channels=3,
    #     emb_channels=4,
    #     num_embeddings = 8192,
    #     spatial_dims=2,
    #     hid_chs =    [64, 128, 256, 512],
    #     embedding_loss_weight=1,
    #     beta=1,
    #     loss = torch.nn.L1Loss,
    #     deep_supervision=1,
    #     use_attention = 'none',
    # )

    # model = VQGAN(
    #     in_channels=3,
    #     out_channels=3,
    #     emb_channels=4,
    #     num_embeddings = 8192,
    #     spatial_dims=2,
    #     hid_chs =    [64, 128, 256, 512],
    #     embedding_loss_weight=1,
    #     beta=1,
    #     start_gan_train_step=-1,
    #     pixel_loss = torch.nn.L1Loss,
    #     deep_supervision=1,
    #     use_attention='none',
    # )

    # model.vqvae.load_pretrained(Path.cwd()/'runs/2022_12_13_093727_patho_vqvae/last.ckpt')

    # -------------- Training Initialization ---------------
    to_monitor = "val/loss"  # "train/L1"  #
    min_max = "min"
    save_and_sample_every = 50
    tensorboard_logger = TensorBoardLogger(
        "tb_logs", name="train_VAE_IDRID_preprocessed_1024_lr-10-3"
    )

    early_stopping = EarlyStopping(
        monitor=to_monitor,
        min_delta=0.0,  # minimum change in the monitored quantity to qualify as an improvement
        patience=30,  # number of checks with no improvement
        mode=min_max,
    )
    checkpointing = ModelCheckpoint(
        dirpath=str(path_run_dir),  # dirpath
        monitor=to_monitor,
        every_n_train_steps=save_and_sample_every,
        save_last=True,
        save_top_k=5,
        mode=min_max,
    )
    trainer = Trainer(
        logger=tensorboard_logger,
        accelerator="gpu",
        devices="auto",
        strategy="ddp",
        precision=16,
        # amp_backend='apex',
        # amp_level='O2',
        # gradient_clip_val=0.5,
        default_root_dir=str(path_run_dir),
        # callbacks=[checkpointing],
        callbacks=[checkpointing, early_stopping],
        enable_checkpointing=True,
        check_val_every_n_epoch=1,
        log_every_n_steps=1,
        auto_lr_find=False,
        # limit_train_batches=1000,
        limit_val_batches=10,  # 0 = disable validation - Note: Early Stopping no longer available
        min_epochs=100,
        max_epochs=1001,
        num_sanity_val_steps=2,
    )

    # ---------------- Execute Training ----------------
    trainer.fit(model, datamodule=dm)

    # ------------- Save path to best model -------------
    model.save_best_checkpoint(trainer.logger.log_dir, checkpointing.best_model_path)
