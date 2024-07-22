
from email.mime import audio
from pathlib import Path
from datetime import datetime

import torch 
import torch.nn as nn
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import numpy as np 
import torchio as tio 

from medical_diffusion.data.datamodules import SimpleDataModule
from medical_diffusion.data.datasets import AIROGSDataset, MSIvsMSS_2_Dataset, CheXpert_2_Dataset, SimpleDataset2D, OCT_2_Dataset, RFMID_Dataset, IDRID_Dataset, DicomDataset3D, TOPCON_Dataset
from medical_diffusion.models.pipelines import DiffusionPipeline
from medical_diffusion.models.estimators import UNet
from medical_diffusion.external.stable_diffusion.unet_openai import UNetModel
from medical_diffusion.models.noise_schedulers import GaussianNoiseScheduler
from medical_diffusion.models.embedders import LabelEmbedder, LabelEmbedderRFMID, TimeEmbbeding, IDRIDLabelEmbedder
from medical_diffusion.models.embedders.latent_embedders import VAE, VAEGAN, VQVAE, VQGAN

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')



if __name__ == "__main__":
    # ------------ Load Data ----------------
    # ds = AIROGSDataset(
    #     crawler_ext='jpg',
    #     augment_horizontal_flip = False,
    #     augment_vertical_flip = False, 
    #     # path_root='/home/gustav/Documents/datasets/AIROGS/data_256x256/',
    #     path_root='/mnt/hdd/datasets/eye/AIROGS/data_256x256',
    # )

    # ds = MSIvsMSS_2_Dataset(
    #     crawler_ext='jpg',
    #     image_resize=None,
    #     image_crop=None,
    #     augment_horizontal_flip=False,
    #     augment_vertical_flip=False, 
    #     # path_root='/home/gustav/Documents/datasets/Kather_2/train',
    #     path_root='/mnt/hdd/datasets/pathology/kather_msi_mss_2/train/',
    # )

    # ds = CheXpert_2_Dataset( #  256x256
    #     augment_horizontal_flip=False,
    #     augment_vertical_flip=False,
    #     path_root = '/mnt/hdd/datasets/chest/CheXpert/ChecXpert-v10/preprocessed_tianyu'
    # )
    # ds = OCT_2_Dataset("/projects/COMPXR/pranay/Eyes/Datasets/OCT/zipped_data/OCT_Train_512", crawler_ext="jpeg")
    # ds_4_val = OCT_2_Dataset("/projects/COMPXR/pranay/Eyes/Datasets/OCT/zipped_data/OCT_Train_512", crawler_ext="jpeg")
    ds_4 = TOPCON_Dataset(
        "/projects/NEI/pranay/Eyes/Datasets/topcon_screen_for_mitre_split/train"
    )
    ds_4_val = TOPCON_Dataset(
        "/projects/NEI/pranay/Eyes/Datasets/topcon_screen_for_mitre_split/val"
    )
    # path_to_data = "/projects/NEI/pranay/Eyes/Datasets/OCT Samples"

    # dataset_type1 = DicomDataset3D(path_to_data, image_resize=(32, 224, 128))
  
    dm = SimpleDataModule(
        ds_train = ds_4,
        ds_val = ds_4_val,
        batch_size=10, 
        num_workers=5,
        pin_memory=True,
        # weights=ds.get_weights()
    ) 
    
    current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    path_run_dir = Path.cwd() / 'runs' / str(current_time)
    path_run_dir.mkdir(parents=True, exist_ok=True)
    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'



    # ------------ Initialize Model ------------
    # cond_embedder = None 
    # cond_embedder = IDRIDLabelEmbedder
    # cond_embedder_kwargs = {
    #     'emb_dim': 1024,
    #     'num_classes': 2
    # }
    cond_embedder = LabelEmbedder
    cond_embedder_kwargs = {
        'emb_dim': 1024,
        'num_classes': 5
    }
 

    time_embedder = TimeEmbbeding
    time_embedder_kwargs ={
        'emb_dim': 1024 # stable diffusion uses 4*model_channels (model_channels is about 256)
    }


    noise_estimator = UNet
    noise_estimator_kwargs = {
        'in_ch':8, 
        'out_ch':8, 
        'spatial_dims':2,
        'hid_chs':  [  256, 256, 512, 1024],
        'kernel_sizes':[3, 3, 3, 3],
        'strides':     [1, 2, 2, 2],
        'time_embedder': time_embedder,
        'time_embedder_kwargs': time_embedder_kwargs,
        'cond_embedder':None ,
        'cond_embedder_kwargs':  None,
        'deep_supervision': False,
        'use_res_block':True,
        'use_attention':'none',
    }


    # ------------ Initialize Noise ------------
    noise_scheduler = GaussianNoiseScheduler
    noise_scheduler_kwargs = {
        'timesteps': 1000,
        'beta_start': 0.002, # 0.0001, 0.0015
        'beta_end': 0.02, # 0.01, 0.0195
        'schedule_strategy': 'scaled_linear'
    }
    
    # ------------ Initialize Latent Space  ------------
    # latent_embedder = None 
    # latent_embedder = VQVAE
    latent_embedder = VAE
    
    # latent_embedder_checkpoint = '/projects/NEI/pranay/Eyes/medfusion/runs/2024_06_05_111819/epoch=174-step=699.ckpt'
    latent_embedder_checkpoint = '/projects/NEI/pranay/Eyes/medfusion/runs/2024_07_19_150326/epoch=136-step=41049.ckpt'
   
    # ------------ Initialize Pipeline ------------
    pipeline = DiffusionPipeline(
        noise_estimator=noise_estimator, 
        noise_estimator_kwargs=noise_estimator_kwargs,
        noise_scheduler=noise_scheduler, 
        noise_scheduler_kwargs = noise_scheduler_kwargs,
        latent_embedder=latent_embedder,
        latent_embedder_checkpoint = latent_embedder_checkpoint,
        estimator_objective='x_T',
        estimate_variance=False, 
        use_self_conditioning=False, 
        use_ema=False,
        classifier_free_guidance_dropout=0.5, # Disable during training by setting to 0
        do_input_centering=False,
        clip_x0=False,
        sample_every_n_steps=1000
    )
    
    # pipeline_old = pipeline.load_from_checkpoint('runs/2022_11_27_085654_chest_diffusion/last.ckpt')
    # pipeline.noise_estimator.load_state_dict(pipeline_old.noise_estimator.state_dict(), strict=True)

    # -------------- Training Initialization ---------------
    to_monitor = "val/loss"  # "pl/val_loss" 
    min_max = "min"
    save_and_sample_every = 100

    early_stopping = EarlyStopping(
        monitor=to_monitor,
        min_delta=0.0, # minimum change in the monitored quantity to qualify as an improvement
        patience=30, # number of checks with no improvement
        mode=min_max
    )
    # checkpointing = ModelCheckpoint(
    #     dirpath=str(path_run_dir), # dirpath
    #     monitor=to_monitor,
    #     every_n_train_steps=save_and_sample_every,
    #     save_last=True,
    #     save_top_k=10,
    #     mode=min_max,
    # )
    checkpointing = ModelCheckpoint(
        dirpath=str(path_run_dir),
        every_n_epochs=10,  # Save every epoch
        save_top_k=-1,
    filename='{epoch}'
)
    tensorboard_logger = TensorBoardLogger("tb_logs", name="train_diff_RFMID_preprocessed_1024_actually")

    # trainer = Trainer(
    #     accelerator=accelerator,
    #     # devices=[0],
    #     # precision=16,
    #     # amp_backend='apex',
    #     # amp_level='O2',
    #     # gradient_clip_val=0.5,
    #     default_root_dir=str(path_run_dir),
    #     callbacks=[checkpointing],
    #     # callbacks=[checkpointing, early_stopping],
    #     enable_checkpointing=True,
    #     check_val_every_n_epoch=1,
    #     log_every_n_steps=save_and_sample_every, 
    #     auto_lr_find=False,
    #     # limit_train_batches=1000,
    #     limit_val_batches=0, # 0 = disable validation - Note: Early Stopping no longer available 
    #     min_epochs=100,
    #     max_epochs=1001,
    #     num_sanity_val_steps=2,
    # )
    trainer = Trainer(
        logger=tensorboard_logger,
        accelerator=accelerator,
        devices='auto',  # Automatically use all available GPUs
        strategy='ddp',  # Use Distributed Data Parallel
        default_root_dir=str(path_run_dir),
        callbacks=[checkpointing],
        enable_checkpointing=True,
        check_val_every_n_epoch=10,
        log_every_n_steps=1,
        auto_lr_find=False,
        limit_val_batches=10, 
        min_epochs=100,
        max_epochs=2000,
        num_sanity_val_steps=2 
    )   
    
    # ---------------- Execute Training ----------------
    trainer.fit(pipeline, datamodule=dm)

    # ------------- Save path to best model -------------
    pipeline.save_best_checkpoint(trainer.logger.log_dir, checkpointing.best_model_path)


