Medfusion - Medical Denoising Diffusion Probabilistic Model 
=============

Paper
=======
Please see: [**Diffusion Probabilistic Models beat GANs on Medical 2D Images**](https://arxiv.org/abs/2212.07501)

![](media/Medfusion.png)
*Figure: Medfusion*

![](media/animation_eye.gif) ![](media/animation_histo.gif) ![](media/animation_chest.gif)\
*Figure: Eye fundus, chest X-ray and colon histology images generated with Medfusion (Warning color quality limited by .gif)*

Demo
=============
[Link](https://huggingface.co/spaces/mueller-franzes/medfusion-app) to streamlit app.

Install
=============

Create virtual environment and install packages: \
`python -m venv venv` \
`source venv/bin/activate`\
`pip install -e .`


Get Started 
=============

1 Prepare Data
-------------

* Go to [medical_diffusion/data/datasets/dataset_simple_2d.py](medical_diffusion/data/datasets/dataset_simple_2d.py) and create a new `SimpleDataset2D` or write your own Dataset. 


2 Train Autoencoder 
----------------
* Go to [scripts/train_latent_embedder_2d.py](scripts/train_latent_embedder_2d.py) and import your Dataset. 
* Load your dataset with eg. `SimpleDataModule` 
* Customize `VAE` to your needs 
* (Optional): Train a `VAEGAN` instead or load a pre-trained `VAE` and set `start_gan_train_step=-1` to start training of GAN immediately.

2.1 Evaluate Autoencoder 
----------------
* Use [scripts/evaluate_latent_embedder.py](scripts/evaluate_latent_embedder.py) to evaluate the performance of the Autoencoder. 

3 Train Diffusion 
----------------
* Go to [scripts/train_diffusion.py](scripts/train_diffusion.py) and import/load your Dataset as before.
* Load your pre-trained VAE or VAEGAN with `latent_embedder_checkpoint=...` 
* Use `cond_embedder = LabelEmbedder` for conditional training, otherwise  `cond_embedder = None`  

3.1 Evaluate Diffusion 
----------------
* Go to [scripts/sample.py](scripts/sample.py) to sample a test image.
* Go to [scripts/helpers/sample_dataset.py](scripts/helpers/sample_dataset.py) to sample a more reprensative sample size.
* Use [scripts/evaluate_images.py](scripts/evaluate_images.py) to evaluate performance of sample (FID, Precision, Recall)

Acknowledgment 
=============
* Code builds upon https://github.com/lucidrains/denoising-diffusion-pytorch 




Instructions to run everything on the cluster:
=============

Can skip set up steps if wokring wiht files on (/projects/NEI/pranay/Eyes/medfusion)
1) Clone the repository
2) Follow above instructions to set up virtual environment 

Training steps

3) Train latent embedder
    1) Make sure everything looks right in the train_latent_embedder.py script (Can change name of training in line 158 so you can find it easily when looking at tensorbaord)
    2) Make sure the data being pointed to is correct (/projects/NEI/pranay/Eyes/Datasets/A. RFMiD_All_Classes_Dataset/1. Original Images Processed 5)
    3) Run sbatch train_vae.sh
    4) Can monitor all trainings using tensorboard
        1) cd to medfusion repo
        2) source venv/bin/activate
        3) tensorboard --logdir tb_logs 
    5) Training is hard to tell when converged (Seems liek visual inspection is pretty much the best avenue (/projects/NEI/pranay/Eyes/medfusion/tb_logs))
    6) Once satisfied with training can train difffusion model

4) Train diffusion model 
    1) Make sure the train_diffusion.py fiel is correct
        1) Need to specify the vae model that was just trained, or the one you want to use. Not ideal set up but have to find model in :/projects/NEI/pranay/Eyes/medfusion/runs
        The model folder will correspond with the datetime of when you stared training, which you can get from tesnorboard 
        2) Can check everyhting in the file to make sure its right, can change line 166 to rename model in tesnorbaord
    2) Run sbatch train.sh
    3) Can monitor in tensorboard like before
    4) Training convergence is easier to tell here, when the validation loss starts increasing

5) Sampling images
    1) Make sure everythign is set up correctly in the sample.py file
        1) Choose the diffusion model that you just trained
            1) Models I have trained in the past that you can use:
            /projects/NEI/pranay/Eyes/medfusion/runs/2024_03_06_171323/epoch=233-step=9099.ckpt
            /projects/NEI/pranay/Eyes/medfusion/runs/2024_03_04_113129/epoch=205-step=7999.ckpt
        2) Choose output path for saving images and quantity of images you woudl like to save
    2) Run sbatch sample.sh



