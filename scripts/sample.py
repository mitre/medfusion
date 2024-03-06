
from pathlib import Path
import torch 
from torchvision import utils 
import math 
from medical_diffusion.models.pipelines import DiffusionPipeline
import pandas as pd

def rgb2gray(img):
    # img [B, C, H, W]
    return  ((0.3 * img[:,0]) + (0.59 * img[:,1]) + (0.11 * img[:,2]))[:, None]
    # return  ((0.33 * img[:,0]) + (0.33 * img[:,1]) + (0.33 * img[:,2]))[:, None]

def normalize(img):
    # img =  torch.stack([b.clamp(torch.quantile(b, 0.001), torch.quantile(b, 0.999)) for b in img])
    return torch.stack([(b-b.min())/(b.max()-b.min()) for b in img])


def generate_and_save_images(batch_size, total_samples, condition_value, path_out, file_prefix, a):
    for i in range(0, total_samples, batch_size):
        # Adjust the actual batch size for the last batch
        actual_batch_size = min(batch_size, total_samples - i)

        condition = condition_value.repeat(actual_batch_size,1)
        results = pipeline.sample(actual_batch_size, (8, 128, 128), condition=condition,guidance_scale=1)
        # results = (results + 1) / 2  # Transform from [-1, 1] to [0, 1]
        # results = results.clamp(0, 1)

        # Save each image individually
        for j, image in enumerate(results):
            file_name = f"{a}/{file_prefix}_{i + j}.png"
            utils.save_image(image, path_out / file_name, normalize=True)


if __name__ == "__main__":
    path_out = Path('/projects/NEI/pranay/Eyes/Datasets/Diff_Generated_experiment_fundus_1024_cond')
    path_out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    device = torch.device('cuda')

    # ------------ Load Model ------------
    # pipeline = DiffusionPipeline.load_best_checkpoint(path_run_dir)
    pipeline = DiffusionPipeline.load_from_checkpoint('/projects/NEI/pranay/Eyes/medfusion/runs/2024_03_04_113129/epoch=205-step=7999.ckpt')
    pipeline.to(device)

    
    # --------- Generate Samples  -------------------
    steps = 150
    use_ddim = True 
    images = {}
    batch_size=10
    n_samples = int(10)
    item_pointers = pd.read_csv("/projects/NEI/pranay/Eyes/Datasets/A. RFMiD_All_Classes_Dataset/2. Groundtruths/a. RFMiD_Training_Labels_mod.csv")
    item_pointers['new_column'] = 1
    # unique_rows = item_pointers.iloc[:, 1:].drop_duplicates()
    unique_row_counts = item_pointers.iloc[:, 1:].groupby(item_pointers.columns[1:].tolist()).size().reset_index(name='counts')
    # Convert the counts to PyTorch tensors
    unique_row_counts = unique_row_counts.sort_values(by='counts', ascending=False).reset_index(drop=True)
    # unique_row_counts_tensor = [torch.tensor(row['counts']).to(device) for index, row in unique_row_counts.iterrows()]
    unique_row_counts.to_csv(path_out/"count.csv",index=False)

    
    for index,row in unique_row_counts.iterrows():
        cond = torch.tensor(row[:-1]).to(device)
        cond_path = path_out/str(index)
        cond_path.mkdir(parents=True, exist_ok=True)
        generate_and_save_images(batch_size, n_samples, cond, path_out, 'test', index)

 
        # # --------- Conditioning ---------
        # condition = torch.tensor([cond]*n_samples, device=device) if cond is not None else None 
        # # un_cond = torch.tensor([1-cond]*n_samples, device=device)
        # un_cond = None 

        # # ----------- Run --------
        # results = pipeline.sample(n_samples, (8, 32, 32), guidance_scale=8, condition=condition, un_cond=un_cond, steps=steps, use_ddim=use_ddim )
        # # results = pipeline.sample(n_samples, (4, 64, 64), guidance_scale=1, condition=condition, un_cond=un_cond, steps=steps, use_ddim=use_ddim )

        # # --------- Save result ---------------
        # results = (results+1)/2  # Transform from [-1, 1] to [0, 1]
        # results = results.clamp(0, 1)
        # utils.save_image(results, path_out/f'test_{cond}.png', nrow=int(math.sqrt(results.shape[0])), normalize=True, scale_each=True) # For 2D images: [B, C, H, W]
        # images[cond] = results


    # diff = torch.abs(normalize(rgb2gray(images[1]))-normalize(rgb2gray(images[0]))) # [0,1] -> [0, 1]
    # diff = torch.abs(images[1]-images[0])
    # utils.save_image(diff, path_out/'diff.png', nrow=int(math.sqrt(results.shape[0])), normalize=True, scale_each=True) # For 2D images: [B, C, H, W]
    

        
