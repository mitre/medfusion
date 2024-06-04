from pathlib import Path
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms
from pytorch_msssim import ssim
import shutil
from tqdm import tqdm


def load_and_resize_image(image_path, size=(1024, 1024)):
    """
    Load and resize an image using PyTorch, leveraging GPU if available.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = Image.open(image_path).convert("L")  # Convert to grayscale
    resize_transform = transforms.Compose(
        [
            transforms.Resize(size),
            transforms.ToTensor(),
        ]
    )
    image = resize_transform(image).unsqueeze(0).to(device)
    return image


def mse(image1, image2):
    """
    Compute the Mean Squared Error between two images.
    Assumes the images are PyTorch tensors of the same shape.
    """
    return torch.mean((image1 - image2) ** 2)


def compare_images(img1, img2):
    """
    Compare two images using Mean Squared Error.
    This function assumes that img1 and img2 are PyTorch tensors.
    """
    return mse(img1, img2).item()


def find_most_similar_pairs(directory1, directory2, output_directory, top_n=10):
    """
    Find the most similar image pairs between two directories and save the top N pairs.
    """
    images1 = [
        (path, load_and_resize_image(path))
        for path in tqdm(
            list(Path(directory1).rglob("*.[jp][pn]g")),
            desc="Loading images from directory 1",
        )
    ]
    images2 = [
        (path, load_and_resize_image(path))
        for path in tqdm(
            list(Path(directory2).rglob("*.[jp][pn]g")),
            desc="Loading images from directory 2",
        )
    ]

    similarity_scores = []

    total_comparisons = len(images1) * len(images2)
    with tqdm(total=total_comparisons, desc="Comparing Images") as pbar:
        for path1, img1 in images1:
            for path2, img2 in images2:
                score = compare_images(img1, img2)
                similarity_scores.append((score, path1, path2))
                pbar.update(1)

    # Sort by score in descending order
    similarity_scores.sort(reverse=True, key=lambda x: x[0])

    # Prepare output directory
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save top N pairs and prepare data for CSV
    csv_data = []
    for i, (score, path1, path2) in enumerate(similarity_scores[:top_n]):
        pair_folder = output_path / f"pair_{i+1}"
        pair_folder.mkdir(exist_ok=True)
        shutil.copy(path1, pair_folder / path1.name)
        shutil.copy(path2, pair_folder / path2.name)

        csv_data.append(
            {
                "Pair ID": i + 1,
                "Image 1": str(pair_folder / path1.name),
                "Image 2": str(pair_folder / path2.name),
                "SSIM Score": score,
            }
        )

    # Save CSV
    pd.DataFrame(csv_data).to_csv(output_path / "similar_image_pairs.csv", index=False)


# Example usage
directory1 = (
    "/projects/NEI/pranay/Eyes/Datasets/Diff_Generated_experiment_fundus_1024_cond"
)
directory2 = "/projects/NEI/pranay/Eyes/Datasets/A. RFMiD_All_Classes_Dataset/1. Original Images Processed 5/a. Training Set"
output_directory = "/projects/NEI/pranay/Eyes/Datasets/fundus_mse"

find_most_similar_pairs(directory1, directory2, output_directory)
