import pandas as pd
import numpy as np
import shutil
import os
from PIL import Image
import random


def move_files(source_folder, destination_folder, num_files_per_folder):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    for root, dirs, files in os.walk(source_folder):
        for dir_name in dirs:
            full_dir_path = os.path.join(root, dir_name)
            full_dest_dir_path = os.path.join(
                destination_folder, os.path.relpath(full_dir_path, source_folder)
            )

            if not os.path.exists(full_dest_dir_path):
                os.makedirs(full_dest_dir_path)

            all_files = [
                f
                for f in os.listdir(full_dir_path)
                if os.path.isfile(os.path.join(full_dir_path, f))
            ]
            selected_files = random.sample(
                all_files, min(num_files_per_folder, len(all_files))
            )

            for file_name in selected_files:
                src_file_path = os.path.join(full_dir_path, file_name)
                dest_file_path = os.path.join(
                    full_dest_dir_path, file_name[:-4] + "png"
                )
                with Image.open(src_file_path) as img:
                    img.convert("RGB").save(dest_file_path, "PNG")


# Usage example
source_folder = "/projects/NEI/pranay/Eyes/Datasets/OCT/zipped_data/OCT_Test_512"
destination_folder = "/projects/NEI/pranay/Eyes/Datasets/Final_experiment_OCT/Original"
num_files_per_folder = 50  # Number of files to move from each subfolder

move_files(source_folder, destination_folder, num_files_per_folder)
