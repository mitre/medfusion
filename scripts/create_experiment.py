import os
import random
import shutil
import pandas as pd


def sample_and_copy_files(
    base_folder, destination_folder, subfolder_mapping, folders_to_use, max_files=20
):
    # Create the destination folder if it doesn't exist
    os.makedirs(destination_folder, exist_ok=True)

    # List to store CSV data
    csv_data = []

    # Traverse through the subfolders
    for folder_name in os.listdir(base_folder):
        folder_path = os.path.join(base_folder, folder_name)

        if os.path.isdir(folder_path):

            for subfolder_name in os.listdir(folder_path):
                print(subfolder_name)
                subfolder_path = os.path.join(folder_path, subfolder_name)
                if os.path.isdir(subfolder_path) and subfolder_name in folders_to_use:
                    print(subfolder_path)
                    # Get all .png files
                    files = [
                        f for f in os.listdir(subfolder_path) if f.endswith(".png")
                    ]
                    # Randomly sample files
                    sampled_files = random.sample(files, min(max_files, len(files)))

                    for file in sampled_files:
                        original_file_path = os.path.join(subfolder_path, file)
                        random_file_name = f"{random.getrandbits(64)}.png"
                        new_file_path = os.path.join(
                            destination_folder, random_file_name
                        )

                        # Copy file with a new name
                        shutil.copy(original_file_path, new_file_path)

                        # Map original folder and subfolder to new file name
                        csv_data.append(
                            {
                                "New FileName": random_file_name,
                                "Original Folder": folder_name,
                                "Original Subfolder": subfolder_mapping.get(
                                    subfolder_name, subfolder_name
                                ),
                            }
                        )

    # Create a DataFrame and save as CSV
    df = pd.DataFrame(csv_data)
    df.to_csv(os.path.join(destination_folder, "file_mapping.csv"), index=False)


# Usage example
base_folder = "/projects/NEI/pranay/Eyes/Datasets/Final_experiment_fundus"
destination_folder = (
    "/projects/NEI/pranay/Eyes/Datasets/Final_experiment_fundus_randomized"
)
subfolder_mapping = {
    "0": "Normal",
    "1": "DR",
    "2": "MH",
    "3": "ODC",
    "4": "DN",
    "5": "BRVO",
    "6": "ODE",
}
folders_to_use = ["0", "1", "3", "4"]  # Your mapping here

sample_and_copy_files(
    base_folder, destination_folder, subfolder_mapping, folders_to_use
)
