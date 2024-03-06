import pandas as pd
import os

def clean_csv_by_file_existence(csv_file, directory, output_csv):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_file)

    # Iterate over the DataFrame and check if the file exists
    for index, row in df.iterrows():
        # Construct the full file path
        file_path = os.path.join(directory, str(row.iloc[0])+ ".png")

        # Check if the file does not exist
        if not os.path.isfile(file_path):
            # Drop the row if the file does not exist
            df.drop(index, inplace=True)
            print(f"Fiel removed{file_path}")

    # Save the cleaned DataFrame to a new CSV file
    df.to_csv(output_csv, index=False)
    print(f"Cleaned CSV saved as {output_csv}")

# Example usage
csv_file = '/projects/NEI/pranay/Eyes/Datasets/A. RFMiD_All_Classes_Dataset/2. Groundtruths/a. RFMiD_Training_Labels.csv'  # Replace with your input CSV file path
directory = '/projects/NEI/pranay/Eyes/Datasets/A. RFMiD_All_Classes_Dataset/1. Original Images/a. Training Set'  # Replace with the directory to check files in
output_csv = '/projects/NEI/pranay/Eyes/Datasets/A. RFMiD_All_Classes_Dataset/2. Groundtruths/a. RFMiD_Training_Labels_mod.csv'  # Replace with your desired output CSV file path
clean_csv_by_file_existence(csv_file, directory, output_csv)