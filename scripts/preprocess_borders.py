import numpy as np
import cv2
import os
from PIL import Image
from tqdm import tqdm

def detect_centered_circle_and_crop(img):
    """
    Scans horizontally and vertically from the edges towards the center of the image to find the edges of the centered circle.
    Returns a bounding box for the circle.
    """
    img_array = np.array(img)
    height, width = img_array.shape[:2]
    center_x, center_y = width // 2, height // 2

    # Convert to grayscale for easier processing
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # Define a function to find the edge in a scan line
    def find_edge_from_outside(scan_line, start_idx, end_idx, step):
        if abs(int(scan_line[start_idx])) > 10:
            return start_idx

        initial_value = int(scan_line[start_idx])
        for i in range(start_idx, end_idx, step):
            if abs(int(scan_line[i]) - initial_value) > 10:  # Threshold can be adjusted
                return i
        return end_idx

    # Find the edges from the outside towards the center
    left_edge = find_edge_from_outside(gray[center_y, :], 0, center_x, 1)
    right_edge = find_edge_from_outside(gray[center_y,:], width -1, center_x, -1)
    top_edge = find_edge_from_outside(gray[:,center_x], 0, center_y, 1)
    bottom_edge = find_edge_from_outside(gray[:,center_x], height -1, center_y, -1)
    # print(left_edge)
    # print(right_edge)
    # print(top_edge)
    # print(bottom_edge)

    # Ensure the crop coordinates are within the image boundaries
    left = max(left_edge, 0)
    top = max(top_edge, 0)
    right = min(right_edge, width)
    bottom = min(bottom_edge, height)

    # Crop and return
    return img.crop((left, top, right, bottom))



def pad_to_square(img):
    """
    Pad img (PIL Image) to make it square.
    """
    width, height = img.size
    size = max(width, height)
    new_img = Image.new("RGB", (size, size))
    new_img.paste(img, ((size - width) // 2, (size - height) // 2))
    return new_img

def find_black_border(img):
    """
    Find the bounding box of the non-black area in the image.
    """
    # Convert image to numpy array
    img_array = np.array(img)

    # Create a binary mask where non-black pixels are True
    non_black_mask = np.all(img_array >= [5, 5, 5], axis=-1)

    # Find non-zero rows and columns
    non_black_rows_percentage = np.mean(non_black_mask, axis=1)
    non_black_cols_percentage = np.mean(non_black_mask, axis=0)

    # Find rows and columns where the percentage of non-black pixels is greater than the threshold
    non_black_rows = non_black_rows_percentage > 0.005
    non_black_cols = non_black_cols_percentage > 0.005
    non_black_row_idx = np.where(non_black_rows)[0]
    non_black_col_idx = np.where(non_black_cols)[0]

    # Get bounding box
    if non_black_row_idx.size > 0 and non_black_col_idx.size > 0:
        return non_black_row_idx[0], non_black_row_idx[-1], non_black_col_idx[0], non_black_col_idx[-1]
    else:
        return 0, 0, 0, 0

def process_directory(input_dir, output_dir, threshold=5):
    """
    Process all images in a directory and its subdirectories, cropping uniform color borders.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for subdir, dirs, files in os.walk(input_dir):
        for filename in tqdm(files):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(subdir, filename)
                img = Image.open(img_path)
                # top, bottom, left, right = find_black_border(img)
                # cropped_img = img.crop((left, top, right + 1, bottom + 1))
                # square_img = pad_to_square(cropped_img)
                crop = detect_centered_circle_and_crop(img)

                # Construct the output path by preserving the subdirectory structure
                relative_subdir = os.path.relpath(subdir, input_dir)
                output_subdir = os.path.join(output_dir, relative_subdir)
                if not os.path.exists(output_subdir):
                    os.makedirs(output_subdir)

                save_path = os.path.join(output_subdir, filename)
                crop.save(save_path)

# Example usage
input_directory = '/projects/NEI/pranay/Eyes/Datasets/topcon_screen_for_mitre'
output_directory = '/projects/NEI/pranay/Eyes/Datasets/topcon_screen_for_mitre_processed'
process_directory(input_directory, output_directory)
