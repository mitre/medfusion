
import torch.utils.data as data 
import torch 
from torch import nn
from pathlib import Path 
from torchvision import transforms as T
import pandas as pd 
import pydicom
import numpy as np
import os
from PIL import Image
from collections import Counter


from medical_diffusion.data.augmentation.augmentations_2d import Normalize, ToTensor16bit

from monai.transforms import LoadImage
from monai.data import Dataset, DataLoader
from tqdm import tqdm


class SimpleDataset2D(data.Dataset):
    def __init__(
        self,
        path_root,
        item_pointers =[],
        crawler_ext = 'jpeg', # other options are ['jpg', 'jpeg', 'png', 'tiff'],
        transform = None,
        image_resize = None,
        augment_horizontal_flip = False,
        augment_vertical_flip = False, 
        image_crop = None,
    ):
        super().__init__()
        self.path_root = Path(path_root)
        self.crawler_ext = crawler_ext
        if len(item_pointers):
            self.item_pointers = item_pointers
        else:
            self.item_pointers = self.run_item_crawler(self.path_root, self.crawler_ext) 

        if transform is None: 
            self.transform = T.Compose([
                T.CenterCrop(image_crop) if image_crop is not None else nn.Identity(),
                T.Resize(image_resize) if image_resize is not None else nn.Identity(),
                T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
                T.RandomVerticalFlip() if augment_vertical_flip else nn.Identity(),
                
                T.ToTensor(),
                # T.Lambda(lambda x: torch.cat([x]*3) if x.shape[0]==1 else x),
                # ToTensor16bit(),
                # Normalize(), # [0, 1.0]
                # T.ConvertImageDtype(torch.float),
                T.Normalize(mean=0.5, std=0.5) # WARNING: mean and std are not the target values but rather the values to subtract and divide by: [0, 1] -> [0-0.5, 1-0.5]/0.5 -> [-1, 1]
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.item_pointers)

    def __getitem__(self, index):
        rel_path_item = self.item_pointers[index]
        path_item = self.path_root/rel_path_item
        # img = Image.open(path_item) 
        img = self.load_item(path_item)
        return {'uid':rel_path_item.stem, 'source': self.transform(img)}
    
    def load_item(self, path_item):
        return Image.open(path_item).convert('RGB') 
        # return cv2.imread(str(path_item), cv2.IMREAD_UNCHANGED) # NOTE: Only CV2 supports 16bit RGB images 
    
    @classmethod
    def run_item_crawler(cls, path_root, extension, **kwargs):
        return [path.relative_to(path_root) for path in Path(path_root).rglob(f'*.{extension}')]

    def get_weights(self):
        """Return list of class-weights for WeightedSampling"""
        return None 


class AIROGSDataset(SimpleDataset2D):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.labels = pd.read_csv(self.path_root.parent/'train_labels.csv', index_col='challenge_id')
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        uid = self.labels.index[index]
        path_item = self.path_root/f'{uid}.jpg'
        img = self.load_item(path_item)
        str_2_int = {'NRG':0, 'RG':1} # RG = 3270, NRG = 98172 
        target = str_2_int[self.labels.loc[uid, 'class']]
        # return {'uid':uid, 'source': self.transform(img), 'target':target}
        return {'source': self.transform(img), 'target':target}
    
    def get_weights(self):
        n_samples = len(self)
        weight_per_class = 1/self.labels['class'].value_counts(normalize=True) # {'NRG': 1.03, 'RG': 31.02}
        weights = [0] * n_samples
        for index in range(n_samples):
            target = self.labels.iloc[index]['class']
            weights[index] = weight_per_class[target]
        return weights
    
    @classmethod
    def run_item_crawler(cls, path_root, extension, **kwargs):
        """Overwrite to speed up as paths are determined by .csv file anyway"""
        return []

class MSIvsMSS_Dataset(SimpleDataset2D):
    # https://doi.org/10.5281/zenodo.2530835
    def __getitem__(self, index):
        rel_path_item = self.item_pointers[index]
        path_item = self.path_root/rel_path_item
        img = self.load_item(path_item)
        uid = rel_path_item.stem
        str_2_int = {'MSIMUT':0, 'MSS':1}
        target = str_2_int[path_item.parent.name] #
        return {'uid':uid, 'source': self.transform(img), 'target':target}





class MSIvsMSS_2_Dataset(SimpleDataset2D):
    # https://doi.org/10.5281/zenodo.3832231
    def __getitem__(self, index):
        rel_path_item = self.item_pointers[index]
        path_item = self.path_root/rel_path_item
        img = self.load_item(path_item)
        uid = rel_path_item.stem
        str_2_int = {'MSIH':0, 'nonMSIH':1} # patients with MSI-H = MSIH; patients with MSI-L and MSS = NonMSIH)
        target = str_2_int[path_item.parent.name] 
        # return {'uid':uid, 'source': self.transform(img), 'target':target}
        return {'source': self.transform(img), 'target':target}

class OCT_2_Dataset(SimpleDataset2D):
    def __init__(self, path_root, item_pointers, crawler_ext='jpeg', transform=None, image_resize=None, augment_horizontal_flip=False, augment_vertical_flip=False, image_crop=None):
        image_resize = (1024,1024)
        super().__init__(path_root, item_pointers, crawler_ext, transform, image_resize, augment_horizontal_flip, augment_vertical_flip, image_crop)
    # https://doi.org/10.5281/zenodo.3832231
    def __getitem__(self, index):
        rel_path_item = self.item_pointers[index]
        path_item = self.path_root/rel_path_item
        img = self.load_item(path_item)
        uid = rel_path_item.stem
        str_2_int = {'NORMAL':0, 'CNV':1, 'DME':2, 'DRUSEN':3} # patients with MSI-H = MSIH; patients with MSI-L and MSS = NonMSIH)
        target = str_2_int[path_item.parent.name] 
        # return {'uid':uid, 'source': self.transform(img), 'target':target}
        return {'source': self.transform(img), 'target':target}

class TOPCON_Dataset2(SimpleDataset2D):
    def __init__(self, path_root, 
                 item_pointers, 
                 automoprh_results=None,
                 crawler_ext='jpg', 
                 transform=None, 
                 image_resize=(1024, 1024), 
                 augment_horizontal_flip=False, 
                 augment_vertical_flip=False, 
                 image_crop=(1864, 1864)):

        # Load the preprocessed CSV file
        self.item_pointers = pd.read_csv(item_pointers)
        self.automorph_results = automoprh_results
        
        if self.automorph_results is not None:
            self.automorph_results = pd.read_csv(self.automorph_results)
            # Filter item_pointers based on automorph_results and Interpretation column
            self.filter_item_pointers()
        
        # Calculate class-level weights
        self.class_weights = self.calculate_class_weights()
        
        # Calculate sample weights
        self.sample_weights = self.calculate_sample_weights()
        
        super().__init__(path_root, self.item_pointers, crawler_ext, transform, image_resize, augment_horizontal_flip, augment_vertical_flip, image_crop)
    
    def filter_item_pointers(self):
        # Assuming both DataFrames have a matching column, e.g., 'Absolute_File_Path'
        # Merge the two DataFrames on the matching column
        merged_df = pd.merge(self.item_pointers, self.automorph_results, on='Absolute_File_Path', suffixes=('_item', '_auto'))
        
        # Filter to remove class 5 and include only files with 'NW400' in their name
        str_2_int = {'NDRP': 0, 'MildNPDR': 1, 'ModerateNPDR': 2, 'SevereNPDR': 3, 'ProlDiaRet': 4}
        
        def interpretation_to_class(interpretation):
        # Check if the interpretation is a string before applying the mapping
            if isinstance(interpretation, str):
                for key in str_2_int:
                    if key in interpretation:
                        return str_2_int[key]
            return 5 

        # Apply filtering to exclude class 5 and keep files with 'NW400' in their name
        merged_df['class'] = merged_df['Interpretation'].apply(interpretation_to_class)
        filtered_df = merged_df[(merged_df['class'] != 5) & (merged_df['Absolute_File_Path'].str.contains('NW400'))]
        filtered_indices = (filtered_df['Prediction'].isin([0, 1])) | (filtered_df['Interpretation'] == 'ProlDiaRet')

        
        # Update the item_pointers DataFrame with the filtered data
        self.item_pointers = filtered_df.reset_index(drop=True)
        self.automorph_results = filtered_df.reset_index(drop=True)
    
    def calculate_class_weights(self):
        # Count the occurrences of each class based on the 'class' column generated earlier
        class_counts = Counter(self.item_pointers['class'])
        
        # Calculate total number of samples
        total_samples = sum(class_counts.values())
        
        # Calculate weights: inverse of the frequency of each class
        class_weights = {cls: total_samples / count for cls, count in class_counts.items()}
        
        # Print the calculated class weights (for debugging purposes)
        print(class_weights)
        
        return class_weights
    
    def calculate_sample_weights(self):
        sample_weights = []
        
        # Iterate over the rows of item_pointers and use the pre-calculated 'class' column
        for _, row in self.item_pointers.iterrows():
            target_class = row['class']  # Use the 'class' column that was generated
            
            # Assign the sample weight based on the class weight
            sample_weights.append(self.class_weights[target_class])
        
        # Convert the sample weights list to a torch tensor
        return torch.tensor(sample_weights, dtype=torch.float)
    
    def get_weights(self):
        return self.sample_weights
    
    def __getitem__(self, index):
        row = self.item_pointers.iloc[index]
        path_item = Path(row['Absolute_File_Path'])
        img = self.load_item(path_item)
        
        # Use the pre-calculated 'class' column for the target class
        target = row['class']
        
        # Determine the eye flag
        eye_flag = 0 if row['Eye'] == 'Left' else 1
        combined_target = torch.tensor(target, dtype=torch.long)
        
        return {'source': self.transform(img), 'target': combined_target}
    
class TOPCON_Dataset(SimpleDataset2D):
    def __init__(self, path_root, 
                item_pointers, 
                automoprh_results=None,
                crawler_ext='jpg', 
                transform=None, 
                image_resize=(1024,1024), 
                augment_horizontal_flip=False, 
                augment_vertical_flip=False, 
                image_crop=(1864,1864)):
        
        # Load the preprocessed CSV file
        self.item_pointers = pd.read_csv(item_pointers)
        self.automorph_results = automoprh_results
        
        if self.automorph_results is not None:
            self.automorph_results = pd.read_csv(self.automorph_results)
            # Filter item_pointers based on automorph_results and Interpretation column
            self.filter_item_pointers()
        
        # Calculate class-level weights
        self.class_weights = self.calculate_class_weights(self.item_pointers)
        
        # Calculate sample weights
        self.sample_weights = self.calculate_sample_weights()
        
        super().__init__(path_root, self.item_pointers, crawler_ext, transform, image_resize, augment_horizontal_flip, augment_vertical_flip, image_crop)
    
    def filter_item_pointers(self):
        # Assuming both DataFrames have a matching column, e.g., 'Absolute_File_Path'
        # Merge the two DataFrames on the matching column
        merged_df = pd.merge(self.item_pointers, self.automorph_results, on='Absolute_File_Path', suffixes=('_item', '_auto'))
        
        # Apply the filtering based on automorph_results' last column and Interpretation
        filtered_indices = (merged_df.iloc[:, -1].isin([0, 1])) | (merged_df['Interpretation'] == 'ProlDiaRet')
        
        # Filter the item_pointers DataFrame based on the filtered indices
        self.item_pointers = merged_df[filtered_indices].reset_index(drop=True)
        self.automorph_results = merged_df[filtered_indices].reset_index(drop=True)
    
    def calculate_class_weights(self, item_pointers):
        # Mapping for interpretation classes
        str_2_int = {'NDRP': 0, 'MildNPDR': 1, 'ModerateNPDR': 2, 'SevereNPDR': 3, 'ProlDiaRet': 4}
        
        # Count the occurrences of each class
        class_counts = Counter()
        for interpretation in item_pointers['Interpretation']:
            if pd.isna(interpretation):
                class_counts[5] += 1
            else:
                class_found = False
                for key in str_2_int:
                    if key in interpretation:
                        class_counts[str_2_int[key]] += 1
                        class_found = True
                        break
                if not class_found:
                    class_counts[5] += 1  # Default class if no match
        
        # Calculate weights: inverse of the frequency
        total_samples = sum(class_counts.values())
        class_weights = {cls: total_samples / count for cls, count in class_counts.items()}
        print(class_weights)
        
        return class_weights
    
    def calculate_sample_weights(self):
        sample_weights = []
        
        # Mapping for interpretation classes
        str_2_int = {'NDRP': 0, 'MildNPDR': 1, 'ModerateNPDR': 2, 'SevereNPDR': 3, 'ProlDiaRet': 4}
        
        for _, row in self.item_pointers.iterrows():
            interpretation = row['Interpretation']
            if pd.isna(interpretation):
                target = 5  # or any default value you prefer
            else:
                target = next((str_2_int[key] for key in str_2_int if key in interpretation), 5)
            
            # Assign the sample weight based on the class weight
            sample_weights.append(self.class_weights[target])
        
        return torch.tensor(sample_weights, dtype=torch.float)
    
    def get_weights(self):
        return self.sample_weights
    
    def __getitem__(self, index):
        row = self.item_pointers.iloc[index]
        path_item = Path(row['Absolute_File_Path'])
        img = self.load_item(path_item)
        
        # Mapping for interpretation classes
        str_2_int = {'NDRP': 0, 'MildNPDR': 1, 'ModerateNPDR': 2, 'SevereNPDR': 3, 'ProlDiaRet': 4}
        
        # Determine the target class from the Interpretation column
        interpretation = row['Interpretation']
        if pd.isna(interpretation):
            target = 5  # or any default value you prefer
        else:
            target = next((str_2_int[key] for key in str_2_int if key in interpretation), 5)
        
        # Determine the eye flag
        eye_flag = 0 if row['Eye'] == 'Left' else 1
        combined_target = torch.tensor(target, dtype=torch.long)
        
        return {'source': self.transform(img), 'target': combined_target}
   
    
class RFMID_Dataset(SimpleDataset2D):
    def __init__(self, path_root, item_pointers, crawler_ext='jpeg', transform=None, image_resize=None, augment_horizontal_flip=False, augment_vertical_flip=False, image_crop=None):
        item_pointers = pd.read_csv(item_pointers)
        # item_pointers['new_column'] = 1
        image_resize = (1024,1024)
        super().__init__(path_root, item_pointers, crawler_ext, transform, image_resize, augment_horizontal_flip, augment_vertical_flip, image_crop)
    def __getitem__(self, index):
        row = self.item_pointers.iloc[index]
        # print(len(row))
        rel_path_item = Path(str(row[0])+".png")
        binary_columns = row[1:]
        target = torch.tensor(binary_columns)
        path_item = self.path_root/rel_path_item
        img = self.load_item(path_item)
        uid = rel_path_item.stem
        return {'source': self.transform(img), 'target':target}
    
    
class IDRID_Dataset(SimpleDataset2D):
    def __init__(self, path_root, item_pointers, crawler_ext='jpeg', transform=None, image_resize=None, augment_horizontal_flip=False, augment_vertical_flip=False, image_crop=None):
        item_pointers = pd.read_csv(item_pointers)
        # item_pointers['new_column'] = 1
        image_resize = (1024,1024)
        super().__init__(path_root, item_pointers, crawler_ext, transform, image_resize, augment_horizontal_flip, augment_vertical_flip, image_crop)
    def __getitem__(self, index):
        row = self.item_pointers.iloc[index]
        # print(len(row))
        rel_path_item = Path(str(row[0])+".jpg")
        data_columns = row[2:3]
        target = torch.tensor(data_columns)
        path_item = self.path_root/rel_path_item
        img = self.load_item(path_item)
        uid = rel_path_item.stem
        return {'source': self.transform(img), 'target':target}

class CheXpert_Dataset(SimpleDataset2D):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        mode = self.path_root.name
        labels = pd.read_csv(self.path_root.parent/f'{mode}.csv', index_col='Path')
        self.labels = labels.loc[labels['Frontal/Lateral'] == 'Frontal'].copy()
        self.labels.index = self.labels.index.str[20:]
        self.labels.loc[self.labels['Sex'] == 'Unknown', 'Sex'] = 'Female' # Affects 1 case, must be "female" to match stats in publication
        self.labels.fillna(2, inplace=True) # TODO: Find better solution, 
        str_2_int = {'Sex': {'Male':0, 'Female':1}, 'Frontal/Lateral':{'Frontal':0, 'Lateral':1}, 'AP/PA':{'AP':0, 'PA':1}}
        self.labels.replace(str_2_int, inplace=True)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        rel_path_item = self.labels.index[index]
        path_item = self.path_root/rel_path_item
        img = self.load_item(path_item)
        uid = str(rel_path_item)
        target = torch.tensor(self.labels.loc[uid, 'Cardiomegaly']+1, dtype=torch.long)  # Note Labels are -1=uncertain, 0=negative, 1=positive, NA=not reported -> Map to [0, 2], NA=3
        return {'uid':uid, 'source': self.transform(img), 'target':target}

    
    @classmethod
    def run_item_crawler(cls, path_root, extension, **kwargs):
        """Overwrite to speed up as paths are determined by .csv file anyway"""
        return []

class CheXpert_2_Dataset(SimpleDataset2D):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        labels = pd.read_csv(self.path_root/'labels/cheXPert_label.csv', index_col=['Path', 'Image Index']) # Note: 1 and -1 (uncertain) cases count as positives (1), 0 and NA count as negatives (0)
        labels = labels.loc[labels['fold']=='train'].copy() 
        labels = labels.drop(labels='fold', axis=1)

        labels2 = pd.read_csv(self.path_root/'labels/train.csv', index_col='Path')
        labels2 = labels2.loc[labels2['Frontal/Lateral'] == 'Frontal'].copy()
        labels2 = labels2[['Cardiomegaly',]].copy()
        labels2[ (labels2 <0) | labels2.isna()] = 2 # 0 = Negative, 1 = Positive, 2 = Uncertain
        labels = labels.join(labels2['Cardiomegaly'], on=["Path",], rsuffix='_true')
        # labels = labels[labels['Cardiomegaly_true']!=2]

        self.labels = labels 
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        path_index, image_index = self.labels.index[index]
        path_item = self.path_root/'data'/f'{image_index:06}.png'
        img = self.load_item(path_item)
        uid = image_index
        target = int(self.labels.loc[(path_index, image_index), 'Cardiomegaly'])
        # return {'uid':uid, 'source': self.transform(img), 'target':target}
        return {'source': self.transform(img), 'target':target}
    
    @classmethod
    def run_item_crawler(cls, path_root, extension, **kwargs):
        """Overwrite to speed up as paths are determined by .csv file anyway"""
        return []
    
    def get_weights(self):
        n_samples = len(self)
        weight_per_class = 1/self.labels['Cardiomegaly'].value_counts(normalize=True)
        # weight_per_class = {2.0: 1.2, 1.0: 8.2, 0.0: 24.3}
        weights = [0] * n_samples
        for index in range(n_samples):
            target = self.labels.loc[self.labels.index[index], 'Cardiomegaly']
            weights[index] = weight_per_class[target]
        return weights
    
    
class DicomDataset(Dataset):
    def __init__(self, list_of_dicom_files):
        self.list_of_dicom_files = list_of_dicom_files
    def __len__(self):
        return len(self.list_of_dicom_files)
    def __getitem__(self, idx):
        dicom_file = self.list_of_dicom_files[idx]
        image = pydicom.dcmread(dicom_file).pixel_array
        image = torch.from_numpy(image)  # convert to PyTorch tensor
        image = image.unsqueeze(0)
        return {'source': image}