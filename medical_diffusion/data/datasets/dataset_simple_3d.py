
import os
import pydicom
import torch
import torch.utils.data as data 
from pathlib import Path 
from torchvision import transforms as T


import torchio as tio 

from medical_diffusion.data.augmentation.augmentations_3d import ImageToTensor


class SimpleDataset3D(data.Dataset):
    def __init__(
        self,
        path_root,
        item_pointers =[],
        crawler_ext = ['nii'], # other options are ['nii.gz'],
        transform = None,
        image_resize = None,
        flip = False,
        image_crop = None,
        use_znorm=True, # Use z-Norm for MRI as scale is arbitrary, otherwise scale intensity to [-1, 1]
    ):
        super().__init__()
        self.path_root = path_root
        self.crawler_ext = crawler_ext

        if transform is None: 
            self.transform = T.Compose([
                tio.Resize(image_resize) if image_resize is not None else tio.Lambda(lambda x: x),
                tio.RandomFlip((0,1,2)) if flip else tio.Lambda(lambda x: x),
                tio.CropOrPad(image_crop) if image_crop is not None else tio.Lambda(lambda x: x),
                tio.ZNormalization() if use_znorm else tio.RescaleIntensity((-1,1)),
                ImageToTensor() # [C, W, H, D] -> [C, D, H, W]
            ])
        else:
            self.transform = transform
        
        if len(item_pointers):
            self.item_pointers = item_pointers
        else:
            self.item_pointers = self.run_item_crawler(self.path_root, self.crawler_ext) 

    def __len__(self):
        return len(self.item_pointers)

    def __getitem__(self, index):
        rel_path_item = self.item_pointers[index]
        path_item = self.path_root/rel_path_item
        img = self.load_item(path_item)
        return {'uid':rel_path_item.stem, 'source': self.transform(img), 'target': 0}
    
    def load_item(self, path_item):
        return tio.ScalarImage(path_item) # Consider to use this or tio.ScalarLabel over SimpleITK (sitk.ReadImage(str(path_item)))
    
    @classmethod
    def run_item_crawler(cls, path_root, extension, **kwargs):
        return [path.relative_to(path_root) for path in Path(path_root).rglob(f'*.{extension}')]
    
    
    
class DicomDataset3D(SimpleDataset3D):
    def __init__(
        self,
        path_to_data,
        crawler_ext=['1.1.dcm'], # List of patterns to search for
        transform=None,
        image_resize=None,
        flip=False,
        image_crop=None,
        use_znorm=True,
    ):
        # Initialize with dummy values
        super().__init__(
            path_root=path_to_data,
            transform=transform,
            image_resize=image_resize,
            flip=flip,
            image_crop=image_crop,
            use_znorm=use_znorm,
        )
        # Override item_pointers with crawled DICOM files
        self.item_pointers = self.run_item_crawler(path_to_data, crawler_ext)
    def load_item(self, path_item):
        # Load DICOM files
        dicom_image = pydicom.dcmread(str(path_item)).pixel_array
        dicom_tensor = torch.from_numpy(dicom_image).unsqueeze(0).float()
        dicom_image_tio = tio.ScalarImage(tensor=dicom_tensor)
        return dicom_image_tio
    @classmethod
    def run_item_crawler(cls, path_to_data, dicom_file_patterns):
        # Crawl the directory for files matching the patterns
        list_of_dicom_files = []
        for root, dirs, files in os.walk(path_to_data):
            for file in files:
                if file.endswith(dicom_file_patterns[0]):
                    list_of_dicom_files.append(Path(root) / file)
        return list_of_dicom_files