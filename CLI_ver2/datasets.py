from torch.utils.data import Dataset
import torch
from PIL import Image 
from typing import  Callable, Optional,Any
import os
import numpy as np


class UTKFace(Dataset):
    """
    Custom pytorch dataset class for loding utk face images along with labels : age, gender  and race

    :param root_dir: root folder containing the raw images
    :param transform: torchvision `transforms.Compose` object containing transformation to be applied to images
    """
    def __init__(
        self, 
        root_dir: str, 
        filename: str,
        transform:Optional[Callable] = None, 
        ) -> None:
        super(UTKFace, self).__init__()

        self.data_dir = root_dir
        self.file_name = filename
        self.transform = transform

        # Loaad the data
        self.data, self.gender, self.race = self.__loadutk__()

    def __len__(self):
        """
        :return: The total number of samples in the dataset
        """
        return len(self.data)
    
    def __loadutk__(self, label='gender', attr='race'):
        with np.load(os.path.join(self.data_dir, self.file_name)) as f:
            X = f['imgs']
            y = f[label + 's']
            a = f[attr + 's']
            return X, y, a

    def __getitem__(self, idx):
        """
        selects a sample and returns it in the right format as model input
        
        Args:
        idx (int) : representing a sample index in the whole dataset

        Returns:
         tuple : (image, age, gender, race) where he sample image at position idx, as pytorch tensor, 
                 and corresponding labels age,gender,race
        """
        image = self.data[idx]
        # to return a PIL Image
        image = Image.fromarray(image)

        if self.transform is not None:
            image = self.transform(image).float()
        
        label_gender = torch.from_numpy(np.asarray(self.gender[idx])).type(torch.LongTensor)
        label_race = torch.from_numpy(np.asarray(self.race[idx])).type(torch.LongTensor)
        
        return image, label_gender, label_race
        