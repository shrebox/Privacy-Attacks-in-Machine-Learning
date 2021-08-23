from torch.utils.data import Dataset
import torch
import PIL

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class UTKFace(Dataset):
    """UTK Face dataset."""

    def __init__(self, samples, label, transform=None):
        """
            samples (df): df containing the path and labels of each sample (columns=['filename','gender','race'])
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.samples = samples
        self.label = label
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # img_name = self.samples.iloc[idx, 0]
        # image = PIL.Image.open(img_name)
        image_array = self.samples.iloc[idx, 3]
        image = PIL.Image.fromarray(image_array)
        
        if self.transform:
            image = self.transform(image)
        
        if self.label =='gender':
            label = int(self.samples.iloc[idx, 1])
            sample =  {'image': image, 'gender': label}
            
        if self.label == 'race':
            label = int(self.samples.iloc[idx, 2])
            sample =  {'image': image, 'race': label}
        
        return sample

class AttackData(Dataset):
    """UTK Face dataset."""

    def __init__(self, samples, target_model, transform=None):
        """
        Args:
            samples (df): df containing the path and labels of each sample (columns=['filename','gender','race'])
            taget_model (nn.Module): model that should be queryed to get the posteriors
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        
        self.samples = samples
        self.target_model = target_model
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # img_name = self.samples.iloc[idx, 0]
        # image = PIL.Image.open(img_name)
        image_array = self.samples.iloc[idx, 3]
        image = PIL.Image.fromarray(image_array)
        
        if self.transform:
            image = self.transform(image)
                
        _, z = self.target_model(image.unsqueeze(0))
        
        label = int(self.samples.iloc[idx, 2])
        sample =  {'z': z, 'race': label}
        
        return sample