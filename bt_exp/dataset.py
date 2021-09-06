import torch
import torchvision.transforms as T_transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import random

class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img

class BT_Transform():
    def __init__(self, image_size = 28, mean=None,std=None):
        
        
        self.transform = T_transforms.Compose([
            T_transforms.ToPILImage(),
            T_transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            T_transforms.RandomHorizontalFlip(),
            T_transforms.RandomApply([T_transforms.ColorJitter(0.4,0.4,0.2,0.1)], p=0.8),
            T_transforms.RandomGrayscale(p=0.2),
            Solarization(p=0.2),
            T_transforms.RandomApply([T_transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.2),
            T_transforms.ToTensor(),
            T_transforms.Normalize(mean = mean, std = std)
        ])
    def __call__(self, x):
        x1 = self.transform(x)
        x2 = self.transform(x)
        return x1, x2 
    
class pathDataset(Dataset):
    
    """Dataset for PathMnist"""
    
    def __init__(self,root_dir = "data", split = "train", transform=None):
        """
        Args:
           
            root_dir (string): Directory with all the images.
            split: train,val,test or ssl
            transform (callable, optional): Optional transform to be applied
                on a sample.
                
        """
        
        self.root_dir = root_dir
        self.images = np.load(f"{root_dir}/{split}_images.npy")
        self.labels =  np.load(f"{root_dir}/{split}_labels.npy")
        self.transform = transform
        self.split = split
        self.mean,self.std  = [0.7405, 0.5330, 0.7058],[0.1237, 0.1768, 0.1244]
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()

        images = self.images[idx]
        labels = self.labels[idx]
        
        
        
        images = self.transform(images)
        return images,labels
        
        
    

class DatasetFromSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)
    
    
    
def get_dataset(train_percent = 0.1,root_dir = '../data'):
    
    mean,std = [0.7405, 0.5330, 0.7058],[0.1237, 0.1768, 0.1244]
    train_transform = T_transforms.Compose([
                     T_transforms.ToPILImage(),
                     T_transforms.RandomHorizontalFlip(p=0.5),
                     T_transforms.RandomVerticalFlip(p=0.5),
                     T_transforms.RandomRotation(20),
                     T_transforms.ToTensor(),
                     T_transforms.Normalize(mean=mean,std=std)
     ])
    ssl_transform = BT_Transform(mean = mean,std = std)
    val_transform = T_transforms.Compose([
                     T_transforms.ToPILImage(),
                     T_transforms.ToTensor(),
                     T_transforms.Normalize(mean=mean,std=std)])
    
    training_set = pathDataset(root_dir = root_dir,transform = train_transform)
    val_set,test_set = pathDataset(root_dir = root_dir,split= "val", transform = val_transform), \
    pathDataset(root_dir = root_dir , split= "test",transform = val_transform)
    
    train_percent,n = 0.1,len(training_set)
    train_size,unlabel_size = int(n*train_percent),n - int(n*train_percent)
    train_set,unlabel_set = torch.utils.data.random_split(training_set,[train_size,unlabel_size],
                                                          generator=torch.Generator().manual_seed(1598))
    
    unlabel_set = DatasetFromSubset(unlabel_set,transform =  BT_Transform(mean = [0.7405, 0.5330, 0.7058],std = [0.1237, 0.1768, 0.1244]))
    
    return {'train_set': train_set,'val_set': val_set, 'test_set': test_set,'unlabel_set':unlabel_set}

if __name__ == "__main__":
    
    dataset = get_dataset()