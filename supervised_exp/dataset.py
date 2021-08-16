import torch
import torchvision.transforms as T_transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np

#Adapted from: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
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
        
        
        if(self.split != "ssl"):
            if(self.transform == None):self.transform = T_transforms.ToTensor()
            images = self.transform(images)
            return images,labels
        
        return None
    

    
    
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
    
    return {'train_set': train_set, 'val_set': val_set, 'test_set': test_set}
    

if __name__ == "__main__":
    
    dataset = get_dataset()