import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torchvision.models as models
from dataset import get_dataset
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_model():
    
    model = models.resnet34(False)
    model.fc = nn.Linear(512,9)
    return model
    
def main():
    
    dataset = get_dataset()
    train_loader =  DataLoader(dataset['train_set'], batch_size=512,shuffle = True, pin_memory = True,num_workers = 4)
    val_loader =  DataLoader(dataset['val_set'], batch_size=512,shuffle = False, pin_memory = True,num_workers = 4)



    # Sweep through Learning Rate and L2 weight Decay
    config = {'lr': [1e-5,1e-4,5e-3,1e-3,5e-2,1e-2], 'wd' :[0,1e-6,1e-5,1e-4]}
    
    
    best_acc = 0
    
    for i in range(len(config['lr'])):
        for j in range(len(config['wd'])):
            cur_lr = config['lr'][i]
            cur_wd = config['wd'][j]
            model = get_model()
            model = model.to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optimizer = optim.SGD(model.parameters(), lr= cur_lr,weight_decay = cur_wd,momentum=0.9)
            
            acc,loss = train_n_val(model,optimizer,criterion,train_loader,val_loader,25)
            
            print(f"Current Config: LR = {cur_lr}  WD = {cur_wd} Acc:{acc} Loss: {loss} ")
            

            
            
def train_n_val(model,optimizer,criterion,train_loader,val_loader,epochs):
    
    best_loss,best_acc = 100,0
    
    for epoch in range(epochs):
        
        model.train()
        for x,y in train_loader:
            B = y.size(0)
            optimizer.zero_grad()
            
            x = x.to(device)
            y = y.to(device)
            y = y.view(B).long()
            
            logits = model(x)
            preds = torch.argmax(logits,1)
            
            loss = criterion(logits,y)
            loss.backward()
            optimizer.step()
            
        epoch_loss = 0.0
        epoch_acc = 0.0
        
        model.eval()
        total,correct = 0,0
        with torch.no_grad():
            for x,y in val_loader:
                B = y.size(0)
                
                x = x.to(device)
                y = y.to(device)
                y = y.view(B).long()
                
                logits = model(x)
                preds = torch.argmax(logits,1)

                loss = criterion(logits,y)
                epoch_loss += (loss.item()/len(val_loader))
                total += y.size(0)
                correct += (preds == y).sum().item()
        epoch_acc = correct/total
        if(epoch_acc > best_acc):best_acc = epoch_acc
        if(epoch_loss < best_loss):best_loss = epoch_loss

        
    return best_acc,best_loss

if __name__ == "__main__":
    main()