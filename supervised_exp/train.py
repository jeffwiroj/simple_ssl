import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torchvision.models as models
from dataset import get_dataset
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import argparse
from torch.utils.tensorboard import SummaryWriter
import sys
sys.path.append('../')
import utils
import math
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def get_config():
    parser = argparse.ArgumentParser(description='Supervised Training')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
    parser.add_argument('--sched',default = None,type = str, help = 'Scheduler Type: None or Cosine')
    parser.add_argument('--lr',default = 0.05,type = float, help = 'Learning Rate')
    parser.add_argument('--wd',default = 1e-05,type = float, help = 'Weight Decay')
    
    parser.add_argument('--epochs',default = 200, type = int)
    
    return parser.parse_args()


def get_model():
    
    model = models.resnet34(False)
    model.fc = nn.Linear(512,9)
    return model




def val(model,criterion,val_loader):
    
    total_acc,total_loss = 0,0
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
            total_loss += (loss.item()/len(val_loader))
            total += y.size(0)
            correct += (preds == y).sum().item()
            
    total_acc = correct/total
    
    return total_acc,total_loss
    
def train_n_val(model,optimizer,criterion,train_loader,val_loader,writer,config):
    
    epochs,lr = config["epochs"],config["lr"]
    
    
    
    
    if(config["sched"] is not None):
        scheduler = utils.CosineScheduler(max_update = epochs,base_lr = 1.5*lr,final_lr = lr/100)
        
    
    best_acc,best_loss = 0,1000
    for epoch in range(epochs):
        train_acc,train_loss = 0,0
        total,correct = 0,0
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
            train_loss += (loss.item()/len(train_loader))
            total += y.size(0)
            correct += (preds == y).sum().item()
            
            loss.backward()
            optimizer.step()
        
        
        train_acc = (correct/total)
        
        
        if(config["sched"] is not None):
            for param_group in optimizer.param_groups:
                    current_lr = param_group['lr']
                    param_group['lr'] = scheduler(epoch)
                    
                    
        val_acc,val_loss = val(model,criterion,val_loader)
        
        if(val_acc > best_acc):
            print(f"Achieved New Best Acc: {val_acc}")
            best_acc = val_acc
            sched_type = str(config["sched"])
            torch.save(model.state_dict(), f"results/checkpoints/sched_type_{sched_type}.pth")
        best_loss = min(val_loss,best_loss)
        
        print(f"Epoch: {epoch}, Train Acc: {train_acc},  Train Loss: {train_loss}")
        print(f"Epoch: {epoch}, Val Acc: {val_acc},  Val Loss: {val_loss}, BEST ACC:{best_acc}")
        
        writer.add_scalar('Loss/train',train_loss, epoch)
        writer.add_scalar('Loss/val',val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        if(config["sched"] is not None):
            writer.add_scalar('Learning rate', current_lr, epoch)
        
    return best_acc,best_loss




def main():
    
    config = vars(get_config())
    print(f"Current Config: {config}")
    writer = SummaryWriter(log_dir = "results/log_dir")
    
    #Save Hyperparameter values:
    writer.add_text("Sched Type", str(config["sched"]))
    writer.add_text("LR", str(config["lr"]))
    writer.add_text("WD", str(config["wd"]))
    

    dataset = get_dataset()
    train_loader =  DataLoader(dataset['train_set'], batch_size=512,shuffle = True, pin_memory = True,num_workers = 4)
    val_loader =  DataLoader(dataset['val_set'], batch_size=512,shuffle = False, pin_memory = True,num_workers = 4)

    model = get_model()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr= config["lr"],weight_decay = config["wd"],momentum=0.9)
    train_n_val(model,optimizer,criterion,train_loader,val_loader,writer,config)
    writer.close()
    
    
if __name__ == "__main__":
    main()
