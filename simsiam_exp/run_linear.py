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
from simsiam.model import SimSiam,SimSiamLoss
import shutil





def get_config():
    
    parser = argparse.ArgumentParser(description='Supervised Training')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
    parser.add_argument('--sched_type',default = None,type = str, help = 'Scheduler Type: None or Cosine')
    parser.add_argument('--batch_size',default = 512,type = int, help = 'Batch Size')
    parser.add_argument('--lr',default = 0.05,type = int, help = 'base learning rate')
    parser.add_argument('--wd',default = 0.0001,type = float, help = 'Weight Decay')
    parser.add_argument('--epochs',default = 200, type = int)
    args = vars(parser.parse_args())
    return args


#Loads weights using last epoch or best epoch
#Default = Last
def get_model(use_best = False):
   
    filename = "model_best.pth.tar" if use_best else "checkpoint.pth.tar"
    checkpoint  = torch.load(f"results/checkpoint/{filename}",map_location=device)
    
    simsiam_ = SimSiam()
    simsiam_.load_state_dict(checkpoint['model_state_dict'])
    backbone = simsiam_.backbone
    for param in backbone.parameters():
        param.require_grad = False
    model = nn.Sequential(backbone,nn.Flatten(),nn.Linear(512,9))

    return model
    

        
        
        
'''
Trains SimSiam Model
Model: SimSiam Model
Criterion: Negative Cosine Similarity Loss defined in paper, see simsiam folder
'''
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
    
    
    
    if(config["sched_type"] is not None):
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
        
        
        if(config["sched_type"] is not None):
            for param_group in optimizer.param_groups:
                    current_lr = param_group['lr']
                    param_group['lr'] = scheduler(epoch)
                    
                    
        val_acc,val_loss = val(model,criterion,val_loader)
        
        if(val_acc > best_acc):
            print(f"Achieved New Best Acc: {val_acc}")
            best_acc = val_acc
            sched_type = str(config["sched_type"])
            torch.save(model.state_dict(), f"results/linear/checkpoint/sched_type_{sched_type}.pth")
        best_loss = min(val_loss,best_loss)
        
        print(f"Epoch: {epoch}, Train Acc: {train_acc},  Train Loss: {train_loss}")
        print(f"Epoch: {epoch}, Val Acc: {val_acc},  Val Loss: {val_loss}, BEST ACC:{best_acc}")
        
        writer.add_scalar('Loss/train',train_loss, epoch)
        writer.add_scalar('Loss/val',val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        if(config["sched_type"] is not None):
            writer.add_scalar('Learning rate', current_lr, epoch)
        
    return best_acc,best_loss


def main():

    config = get_config()
    print(f"Current Configuration: {config}")
    model = get_model()
    model = model.to(device)
    
    writer = SummaryWriter(log_dir = "results/linear/log_dir")
    
    #Save Hyperparameter values:
    writer.add_text("Sched Type", str(config["sched_type"]))
    writer.add_text("LR", str(config["lr"]))
    writer.add_text("WD", str(config["wd"]))
    

    dataset = get_dataset()
    train_loader =  DataLoader(dataset['train_set'], batch_size=512,shuffle = True, pin_memory = True,num_workers = 4)
    val_loader =  DataLoader(dataset['val_set'], batch_size=512,shuffle = False, pin_memory = True,num_workers = 4)
    test_loader =  DataLoader(dataset['test_set'], batch_size=512,shuffle = False, pin_memory = True,num_workers = 4,drop_last = False)

    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr= config["lr"],weight_decay = config["wd"],momentum=0.9)
    train_n_val(model,optimizer,criterion,train_loader,val_loader,writer,config)
    test_acc,test_loss = val(model,criterion,test_loader)
    print(f"Test Accuracy: {test_acc}, Test Loss:{test_loss}")
    writer.add_text("Test_Acc", str(test_acc))
    writer.close()

    
    
if __name__ == "__main__":
    main()
    
     
