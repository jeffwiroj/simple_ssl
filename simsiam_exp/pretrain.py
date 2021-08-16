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
    parser.add_argument('--base_lr',default = 0.05,type = int, help = 'base learning rate')
    parser.add_argument('--wd',default = 0.0001,type = float, help = 'Weight Decay')

    parser.add_argument('--epochs',default = 200, type = int)

    args = vars(parser.parse_args())
    args["init_lr"] = (args["batch_size"]/256)*args["base_lr"]
    return args


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    cur_lr = args["init_lr"] * 0.5 * (1. + math.cos(math.pi * epoch / args["epochs"]))
    for param_group in optimizer.param_groups:
         param_group['lr'] = cur_lr
    return cur_lr



def save_checkpoint(model,optimizer, is_best, epoch,loss,filename='checkpoint.pth.tar'):
    torch.save({
            'epoch': epoch,
            'model_state_dict':model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss':loss
            }, f"results/checkpoint/{filename}")
    
    if is_best:
        best_file = 'model_best.pth.tar' if "cosine" not in filename else 'model_best_cosine.pth.tar'
        shutil.copyfile(f"results/checkpoint/{filename}", f"results/checkpoint/{best_file}")

        
        
        
'''
Trains SimSiam Model
Model: SimSiam Model
Criterion: Negative Cosine Similarity Loss defined in paper, see simsiam folder
'''
def train(model,criterion,optimizer,data_loader,writer,args):
    
    epochs,sched_type,init_lr = args["epochs"],args["sched_type"],args["init_lr"]
    
    filename = 'checkpoint.pth.tar' if sched_type is None else 'checkpoint_cosine.pth.tar'
    
    model.train()
    
    cur_lr = init_lr
    best_loss = 100
    start_epoch = 0
    for epoch in range(start_epoch, epochs):
        epoch_loss = 0
        is_best = False
        for img,_ in data_loader:
            x1,x2 = img

            x1 = x1.to(device)
            x2 = x2.to(device)

            out1,out2 = model(x1,x2)
            
            #z = out1[0].detach()
            
            
            loss = criterion(out1,out2)
            epoch_loss += (loss.item() / len(data_loader))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if(epoch_loss < best_loss): 
            is_best = True
            best_loss = epoch_loss
            
        save_checkpoint(model,optimizer, is_best, epoch,epoch_loss,filename=filename)
        
        if(sched_type is not None):
            cur_lr = adjust_learning_rate(optimizer,init_lr,epoch,args)
            writer.add_scalar('LR',cur_lr, epoch)
        
        print(f"Epoch: {epoch}, Loss: {epoch_loss}, Cur LR: {cur_lr}")
        writer.add_scalar('Loss',epoch_loss, epoch)
    

def main():

    config = get_config()
    print(f"Current Configuration: {config}")
    
    
    writer = SummaryWriter(log_dir = "results/log_dir")
    
    #Save Hyperparameter values:
    writer.add_text("Sched Type", str(config["sched_type"]))
    writer.add_text("WD", str(config["wd"]))
    dataset = get_dataset()
    ssl_loader =  DataLoader(dataset['unlabel_set'], batch_size=512,shuffle = True, pin_memory = True,num_workers = 4)
    model = SimSiam()
    model = model.to(device)
    criterion = SimSiamLoss()
    optimizer = optim.SGD(model.parameters(), lr = config["init_lr"],weight_decay = config["wd"],momentum=0.9)

    train(model,criterion,optimizer,ssl_loader,writer,config)
    writer.close()
    
if __name__ == "__main__":
    main()