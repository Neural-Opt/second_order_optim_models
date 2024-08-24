import os
from benchmark.benchmark import Benchmark
from benchmark.state import BenchmarkState
from config.loader import getConfig, getOptim, getLRScheduler 
from models.demux import getBenchmarkSet
import torch
import random
import numpy as np
from utils.utils import MeanAggregator
from log.Logger import Logger, createNewRun
import torch.distributed as dist
import torch.multiprocessing as mp

from tqdm import tqdm

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ddp_setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
def ddp_cleanup():
    dist.destroy_process_group()

def validate(model, device, val_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc



def main(device:int,base_path:str,world_size:int,num_epochs:int = 25):
   # ddp_setup(rank,world_size)
    seed = 606
    set_seed(seed)
    print(device)
    logger = Logger(base_path=base_path,rank=device,world_size=world_size)
   
    dataset = getBenchmarkSet()
    train_loader ,test_loader,len_train = dataset.getDataLoader()
    criterion = dataset.getAssociatedCriterion()
    names,optimizers,params = getOptim(["AdaBelief","AdaHessian","AdamW","Apollo","ApolloW","RMSprop","SGD"])#["AdaBelief","AdaHessian","Adam","AdamW","Apollo","ApolloW","RMSprop","SGD"]

    for optim_class, name in zip(optimizers, names):
      
        set_seed(seed)
      
        model  = dataset.getAssociatedModel(device)
       
        optim = optim_class(model.parameters(),**params[name])
      #  dataset.setup(optim)
       # dataset.setup(optim)

# Adam optimizer parameters
       # optim = torch.optim.AdamW(model.parameters(), lr=5e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)

# Learning rate scheduler
      #  dataset.init(optim)
        logger.setup(optim=name)
        lr_scheduler = getLRScheduler(optim)
        if device == 0 or device == "cuda":
            epoch_bar = tqdm(total=num_epochs, desc='Training Progress', unit='epoch')

        for epoch in range(num_epochs):
            train_loss, train_acc = dataset.train(model, device, train_loader, optim, criterion,name=="AdaHessian",lr_scheduler)

            #TODO replace train_set
            test_acc = dataset.test(model, device, test_loader, criterion)
            if device == 0 or device == "cuda":
                lr = dataset.lr_scheduler.optimizer.param_groups[0]['lr']

                epoch_bar.set_postfix({'optim': f'{name}',
                                  'lr':f'{lr}',
                                  'loss': f'{train_loss:.4f}',
                                  'train-accuracy': f'{train_acc:.4f}',
                                  'test-accuracy': f'{test_acc:.4f}'})
                                  
                epoch_bar.update(1)
           # lr_scheduler.stepEpoch()
            
        if device == 0 or device == "cuda":
            epoch_bar.close()
        logger.trash()
        logger.save(optim=name)
    if device == 0 or device == "cuda":
        logger.plot(names=names)
    if device != "cuda":
        ddp_cleanup()

if __name__ == "__main__":
    conf = getConfig()
    world_size = torch.cuda.device_count()
    base_path = createNewRun(f"{conf['runs']['dir']}/{conf['runs']['name']}")
    num_epochs = conf['runs']['epochs']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    main(device,base_path,world_size,num_epochs)
  