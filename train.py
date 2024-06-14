from benchmark.benchmark import Benchmark
from config.loader import getOptim,getLRScheduler
from models.demux import getBenchmarkSet
import torch
import random
import numpy as np
from utils.utils import MeanAggregator
from log.Logger import Logger
from tqdm import tqdm

num_epochs = 25
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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



def main():
    logger = Logger("test")
   
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = getBenchmarkSet()
    train_loader ,test_loader , val_loader = dataset.getDataLoader()
    criterion = dataset.getAssociatedCriterion()
    names,optimizers,params = getOptim(["AdaBelief","AdaHessian","SGD","Apollo","AdamW","RMSprop"])#["AdaBelief","AdaHessian","Adam","AdamW","Apollo","RMSprop","SGD"]

    for optim_class, name in zip(optimizers, names):
        set_seed(404)
        model  = dataset.getAssociatedModel()
        model = model.to(device)
        optim = optim_class(model.parameters(),**params[name])

        logger.setup(optim=name)
        lr_scheduler = getLRScheduler(optim)
    
        with tqdm(total=num_epochs, desc='Training Progress', unit='epoch') as epoch_bar:
            for epoch in range(num_epochs):
               
                train_loss, train_acc = dataset.train(model, device, train_loader, optim, criterion, lr_scheduler)
                test_acc = dataset.test(model, device, test_loader, criterion)

                epoch_bar.set_postfix({'optim':f'{name}',
                                       'loss': f'{train_loss:.4f}',
                                        'train-accuracy': f'{train_acc:.4f}',
                                        'test-accuracy': f'{test_acc:.4f}',
                                        })
                epoch_bar.update(1)
        
        logger.trash()
    
    logger.plot(names=names)
if __name__ == "__main__":
    set_seed(404)
    main()