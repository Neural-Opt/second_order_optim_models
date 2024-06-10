from benchmark.benchmark import Benchmark
from config.loader import getOptim,getLRScheduler
from models.cifar import CIFAR
from models.demux import getBenchmarkSet
import torch
from utils.utils import MeanAggregator
from log.Logger import Logger
from tqdm import tqdm

num_epochs = 4
best_val_acc = 0.0
def train(model, device, train_loader, optimizer, criterion,lr_scheduler):
    model.train()
    benchmark = Benchmark.getInstance(None)
    accuracy = MeanAggregator(measure=lambda *args:(args[0].eq(args[1]).sum().item() / args[1].size(0)))
    avg_loss = MeanAggregator()
    i = 0
    for inputs, targets in train_loader:
       # print(i)
        benchmark.measureGPUMemUsageStart()
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

    
        benchmark.stepStart()
        optimizer.step()
        benchmark.stepEnd()

        lr_scheduler.step()
        _, predicted = outputs.max(1)

        benchmark.measureGPUMemUsageEnd()

        avg_loss(loss.item() * inputs.size(0))
        accuracy(predicted,targets)
        
    benchmark.addTrainAcc(accuracy.get())
    benchmark.addTrainLoss(avg_loss.get())
    benchmark.flush()

    return avg_loss.get(), accuracy.get()

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

def test(model, device, test_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
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
    print(device)
    dataset = getBenchmarkSet()
    train_loader ,test_loader , val_loader = dataset.getDataLoader()
    criterion = dataset.getAssociatedCriterion()
    names,optimizers,params = getOptim(["AdaHessian","RMSprop"])

    for optim_class, name in zip(optimizers, names):
        model  = dataset.getAssociatedModel()
        model = model.to(device)
        optim = optim_class(model.parameters(),**params[name])

        logger.setup(optim=name)
        lr_scheduler = getLRScheduler(optim)
    
        with tqdm(total=num_epochs, desc='Training Progress', unit='epoch') as epoch_bar:
            for epoch in range(num_epochs):
                train_loss, train_acc = train(model, device, train_loader, optim, criterion, lr_scheduler)
                    
                epoch_bar.set_postfix({'optim':f'{name}','loss': f'{train_loss:.4f}', 'accuracy': f'{train_acc:.4f}'})
                epoch_bar.update(1)
        
        logger.trash()
    
    logger.plot(names=names)
if __name__ == "__main__":
    main()