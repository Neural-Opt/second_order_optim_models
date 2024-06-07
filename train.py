from config.loader import getOptim,getLRScheduler
from models.cifar import CIFAR
from models.demux import getBenchmarkSet
import torch
from utils.utils import AverageAggregator
from log.Logger import Logger
num_epochs = 25
best_val_acc = 0.0
def train(model, device, train_loader, optimizer, criterion,lr_scheduler):
    model.train()

    accuracy = AverageAggregator()
    avg_loss = AverageAggregator(measure=lambda loss:loss)
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        _, predicted = outputs.max(1)

        avg_loss(loss.item() * inputs.size(0),total=inputs.size(0))
        accuracy(predicted,targets,total=inputs.size(0))

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
    logger = Logger()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = getBenchmarkSet()
    train_loader ,test_loader , val_loader = dataset.getDataLoader()
    model  = dataset.getAssociatedModel()
    criterion = dataset.getAssociatedCriterion()
    _,optimizers,_ = getOptim(model,["AdaBelief","AdaHessian","AdamW","Apollo","RMSprop","SGD"])
    for optim in optimizers:
        lr_scheduler = getLRScheduler(optim)
        print(lr_scheduler)
        for epoch in range(num_epochs):
            train_loss, train_acc = train(model, device, train_loader, optim, criterion,lr_scheduler)
           # val_loss, val_acc = validate(model, device, val_loader, criterion)
           
            print(f'Epoch {epoch+1}/{num_epochs}')
            print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}')
            break
            print(f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}')
            
            # Save the best model based on validation accuracy
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), 'best_model.pth')


if __name__ == "__main__":
    main()