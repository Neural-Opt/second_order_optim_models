from config.loader import getConfig
from models.benchmarkset import BenchmarkSet
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import torch.nn as nn


class CIFAR(BenchmarkSet):
    def __init__(self,batch_size=16,dataset="cifar10") -> None:
        super().__init__()
        self.conf = getConfig()
        self.batch_size = batch_size
        if dataset == 'cifar10':
            self.dataset = datasets.CIFAR10
            self.num_classes = 10
        else:
            self.dataset = datasets.CIFAR100
            self.num_classes = 100
    def log(self):
        pass
    def setup(self,):
        pass
    def getDataLoader(self,):       
        trainset = self.dataset(self.conf["dataset"]["path"], train=True, download=True,
                       transform=transforms.Compose([
                           transforms.RandomCrop(32, padding=4),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))#https://github.com/kuangliu/pytorch-cifar/issues/19
                       ]))
        testset = self.dataset(self.conf["dataset"]["path"], train=False, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))#https://github.com/kuangliu/pytorch-cifar/issues/19
                       ]))
      
        train_size = int(0.8 * len(trainset))
        val_size = len(trainset) - train_size
        train_dataset, val_dataset = random_split(trainset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=1)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=1)
        test_loader = DataLoader(testset, batch_size=self.batch_size, shuffle=False, num_workers=1)
        return (train_loader ,test_loader , val_loader)
    def getAssociatedModel(self):
        model = models.resnet18()
        model.fc = nn.Linear(model.fc.in_features, self.num_classes)
        return model
    def getAssociatedCriterion(self):
        return nn.CrossEntropyLoss()
        