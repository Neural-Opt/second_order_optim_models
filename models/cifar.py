from config.loader import getConfig
from models.benchmarkset import BenchmarkSet
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch
import numpy as np
import random
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
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def getDataLoader(self,):   
        g = torch.Generator()
        g.manual_seed(404)
    
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
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=1,worker_init_fn=CIFAR.seed_worker, generator=g)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=1,worker_init_fn=CIFAR.seed_worker, generator=g)
        test_loader = DataLoader(testset, batch_size=self.batch_size, shuffle=False, num_workers=1,worker_init_fn=CIFAR.seed_worker, generator=g)
        return (train_loader ,test_loader , val_loader)
    def getAssociatedModel(self):
        model = models.resnet18()
        model.fc = nn.Linear(model.fc.in_features, self.num_classes)
        return model
    def getAssociatedCriterion(self):
        return nn.CrossEntropyLoss()
        