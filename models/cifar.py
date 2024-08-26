from benchmark.benchmark import Benchmark
from config.loader import getConfig
from models.benchmarkset import BenchmarkSet
from models.resnet.resnet110 import ResNet110
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader,DistributedSampler, random_split
from torch.nn.parallel import DistributedDataParallel as DP

import torch.nn as nn
import torch
import numpy as np
import random

from utils.utils import MeanAggregator
class CIFAR(BenchmarkSet):
    def __init__(self,batch_size=256,dataset="cifar10") -> None:
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
    @staticmethod
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
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, num_workers=1,worker_init_fn=CIFAR.seed_worker, generator=g)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size,num_workers=1,worker_init_fn=CIFAR.seed_worker, generator=g)
        test_loader = DataLoader(testset, batch_size=self.batch_size, num_workers=1,worker_init_fn=CIFAR.seed_worker, generator=g)
        return (train_loader ,test_loader , val_loader)
    def getAssociatedModel(self,rank):
        model = models.resnet18()
        model.fc = nn.Linear(model.fc.in_features, self.num_classes)   
        model = model.to(rank)
        
        ddp_model = torch.nn.DataParallel(model)#, device_ids=[rank])
        print(f"Number of params: {sum(p.numel() for p in model.parameters())}")

        return ddp_model
    def getAssociatedCriterion(self):
        return nn.CrossEntropyLoss()
    def train(self, model, device, train_loader, optimizer, criterion,create_graph,lr_scheduler):
        model.train()
        print("CIFAR")
        #benchmark = Benchmark.getInstance(None)
        accuracy = MeanAggregator(measure=lambda *args:(args[0].eq(args[1]).sum().item() / args[1].size(0)))
        avg_loss = MeanAggregator()
        for inputs, targets in train_loader:
            #benchmark.measureGPUMemUsageStart(rank=device)
            #benchmark.stepStart()
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward(create_graph=create_graph) 
            optimizer.step()
            lr_scheduler.stepUpdate()
            _, predicted = outputs.max(1)

            avg_loss(loss.item())
            accuracy(predicted,targets)
            #benchmark.stepEnd()
            #benchmark.measureGPUMemUsageEnd(rank=device)

        #benchmark.add("acc_train",accuracy.get())
        #benchmark.add("train_loss",avg_loss.get())
       # benchmark.flush()
       
        return avg_loss.get(), accuracy.get()
    @torch.no_grad()
    def test(self,model, device, test_loader, criterion):
        model.eval()
        # = Benchmark.getInstance(None)

        accuracy = MeanAggregator(measure=lambda *args:(args[0].eq(args[1]).sum().item() / args[1].size(0)))
        avg_loss = MeanAggregator()
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                _, predicted = outputs.max(1)
                avg_loss(loss.item())
                accuracy(predicted,targets)

            #benchmark.add("acc_test",accuracy.get())
            #benchmark.add("test_loss",avg_loss.get())
        return accuracy.get()
        