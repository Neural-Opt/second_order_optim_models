from benchmark.benchmark import Benchmark
from config.loader import getConfig
from models.benchmarkset import BenchmarkSet
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch
import numpy as np
import random
from utils.utils import MeanAggregator

class TinyImageNet(BenchmarkSet):
    def __init__(self, batch_size=256) -> None:
        super().__init__()
        self.conf = getConfig()
        self.batch_size = batch_size
        self.num_classes = 200

    def log(self):
        pass

    def setup(self):
        pass

    @staticmethod
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def getDataLoader(self):   
        g = torch.Generator()
        g.manual_seed(404)
    
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        trainset = datasets.ImageFolder(root=f"{self.conf['dataset']['path']}/train", transform=train_transform)
        valset = datasets.ImageFolder(root=f"{self.conf['dataset']['path']}/val", transform=val_transform)
        testset = datasets.ImageFolder(root=f"{self.conf['dataset']['path']}/test", transform=val_transform)

     
        """train_size = int(0.8 * len(trainset))
        val_size = len(trainset) - train_size
        train_dataset, val_dataset = random_split(trainset, [train_size, val_size])"""
        
        train_loader = DataLoader(trainset, batch_size=self.batch_size, shuffle=True, num_workers=4, worker_init_fn=TinyImageNet.seed_worker, generator=g)
        val_loader = DataLoader(valset, batch_size=self.batch_size, shuffle=False, num_workers=4, worker_init_fn=TinyImageNet.seed_worker, generator=g)
        test_loader = DataLoader(testset, batch_size=self.batch_size, shuffle=False, num_workers=4, worker_init_fn=TinyImageNet.seed_worker, generator=g)
        
        return (train_loader, test_loader, val_loader)

    def getAssociatedModel(self,rank):
        model = models.resnet50()
        model.fc = nn.Linear(model.fc.in_features, self.num_classes)
        model.to(rank)
        ddp_model = torch.nn.DataParallel(model)
        return ddp_model

    def getAssociatedCriterion(self):
        return nn.CrossEntropyLoss()

    def train(self, model, device, train_loader, optimizer, criterion, create_graph):
        model.train()
        benchmark = Benchmark.getInstance(None)
        accuracy = MeanAggregator(measure=lambda *args: (args[0].eq(args[1]).sum().item() / args[1].size(0)))
        avg_loss = MeanAggregator()
        for inputs, targets in train_loader:
            benchmark.measureGPUMemUsageStart(device)
            benchmark.stepStart()
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward(create_graph=create_graph)
            optimizer.step()
            _, predicted = outputs.max(1)


            avg_loss(loss.item())
            accuracy(predicted, targets)
            benchmark.stepEnd()
            benchmark.measureGPUMemUsageEnd(device)

        benchmark.add("acc_train",accuracy.get())
        benchmark.add("train_loss",avg_loss.get())
        benchmark.flush()

        return avg_loss.get(), accuracy.get()

    @torch.no_grad()
    def test(self, model, device, test_loader, criterion):
        model.eval()
        benchmark = Benchmark.getInstance(None)

        accuracy = MeanAggregator(measure=lambda *args: (args[0].eq(args[1]).sum().item() / args[1].size(0)))

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                accuracy(predicted, targets)

            benchmark.add("acc_test",accuracy.get())
        
        return accuracy.get()
