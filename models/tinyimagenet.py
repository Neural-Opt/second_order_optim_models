from benchmark.benchmark import Benchmark
from config.loader import getConfig
from models.benchmarkset import BenchmarkSet
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import numpy as np
import random
from utils.utils import MeanAggregator
from datasets import load_dataset

class TinyImageNet(BenchmarkSet):
    def __init__(self, batch_size=128) -> None:
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
        dataset = load_dataset("zh-plus/tiny-imagenet")

        train_transform = transforms.Compose([ 
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
        ])
        val_transform = transforms.Compose([ transforms.ToTensor(),
            transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),])
     
    
        def transform_train(examples):
            # Convert list of images to tensors and stack them
            examples['image'] = [train_transform(image.convert("RGB")) for image in examples['image']]
            return examples

        def transform_val(examples):
            # Convert list of images to tensors and stack them
            examples['image'] = [val_transform(image.convert("RGB")) for image in examples['image']]
            return examples

        train_dataset = dataset['train']

        test_dataset = dataset['valid']
        train_dataset.set_transform(transform_train)
        test_dataset.set_transform(transform_val)
      
     
        """train_size = int(0.8 * len(trainset))
        val_size = len(trainset) - train_size
        train_dataset, val_dataset = random_split(trainset, [train_size, val_size])"""
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, worker_init_fn=TinyImageNet.seed_worker, generator=g)
        val_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, worker_init_fn=TinyImageNet.seed_worker, generator=g)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, worker_init_fn=TinyImageNet.seed_worker, generator=g)
        
        return (train_loader, test_loader, val_loader)

    def getAssociatedModel(self,rank):
        model = models.resnet18()
        model.fc = nn.Linear(model.fc.in_features, self.num_classes)
        model.to(rank)
        ddp_model = torch.nn.DataParallel(model)
        return ddp_model

    def getAssociatedCriterion(self):
        return nn.CrossEntropyLoss()
    def train(self, model, device, train_loader, optimizer, criterion, create_graph,lr_scheduler):
        model.train()
        benchmark = Benchmark.getInstance(None)
        accuracy = MeanAggregator(measure=lambda *args: (args[0].eq(args[1]).sum().item() / args[1].size(0)))
        avg_loss = MeanAggregator()
        for batch in train_loader:
            benchmark.measureGPUMemUsageStart(device)
            benchmark.stepStart()
            inputs, targets = batch['image'], batch['label']
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward(create_graph=create_graph)
            optimizer.step()
            lr_scheduler.stepUpdate()
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
        avg_loss = MeanAggregator()

        with torch.no_grad():
            for batch in test_loader:
                inputs, targets = batch['image'], batch['label']
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                avg_loss(loss.item())
                _, predicted = outputs.max(1)
                accuracy(predicted, targets)

            benchmark.add("acc_test",accuracy.get())
            benchmark.add("test_loss",avg_loss.get())

        return accuracy.get()
