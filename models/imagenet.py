from config.loader import getConfig
from models.benchmarkset import BenchmarkSet
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn

class ImageNet(BenchmarkSet):
    def __init__(self, batch_size=128, dataset_path="/path/to/imagenet") -> None:
        super().__init__()
        self.conf = getConfig()
        self.batch_size = batch_size
        self.dataset_path = dataset_path
        self.num_classes = 1000  # ImageNet has 1000 classes

    def log(self):
        pass

    def setup(self):
        pass

    def getDataLoader(self):
        # Data augmentation and normalization for training
        # Just normalization for validation
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        image_datasets = {x: datasets.ImageFolder(root=f"{self.dataset_path}/{x}", transform=data_transforms[x])
                          for x in ['train', 'val']}
        dataloaders = {x: DataLoader(image_datasets[x], batch_size=self.batch_size, shuffle=True, num_workers=4)
                       for x in ['train', 'val']}
        return dataloaders['train'], dataloaders['val']

    def getAssociatedModel(self):
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, self.num_classes)
        return model

    def getAssociatedCriterion(self):
        return nn.CrossEntropyLoss()

