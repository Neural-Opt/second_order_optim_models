from config.loader import getConfig
from models.benchmarkset import BenchmarkSet
from torchvision import datasets, transforms


class CIFAR(BenchmarkSet):
    def __init__(self,batch_size=16,dataset="cifar10") -> None:
        super().__init__()
        self.conf = getConfig()

    def log(self):
        pass
    def setup(self,):
        pass
    def getDataSets(self,):
        if self.dataset == 'cifar10':
            dataset = datasets.CIFAR10
            num_classes = 10
        else:
            dataset = datasets.CIFAR100
            num_classes = 100
        trainset = dataset(self.conf[self.dataset]["path"], train=True, download=True,
                       transform=transforms.Compose([
                           transforms.RandomCrop(32, padding=4),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))#https://github.com/kuangliu/pytorch-cifar/issues/19
                       ]))
        valset = dataset(self.conf[self.dataset]["path"], train=False, download=False,
                     transform=transforms.Compose([
                         transforms.ToTensor(),
                         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),#https://github.com/kuangliu/pytorch-cifar/issues/19
                      ]))

        return (trainset, valset, len(trainset), len(valset))
    def getAssociatedModel(self):
        pass
        