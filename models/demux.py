from config.loader import getConfig
from models.cifar import CIFAR
from models.tinyimagenet import TinyImageNet
from models.wmt14 import WMT14


def getBenchmarkSet():
    
    conf = getConfig()
    name = conf["dataset"]["name"]
    if "cifar" in name:
        return CIFAR()
    if "wmt14" in name:
        return WMT14()
    if "tinyimagenet" in name:
        return TinyImageNet()

