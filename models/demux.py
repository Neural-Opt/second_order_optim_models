from config.loader import getConfig
from models.cifar import CIFAR
from models.wmt14 import WMT14


def getBenchmarkSet():
    
    conf = getConfig()
    name = conf["dataset"]["name"]
    if "cifar" in name:
        return CIFAR()

