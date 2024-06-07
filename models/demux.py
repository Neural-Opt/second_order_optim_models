from config.loader import getConfig
from models.cifar import CIFAR


def getBenchmarkSet():
    conf = getConfig()
    name = conf["dataset"]["name"]
    if "cifar" in name:
        return CIFAR()