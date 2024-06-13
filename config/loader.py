import yaml
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import optimizer
import torch.optim.lr_scheduler 

class DummyLR:
    def __init__(self,*args):
        pass
    def step(self):
        pass
def getConfig(yaml_file_path="config.yaml"):
    yaml_file_path =  os.path.join(os.path.dirname(os.path.abspath(__file__)), yaml_file_path)
    with open(yaml_file_path, 'r') as file:
        yaml_content = yaml.safe_load(file)
    return yaml_content

def getLRScheduler(optim):
    conf = getConfig()
    if conf["lr_scheduler"]["type"] != "-":
        class_lr = getattr(torch.optim.lr_scheduler, conf["lr_scheduler"]["type"])
    else:
        class_lr = DummyLR
    return class_lr(optim,conf["lr_scheduler"]["step"], conf["lr_scheduler"]["gamma"])         

def getOptim(exclude:list):
    params = getConfig()
    class_names = ["AdaBelief","AdaHessian","Adam","AdamW","Apollo","RMSprop","SGD"]
    class_names = [i for i in class_names if i not in exclude]
    instances = []
    for class_name in class_names:
        if hasattr(optimizer, class_name):
            class_ = getattr(optimizer, class_name)            
            instances.append(class_)
        else:
            print(f"Class {class_name} not found in {optimizer.__name__}.")
    return (class_names,instances,params)
