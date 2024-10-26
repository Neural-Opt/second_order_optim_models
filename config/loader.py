import yaml
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import optimizer
import torch.optim.lr_scheduler 

class WrapperLR:
    def __init__(self,optim,lr_scheduler,conf):
        if lr_scheduler != None:
            optimizer_name = optim.__class__.__name__
            if conf["lr_scheduler"]['type'] == "CosineAnnealingLR":
                conf["lr_scheduler"]["params"]["eta_min"] = conf["optim"][optimizer_name]["params"]['lr'] / 200
            self.conf = conf
            self.lr_scheduler = lr_scheduler(optim,**conf["lr_scheduler"]["params"])  
            self.optimizer = optim
        else:
            self.lr_scheduler = None

    def get_lr(self,):
        return self.lr_scheduler.get_lr()
    def stepEpoch(self):
        
        if self.lr_scheduler != None and self.conf["runs"]["lr_schedule_per_epoch"]:
            self.lr_scheduler.step()
    def stepUpdate(self):
        if self.lr_scheduler != None and not self.conf["runs"]["lr_schedule_per_epoch"]:
            self.lr_scheduler.step()

def getConfig(yaml_file_path="config.yaml"):
    yaml_file_path =  os.path.join(os.path.dirname(os.path.abspath(__file__)), yaml_file_path)
    with open(yaml_file_path, 'r') as file:
        yaml_content = yaml.safe_load(file)
    return yaml_content

def getLRScheduler(optim):
    conf = getConfig()
    if conf["lr_scheduler"]["type"] != "-":
        class_lr = getattr(torch.optim.lr_scheduler, conf["lr_scheduler"]["type"])
        return WrapperLR(optim,class_lr,conf)#class_lr(optim,**conf["lr_scheduler"]["params"])  
    else:
        return WrapperLR(None,None,None)

def getOptim(exclude:list):
    params = getConfig()
    optim_params={}
    class_names = [i for i in params["optim"].keys() if i not in exclude]
    class_names = class_names + [class_names.pop(class_names.index("AdaHessian"))] if "AdaHessian" in class_names else class_names
    instances = []
    for class_name in class_names:
        if hasattr(optimizer, params["optim"][class_name]["name"]):
            class_ = getattr(optimizer, params["optim"][class_name]["name"])            
            instances.append(class_)
            optim_params[class_name] = params["optim"][class_name]["params"]
        else:
            print(f"Class {class_name} not found in {optimizer.__name__}.")
    return (class_names,instances,optim_params)
