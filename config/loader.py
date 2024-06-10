import yaml
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import optimizer
import torch.optim.lr_scheduler 

def getConfig(yaml_file_path="config.yaml"):
    yaml_file_path =  os.path.join(os.path.dirname(os.path.abspath(__file__)), yaml_file_path)
    with open(yaml_file_path, 'r') as file:
        yaml_content = yaml.safe_load(file)
    return yaml_content

def getLRScheduler(optim):
    conf = getConfig()
    class_lr = getattr(torch.optim.lr_scheduler, conf["lr_scheduler"]["type"])
    return class_lr(optim,conf["lr_scheduler"]["step"], conf["lr_scheduler"]["gamma"])         

def getOptim(model,exclude:list):
    params = getConfig()
    class_names = ["AdaBelief","AdaHessian","Adam","AdamW","Apollo","RMSprop","SGD"]
    class_names = [i for i in class_names if i not in exclude]
    instances = []
    print(class_names)
    for class_name in class_names:
        if hasattr(optimizer, class_name):
            print(params.keys())
            print(class_name)
            class_ = getattr(optimizer, class_name)            
            instances.append(class_(model.parameters(),**params[class_name]))
        else:
            print(f"Class {class_name} not found in {optimizer.__name__}.")
    return (class_names,instances,params)
