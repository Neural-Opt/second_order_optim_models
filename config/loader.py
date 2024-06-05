import yaml
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import optimizer

def getConfig(yaml_file_path="config.yaml"):
    yaml_file_path =  os.path.join(os.path.dirname(os.path.abspath(__file__)), yaml_file_path)
    with open(yaml_file_path, 'r') as file:
        yaml_content = yaml.safe_load(file)
        print(yaml_content)
    return yaml_content

def get_optim(exclude:list):
    params = getConfig()
    print(params)
    class_names = ["AdaBelief","AdaHessian","Adam","AdamW","Apollo","RMSprop","SGD"]
    class_names = [i for i in class_names if i not in exclude]
    instances = []
    for class_name in class_names:
        if hasattr(optimizer, class_name):
            class_ = getattr(optimizer, class_name)
            instance = class_(**params[class_names])
            instances.append(instance)
        else:
            print(f"Class {class_name} not found in {optimizer.__name__}.")
    return instances

get_optim([])