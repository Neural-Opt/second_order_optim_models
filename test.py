import matplotlib.pyplot as plt
import numpy as np
from benchmark.postprocess import PostProcessor
from config.loader import getConfig
from log.Logger import Logger
from utils.utils import BenchmarkAnalyzer
#import tikzplotlib

class Plotter():
    def __init__(self,optimizers,data) -> None:
       # self.fig, self.axs = plt.subplots(2, 3, figsize=(12, 8))
        self.data = data
        self.optimizers = optimizers

    def plot(self,metric):
        conf = getConfig()
        kpi = list(filter(lambda x: x['name'] == metric, conf["kpi"]))[0]
        print(kpi)
        if kpi['plot'] == "graph":
            for optim in self.optimizers:
                data = self.data[optim][metric] 
                plt.plot(np.arange(len(data)),data, label=optim)
            plt.xlabel('Epochs')
        elif kpi['plot'] == 'box':
            box_data = [self.data[optim][metric] for optim in self.optimizers]
            plt.figure(figsize=(12,4))
            plt.boxplot(box_data, patch_artist=False, notch=False, vert=True, labels=self.optimizers)
        elif kpi['plot'] == 'bar':
            plt.figure(figsize=(9,4))
            for optim in self.optimizers:
                data = self.data[optim][metric] 
                plt.bar([optim], self.data[optim][metric], color ='maroon')
            plt.ylabel('Epochs')
        plt.legend()
     
        plt.savefig(f'result_plot_{metric}.png')       
  





"""run = 5
optim= ["AdaBelief","AdaHessian","Adam","AdamW","Apollo","ApolloW","RMSprop","SGD"]
l = Logger(rank="cuda",world_size=1, base_path=f"./runs/cifar10-steplr/{run}")
data = l.getData()

[PostProcessor(data[opt]) for opt in optim ]
for i in range(1,6):
    optim= ['Adam']
    l = Logger(rank="cuda",world_size=1, base_path=f"./runs/cifar10-steplr/{i}")
    data = l.getData()

    [PostProcessor(data[opt]) for opt in optim ]
    print(data["Adam"]["ttc"])


a = BenchmarkAnalyzer.mean("cifar10-steplr","Adam",join=False)
v = BenchmarkAnalyzer.v("cifar10-steplr","Adam",join=False)

print(a["ttc"],v["ttc"])
p = Plotter(optim,data)
p.plot(base_file=f"./runs/cifar10-steplr/{run}")"""
#tikzplotlib
optim= ["AdaBelief","AdaHessian","Adam","AdamW","Apollo","ApolloW","RMSprop","SGD"]
l = Logger(rank="cuda",world_size=1, base_path=f"./runs/cifar10-steplr/{5}")
data = l.getData()

[PostProcessor(data[opt]) for opt in optim ]
p = Plotter(optim,data)
p.plot("ttc")