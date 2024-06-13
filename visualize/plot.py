import matplotlib.pyplot as plt
import numpy as np
from config.loader import getConfig
#import tikzplotlib

class Plotter():
    def __init__(self,optimizers,data) -> None:
        self.fig, self.axs = plt.subplots(2, 3, figsize=(12, 8))
        self.data = data
        self.optimizers = optimizers

    def plot(self,base_file=""):
        conf = getConfig()
        kpis = conf["kpi"]
        boxplots={}

        for optim in self.optimizers:
            for i, k in enumerate(self.data[optim]):
                kpi = list(filter(lambda x: x["name"] == k,kpis))[0]
                ax = self.axs.flatten()[i]
                value = self.data[optim][k]
                print(k,value)
                if kpi['plot'] == "graph":
                    ax.plot(np.arange(len(value)),value, label=optim)
                elif kpi['plot'] == "box":
                    if k not in boxplots:
                        boxplots[k] = {"values": [], "labels": []}
                    boxplots[k]["values"].append(value)
                    boxplots[k]["labels"].append(optim)
                    
                    ax.boxplot(boxplots[k]["values"], patch_artist=False, notch=False, vert=True, labels=boxplots[k]["labels"])
                ax.legend()
                ax.set_title(f"{kpi['fqn']} ({kpi['unit']})")
                
        plt.tight_layout() 
        plt.savefig(f'{base_file}/result_plot.png')       
      #  tikzplotlib.save(f'{base_file}/result-tex.tex')


#tikzplotlib