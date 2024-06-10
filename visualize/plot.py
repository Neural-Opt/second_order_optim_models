import matplotlib.pyplot as plt
import numpy as np
from config.loader import getConfig
class Plotter():
    def __init__(self,optimizers,data) -> None:
        self.fig, self.axs = plt.subplots(2, 2, figsize=(12, 8))
        self.data = data
        self.optimizers = optimizers

    def plot(self,base_file=""):
        conf = getConfig()
        kpis = conf["kpi"]
        for optim in self.optimizers:
            for i, k in enumerate(self.data[optim]):
                kpi = list(filter(lambda x: x["name"] == k,kpis))[0]
                ax = self.axs.flatten()[i]
                value = self.data[optim][k]
                ax.plot(np.arange(len(value)),value, label=optim)
                ax.set_title(f"{kpi['fqn']} ({kpi['unit']})")
                ax.legend()
             
        plt.tight_layout() 
        plt.savefig(f'{base_file}/result_plot.png')       

