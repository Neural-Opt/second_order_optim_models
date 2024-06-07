import matplotlib.pyplot as plt
import numpy as np

class Plotter():
    def __init__(self,optimizers,data) -> None:
        self.fig, self.axs = plt.subplots(2, 2, figsize=(12, 8))

        self.data = data
        self.optimizers = optimizers

    def plot(self,):
        for optim in self.optimizers:
            for i, k in enumerate(self.data[optim]):
                ax = self.axs.flatten()[i]
                value = self.data[optim][k]
                print(np.arange(len(value)),value)
                ax.plot(np.arange(len(value)),value, label=optim)
                ax.set_title(k)
                ax.legend()
                print(i, k)
        plt.tight_layout() 
        plt.savefig('multiple_plots.png')       

