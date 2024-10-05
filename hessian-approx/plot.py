import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import matplotlib.pyplot as plt
import numpy as np
import torch
from benchmark.postprocess import PostProcessor
from benchmark.state import BenchmarkState
from config.loader import getConfig
from log.Logger import Logger
from utils.utils import BenchmarkAnalyzer

def moving_average(data, window_size=10):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
class Plotter():
    def __init__(self, optimizers, data) -> None:
        self.fig, self.axs = plt.subplots(2, 2, figsize=(12, 12))
        self.data = data
        self.reducer = lambda x:  x
        self.optimizers = optimizers
    def setReducer(self,reducer):
        self.reducer = reducer
    def plot(self, metric, title, subplot_idx,plot_type):
        conf = getConfig()
        ax = self.axs.flat[subplot_idx]  # Get the specific subplot axis

        if plot_type == "graph":
            for optim in self.optimizers:
                data = self.reducer(self.data[optim][metric])
                print(optim,torch.tensor(data).shape)
                ax.plot(np.arange(len(data)), data, label=optim)
                ax.set_ylabel('PHIO')

        elif plot_type == 'box':
            box_data = []
            for optim in self.optimizers:
                # Append data for both metrics for each optimizer
                data = self.reducer(self.data[optim][metric])

              
                box_data.append(data[0])
                box_data.append(data[1])
               # box_data.append(data[1])

            # Create positions for box plots: two positions per optimizer
            num_optimizers = len(self.optimizers)
            positions = []
            for i in range(num_optimizers):
                positions.extend([2 * i + 1, 2 * i + 2])  # Two positions for each optimizer

            # Set up labels for x-axis
            x_labels = []
            for optim in self.optimizers:
                x_labels.extend([f'{optim}_1', f'{optim}_2'])  # Label for each metric
            boxplots = ax.boxplot(box_data, positions=positions, patch_artist=True, notch=False, vert=True)

            # Change colors: second box plot for each optimizer
            colors = ['lightblue', 'lightgreen']
            for i, box in enumerate(boxplots['boxes']):
                if i % 2 == 1:  # Apply color to every second box
                    box.set_facecolor(colors[1])
                else:
                    box.set_facecolor(colors[0])

            # Adjust x-axis ticks and labels
            ax.set_xticks([2 * i + 1.5 for i in range(num_optimizers)])  # Set mid-point for each optimizer
            ax.set_xticklabels(self.optimizers)  # Optimizer names (centered)

            ax.set_title(title)

            ax.set_title(title)
        elif plot_type == 'bar':
            for optim in self.optimizers:
                data = self.data[optim][metric]
                ax.bar([optim], self.data[optim][metric], color='maroon')
            ax.set_ylabel('Epochs')

        ax.set_title(title)

run = 1
optim = ["Adam","AdaHessian","Apollo","AdaBelief","SApollo"]#,"AdamW","SGD","AdaBelief","Apollo","ApolloW","AdaHessian","RMSprop"]
path = "./results/hessian-approx/sapollo"
l = Logger(rank="cuda", world_size=1, base_path=path)
data = l.getData()
p = Plotter(optim, data)
#[PostProcessor(data[opt]) for opt in optim]
metrics =["cosine_deg","cosine_deg","cosine_deg","cosine_deg"]

#print(data["Apollo"]["loss"])
titles =["conv_layer_0","conv_layer_1","fc_layer_0","fc_layer_1"]

# Dictionary to track unique legend entries
legend_entries = {}

for idx, (metric, title) in enumerate(zip(metrics, titles)):
    p.setReducer(lambda x: moving_average(x[0][2*idx:2+2*idx][1]))

   # p.setReducer(lambda x:  np.log((np.array(x))))
    print(metric)
    p.plot(metric, title, idx,"graph")
    # Collect handles and labels from the current axis
    handles, labels = p.axs.flat[idx].get_legend_handles_labels()
    for handle, label in zip(handles, labels):
        if label not in legend_entries:
            legend_entries[label] = handle

# Create a single legend at the top of the entire figure
fig = p.fig
fig.legend(legend_entries.values(), legend_entries.keys(), loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=8,fontsize='large', handlelength=2)

plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust the layout to make space for the legend
plt.savefig('./results/hessian-approx/sapollo/approx_bias.png')
import tikzplotlib

tikzplotlib.save('./results/hessian-approx/sapollo/approx_bias.tex')
plt.show()
