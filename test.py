import matplotlib.pyplot as plt
import numpy as np
from benchmark.postprocess import PostProcessor
from benchmark.state import BenchmarkState
from config.loader import getConfig
from log.Logger import Logger
from utils.utils import BenchmarkAnalyzer

class Plotter():
    def __init__(self, optimizers, data) -> None:
        self.fig, self.axs = plt.subplots(2, 2, figsize=(12, 12))
        self.data = data
      
        self.optimizers = optimizers

    def plot(self, metric, title, subplot_idx):
        conf = getConfig()
        kpi = list(filter(lambda x: x['name'] == metric, conf["kpi"]))[0]
        ax = self.axs.flat[subplot_idx]  # Get the specific subplot axis

        if kpi['plot'] == "graph":
            for optim in self.optimizers:
                data = self.data[optim][metric]
                ax.plot(np.arange(len(data)), data, label=optim)
            ax.set_xlabel('Epochs')

        elif kpi['plot'] == 'box':
            box_data = [self.data[optim][metric] for optim in self.optimizers]
            box_data = np.log(box_data).tolist()
            ax.boxplot(box_data, patch_artist=False, notch=False, vert=True, labels=self.optimizers)
        elif kpi['plot'] == 'bar':
            for optim in self.optimizers:
                data = self.data[optim][metric]
                ax.bar([optim], self.data[optim][metric], color='maroon')
            ax.set_ylabel('Epochs')

        ax.set_title(title)

run = 1
optim = ["Adam","AdamW","SGD","AdaBelief","Apollo","ApolloW","AdaHessian","RMSprop"]
path = "./results/tinyimagenet-step-lr/second-order-best"
l = Logger(rank="cuda", world_size=1, base_path=path)
data = l.getData()
print(data.keys(),)
[PostProcessor(data[opt]) for opt in optim]

p = Plotter(optim, data)
metrics = metrics = ["acc_train", "acc_test", "train_loss", "test_loss",]#["acc_train", "acc_test", "train_loss", "test_loss",] #["gpu_mem", "tps"] #
#metrics = ["gpu_mem", "tps"]

titles =["Training Accuracy (milestone)", "Test Accuracy (milestone)", "Train Loss (milestone)","Test Loss (milestone)"]
#["Training Accuracy (milestone)", "Test Accuracy (milestone)", "Train Loss (milestone)","Test Loss (milestone)"]# ["GPU Memory (MiB) (milestone)", "Time per step (TPS) (milestone)"]#
#titles = ["GPU Memory (MiB) (milestone)", "Time per step (TPS) (milestone)"]

# Dictionary to track unique legend entries
legend_entries = {}

for idx, (metric, title) in enumerate(zip(metrics, titles)):
    p.plot(metric, title, idx)
    # Collect handles and labels from the current axis
    handles, labels = p.axs.flat[idx].get_legend_handles_labels()
    for handle, label in zip(handles, labels):
        if label not in legend_entries:
            legend_entries[label] = handle

# Create a single legend at the top of the entire figure
fig = p.fig
print(legend_entries.keys())
fig.legend(legend_entries.values(), legend_entries.keys(), loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=8,fontsize='large', handlelength=2)

plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust the layout to make space for the legend
plt.savefig('result_plot_tinyimagent-step-lr-second-order-best.png')
plt.show()
