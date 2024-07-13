from benchmark.postprocess import PostProcessor
from benchmark.state import BenchmarkState
from utils.utils import BenchmarkAnalyzer
from visualize.table import makeTable, print_table 
import numpy as np

optimizers = [ 'SGD','Adam','AdamW','Apollo','ApolloW','AdaBelief',"RMSprop","AdaHessian"]

def eval_kpis_mean():
    runs_to_include = ['cifar10-steplr']
    cols = [" "]+[f"{k} - Speed (TPS)"for k in runs_to_include]
    cols += ([f"{k} - Memory (GPU)" for k in runs_to_include])
    rows = []

    """
    X = []
    Y = []
    Z = X/Y
    Var(Z) = Var(X)/Var(Y) ??
    """

    for optim in optimizers:
        row = [optim]
        for set in runs_to_include:
            sgd_state = BenchmarkAnalyzer.mean(set,"SGD")
            state = BenchmarkAnalyzer.mean(set,optim)
            speed_x, speed_x_std = np.mean(state['tps'] / sgd_state['tps']), np.std(state['tps'] / sgd_state['tps'])
            mem_x, mem_x_std = np.mean(state['gpu_mem'] / sgd_state['gpu_mem']), np.std(state['gpu_mem'] / sgd_state['gpu_mem'])

            row.append(f"{round(speed_x,4)} ± {round(speed_x_std,3)}")
            row.append(f"{round(mem_x,4)} ± {round(mem_x_std,3)}")

        rows.append(row)
    print_table(cols,[f"Row{i}" for i in range(1,len(rows)+1)],rows)
def eval_kpis():
    runs_to_include = ['cifar10-steplr']
    cols = [" "]+[f"{k} - Speed (TPS)"for k in runs_to_include]
    cols += ([f"{k} - Memory (GPU)" for k in runs_to_include])
    rows = []
    run = 1

    for optim in optimizers:
        row = [optim]
        for set in runs_to_include:
            state = BenchmarkState(f"./runs/{set}/{run}/{optim}/benchmark.json")
            sgd_state = BenchmarkState(f"./runs/{set}/{run}/SGD/benchmark.json")
            PostProcessor(state) 
            state = {key :np.array(state[key]) for key in state.dump().keys()}
            sgd_state = {key :np.array(sgd_state[key]) for key in sgd_state.dump().keys()}
 
            speed_x, speed_x_std = np.mean(state['tps'] / sgd_state['tps']), np.std(state['tps'] / sgd_state['tps'])
            mem_x, mem_x_std = np.mean(state['gpu_mem'] / sgd_state['gpu_mem']), np.std(state['gpu_mem'] / sgd_state['gpu_mem'])

            row.append(f"{round(speed_x,4)} ± {round(speed_x_std,3)}")
            row.append(f"{round(mem_x,4)} ± {round(mem_x_std,3)}")

        rows.append(row)
    print_table(cols,[f"Row{i}" for i in range(1,len(rows)+1)],rows)

def eval_acc_mean():
    runs_to_include = ['cifar10-steplr']
    cols = [" "]+[f"{k} - Accuracy"for k in runs_to_include]
    rows = []
    for optim in optimizers:
        row = [optim]
        for set in runs_to_include:
            mean_state = BenchmarkAnalyzer.mean(set,optim,join=False,reducer=lambda x: x[:,x.shape[1] - 1:])
            std_state = BenchmarkAnalyzer.std(set,optim,join=False,reducer=lambda x: x[:,x.shape[1] - 1:])
            row.append(f"{round( mean_state['acc_test'].item(),4)} ± {round(std_state['acc_test'].item(),3)}")
        rows.append(row)
    
    print_table(cols,[f"Row{i}" for i in range(1,len(rows)+1)],rows)
def eval_convergence():
    runs_to_include = ['cifar10-steplr']
    cols = [" "]+[f"{k} - TTC"for k in runs_to_include]
    rows = []
    for optim in optimizers:
        row = [optim]
        for set in runs_to_include:
            mean_state = BenchmarkAnalyzer.mean(set,optim,join=False)
            std_state = BenchmarkAnalyzer.std(set,optim,join=False)
            row.append(f"{round( mean_state['ttc'].item(),4)} ± {round(std_state['ttc'].item(),3)}")
        rows.append(row)
    
    print_table(cols,[f"Row{i}" for i in range(1,len(rows)+1)],rows)

# Example usage
eval_kpis()
#print(eval_kpis())
