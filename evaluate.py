from benchmark.postprocess import PostProcessor
from benchmark.state import BenchmarkState
from utils.utils import BenchmarkAnalyzer
from visualize.table import makeTable, print_table 
import numpy as np
def eval_kpis():
    run = 1
    optimizers = [ 'SGD','Adam','AdamW','Apollo','ApolloW','AdaBelief','AdaHessian',"RMSprop"]
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
    #makeTable(head=optimizers,data= data_dict)

def eval_acc():
    run = 1
    optimizers = [ 'SGD','Adam','AdamW','Apollo','ApolloW','AdaBelief','AdaHessian',"RMSprop"]
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
    run = 1
    optimizers = [ 'SGD','Adam','AdamW','Apollo','ApolloW','AdaBelief','AdaHessian',"RMSprop"]
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
eval_convergence()
eval_acc()
#print(eval_kpis())
