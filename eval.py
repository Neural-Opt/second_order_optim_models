from benchmark.postprocess import PostProcessor
from benchmark.state import BenchmarkState
from utils.utils import BenchmarkAnalyzer
from visualize.table import makeTable, print_table 
import numpy as np

optimizers = ["AdaBelief","AdaHessian","Adam","AdamW","Apollo","ApolloW","RMSprop","SGD"]

def eval_kpis_mean():
    runs_to_include = ['tinyimagenet-cosine']
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
            sgd_state = BenchmarkAnalyzer.getConcatStates(set,"SGD")
            state = BenchmarkAnalyzer.getConcatStates(set,optim)
            speed_arr = (state['tps'] / sgd_state['tps']).reshape(1,-1)
            gpu_mem_arr = (state['gpu_mem'] / sgd_state['gpu_mem']).reshape(1,-1)

            speed_x, speed_x_std = np.mean(speed_arr), np.std(speed_arr)
            mem_x, mem_x_std =np.mean(gpu_mem_arr), np.std(gpu_mem_arr)

            row.append(f"{round(speed_x,4)} ± {round(speed_x_std,3)}")
            row.append(f"{round(mem_x,4)} ± {round(mem_x_std,3)}")

        rows.append(row)
    print_table(cols,[f"Row{i}" for i in range(1,len(rows)+1)],rows)
def eval_kpis():
    runs_to_include = ['cifar10-steplr']
    cols = [" "]+[f"{k} - Speed (TPS)"for k in runs_to_include]
    cols += ([f"{k} - Memory (GPU)" for k in runs_to_include])
    rows = []
    run = "second_order_best"

    for optim in optimizers:
        row = [optim]
        for set in runs_to_include:
            print(f"./results/{set}/{run}/{optim}/benchmark.json")
            state = BenchmarkState(f"./results/{set}/{run}/{optim}/benchmark.json")
            sgd_state = BenchmarkState(f"./results/{set}/{run}/SGD/benchmark.json")
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
            state = BenchmarkAnalyzer.getConcatStates(set,optim,reducer=lambda x: x[:,x.shape[1] - 1:])
            accs = (state['acc_test']).reshape(1,-1)
            acc_mean, acc_sgd = np.mean(accs), np.std(accs)

            row.append(f"{round(acc_mean,4)} ± {round(acc_sgd,3)}")
        rows.append(row)
    
    print_table(cols,[f"Row{i}" for i in range(1,len(rows)+1)],rows)
def eval_acc():
    runs_to_include = ['cifar10-steplr']
    cols = [" "]+[f"{k} - Accuracy"for k in runs_to_include]
    rows = []
    run = "second_order_best"
    for optim in optimizers:
        row = [optim]
        for set in runs_to_include:
            state = BenchmarkAnalyzer.getConcatStates(set,optim,run=run,reducer=lambda x: x[x.shape[0] - 1:])
            state['acc_test']  = state['acc_test']
            accs = (state['acc_test']).reshape(1,-1)
            acc_mean, acc_sgd = np.mean(accs), np.std(accs)

            row.append(f"{round(acc_mean,4)} ± {round(acc_sgd,3)}")
        rows.append(row)
    
    print_table(cols,[f"Row{i}" for i in range(1,len(rows)+1)],rows)
def eval_convergence():
    runs_to_include = ['cifar10-step-lr']
    cols = [" "]+[f"{k} - TTC"for k in runs_to_include]
    rows = []
    for optim in optimizers:
        row = [optim]
        run=5
        for set in runs_to_include:
            state = BenchmarkAnalyzer.getConcatStates(set,optim,run=run)
            ttcs = (state['ttc']).reshape(1,-1)
            ttc_mean, ttc_sgd = np.mean(ttcs), np.std(ttcs)
            row.append(f"{round( ttc_mean,4)} ± {round(ttc_sgd,3)}")
        rows.append(row)
    
    print_table(cols,[f"Row{i}" for i in range(1,len(rows)+1)],rows)

# Example usage
eval_acc()
#print(eval_kpis())
