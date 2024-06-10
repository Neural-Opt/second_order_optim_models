from benchmark.state import BenchmarkState
from utils.utils import BenchmarkAnalyzer
from visualize.table import makeTable 
import numpy as np
def eval():
    optimizers = [ 'SGD','Apollo','Adam','AdamW','AdaBelief']
    runs_to_include = ['test']
    data_dict = {f"{k}\nSpeed (TPS)":[] for k in runs_to_include}
    data_dict.update({f"{k}\nMemory (GPU)": [] for k in runs_to_include})


    for set in runs_to_include:
         for optim in optimizers:
            sgd_ref = BenchmarkState(f"./runs/{set}/1/SGD/benchmark.json")
            state = BenchmarkState(f"./runs/{set}/1/{optim}/benchmark.json")
            tps_curr_set = np.mean(np.array(state.get("tps")))
            tps_ref_sgd = np.mean(np.array(sgd_ref.get("tps")))
            gpu_mem_curr_set = np.mean(np.array(state.get("gpu_mem")))
            gpu_mem_ref_sgd = np.mean(np.array(sgd_ref.get("gpu_mem")))
        
            speed_x = f"{tps_curr_set/tps_ref_sgd:.4f}"
            mem_x = f"{gpu_mem_curr_set/gpu_mem_ref_sgd:.4f}"
            data_dict[f"{set}\nSpeed (TPS)"].append(speed_x)
            data_dict[f"{set}\nMemory (GPU)"].append(mem_x)
    makeTable(head=optimizers,data= data_dict)
print(BenchmarkAnalyzer.var("test","Adam"))
eval()