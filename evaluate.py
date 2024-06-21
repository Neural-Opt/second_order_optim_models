from benchmark.state import BenchmarkState
from utils.utils import BenchmarkAnalyzer
from visualize.table import makeTable 
import numpy as np
def eval():
    run = 1
    optimizers = [ 'SGD','Apollo','Adam','AdamW','AdaBelief']
    runs_to_include = ['test']
    data_dict = {f"{k}\nSpeed (TPS)":[] for k in runs_to_include}
    data_dict.update({f"{k}\nMemory (GPU)": [] for k in runs_to_include})


    for set in runs_to_include:
         for optim in optimizers:
            sgd_ref = BenchmarkState(f"./runs/{set}/{run}/SGD/benchmark.json")
            state = BenchmarkState(f"./runs/{set}/{run}/{optim}/benchmark.json")
            tps_curr_set = np.mean(np.array(state.get("tps")))
            tps_ref_sgd = np.mean(np.array(sgd_ref.get("tps")))
            gpu_mem_curr_set = np.mean(np.array(state.get("gpu_mem")))
            gpu_mem_ref_sgd = np.mean(np.array(sgd_ref.get("gpu_mem")))
        
            speed_x = f"{tps_curr_set/tps_ref_sgd:.4f}"
            mem_x = f"{gpu_mem_curr_set/gpu_mem_ref_sgd:.4f}"
            data_dict[f"{set}\nSpeed (TPS)"].append(speed_x)
            data_dict[f"{set}\nMemory (GPU)"].append(mem_x)
    makeTable(head=optimizers,data= data_dict)
def print_table(column_names, row_names, data):
    column_widths = [max(len(str(item)) for item in column) for column in zip(*data)]
    column_widths = [max(len(name), width) for name, width in zip(column_names, column_widths)]

    header = "| " + " | ".join(f"{name:<{width}}" for name, width in zip(column_names, column_widths)) + " |"
    separator = "+-" + "-+-".join("-" * width for width in column_widths) + "-+"

    print(separator)
    print(header)
    print(separator)
    for row_name, row_data in zip(row_names, data):
        row = "| " + " | ".join(f"{str(item):<{width}}" for item, width in zip(row_data, column_widths)) + " |"
        print(row)
        print(separator)



# Example usage
column_names = ["Name", "Age", "Occupation"]
row_names = ["Row1", "Row2", "Row3"]
data = [
    ["Alice", 30, "Engineer"],
    ["Bob", 25, "Artist"],
    ["Charlie", 35, "Doctor"]
]

print_table(column_names, row_names, data)
