import os
import pickle
from benchmark.benchmark import Benchmark,BenchmarkState
from config.loader import getConfig
from visualize.plot import Plotter
import torch.distributed as dist
import numpy as np
import pickle
import json

def createNewRun(base_dir):
    if not os.path.isdir(base_dir):
        os.makedirs(base_dir)
    dir_count = sum(os.path.isdir(os.path.join(base_dir, d)) for d in os.listdir(base_dir)) 
    os.makedirs( os.path.join(base_dir,  f"{dir_count + 1}"))
    return os.path.join(base_dir,  f"{dir_count + 1}")
    
class Logger:
    def __init__(self,base_path,rank,world_size) -> None:

        self.base_path = base_path
        self.world_size = world_size
        self.rank = rank


    def setup(self,optim):
        self.benchmarkState = BenchmarkState()
        self.benchmark = Benchmark.getInstance(self.benchmarkState)

    def trash(self,):
        Benchmark.cleanUp()

    def getData(self):
        output = {}
        for d in os.listdir(self.base_path):
            if d == "conf.json" or d == "result_plot.png":
                continue
            state = BenchmarkState(f"{os.path.join(self.base_path, d)}/benchmark.json")
            state.load()
            data = state.dump()
           
            output[d] = data
        return output
    
    def plot(self,names):
        plot = Plotter(names,self.getData())
        plot.plot(base_file=self.base_path)
    
    def save(self,optim):
        optim_data = self.gatherGPUData(self.benchmarkState.dump())
        if (self.rank == 0 or self.rank == "cuda") and len(optim_data) != 0:
                os.makedirs(f"{ self.base_path}/{optim}")
                with open(f"{self.base_path}/{optim}/benchmark.json", 'wb') as file:
                    pickle.dump(optim_data, file)
                with open(f"{self.base_path}/conf.json", 'w') as file:
                    json.dump(getConfig(), file, indent=4)
                

    """This is necessary if DDP is used. Due to the nature of AdaHessian, we have to use DP which makes this function
        obsolete. If AdaHessian is not required and DDP is used, use this"""
    def gatherGPUData(self,data):
        if self.rank == "cuda":
            return data
        dist.barrier()
        gathered_data = [None for _ in range(self.world_size)]
        dist.gather_object(data, gathered_data if self.rank == 0 else None, dst=0)
        output = {}
        for key in gathered_data[0].keys():
            measure = np.array([d[key] for d in gathered_data])
            output[key] = list(np.mean(measure,axis=0))
        return output




            

