import psutil

from typing import Callable, Any
import numpy as np
import os

from benchmark.postprocess import PostProcessor
from benchmark.state import BenchmarkState

def CPUMemory():
    return psutil.virtual_memory().used / 1024**3
def GPUMemory():
    return psutil.virtual_memory().used / 1024**3

class MeanAggregator:
    def __init__(self,measure:Callable=lambda x:x) -> None:
        self.total = 0
        self.measure = measure
        self.measure_result = np.array([])
        pass
    def __call__(self, *args: Any,total = 0) -> Any:
        self.measure_result = np.append(self.measure_result,[self.measure(*args)])
        if total != 0:
            self.total += total
        return np.mean(self.measure_result)
    def get(self):
        return np.mean(self.measure_result)

class VarianceAggregator:
    def __init__(self,measure:Callable=lambda x:x) -> None:
        self.total = 0
        self.measure = measure
        self.measure_result = np.array([])
        pass
    def __call__(self, *args: Any,total = 0) -> Any:
        self.measure_result = np.append(self.measure_result,[self.measure(*args)])
        if total != 0:
            self.total += total
    def get(self):
        return np.var(self.measure_result)
    

"""
a = [q]
b = [b]
a = a.append(b)
a == [[q],[b]]
"""
class BenchmarkAnalyzer:
    @staticmethod
    def getRunCount(dir):
        return sum(os.path.isdir(os.path.join(f"./runs/{dir}", d)) for d in os.listdir(f"./runs/{dir}")) 
    @staticmethod
    def getAllData(dir,dirCount,optim,join=False):
        state_collector = BenchmarkState(f"./runs/{dir}/{dirCount}/{optim}/benchmark.json").dump()
        state_collector = {key: [] for key in state_collector}
        state_collector["ttc"] = []
        for _ in range(1,dirCount+1):
            state = BenchmarkState(f"./runs/{dir}/{dirCount}/{optim}/benchmark.json")
            PostProcessor(state)
            for key in state.dump().keys():
                if join:
                    state_collector[key] = state_collector[key] + state[key]
                else:
                    state_collector[key].append(state[key])
        return {key: np.array(state_collector[key]) for key in state_collector}

    @staticmethod
    def var(setName,optim,join=False,reducer=lambda x:x):
        joined = BenchmarkAnalyzer.getAllData(setName,BenchmarkAnalyzer.getRunCount(setName),optim=optim,join=join)
        for key in joined.keys():
            joined[key] =  np.var(reducer(joined[key]),axis=0 )
        return joined
    @staticmethod
    def mean(setName,optim,join=False,reducer=lambda x:x):
        joined = BenchmarkAnalyzer.getAllData(setName,BenchmarkAnalyzer.getRunCount(setName),optim=optim,join=join)
        for key in joined.keys():
            joined[key] =  np.mean(reducer(joined[key]),axis=0 )
        return joined
    @staticmethod
    def std(setName,optim,join=False,reducer=lambda x:x):
        joined = BenchmarkAnalyzer.getAllData(setName,BenchmarkAnalyzer.getRunCount(setName),optim=optim,join=join)
        for key in joined.keys():
            joined[key] =  np.std(reducer(joined[key]),axis=0 )
        return joined
    
    @staticmethod
    def merge(filePaths,mergePath,merge_override = True):
        out = {}
        for path in filePaths:
            for _, dirnames, _ in os.walk(path):
                for optim in dirnames:
                    if not merge_override and optim in out:
                        raise Exception("merge_override is False")
                    out[optim] = BenchmarkState(f"{filePaths}/{optim}/benchmark.json")
        for optim in out.keys():
            state = out[optim]
            state.setup(f"{mergePath}/{optim}/benchmark.json")
            state.save()