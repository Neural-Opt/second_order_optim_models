import psutil

from typing import Callable, Any
import numpy as np
import os

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
    
class BenchmarkAnalyzer:
    @staticmethod
    def getRunCount(dir):
        return sum(os.path.isdir(os.path.join(f"./runs/{dir}", d)) for d in os.listdir(f"./runs/{dir}")) 
    @staticmethod
    def getAllData(dir,dirCount,optim):
        state_collector = BenchmarkState(f"./runs/{dir}/{dirCount}/{optim}/benchmark.json").dump()
        for _ in range(2,dirCount+1):
            state = BenchmarkState(f"./runs/{dir}/{dirCount}/{optim}/benchmark.json")
            for key in state_collector.keys():
                state_collector[key] = state_collector[key]+state[key]
        return state_collector
    @staticmethod
    def var(setName,optim):
        joined = BenchmarkAnalyzer.getAllData(setName,BenchmarkAnalyzer.getRunCount(setName),optim=optim)
        for key in joined.keys():
            joined[key] =  np.var(np.array(joined[key]))
        return joined
    @staticmethod
    def mean(setName,optim):
        joined = BenchmarkAnalyzer.getAllData(setName,BenchmarkAnalyzer.getRunCount(setName),optim=optim)
        for key in joined.keys():
            joined[key] =  np.mean(np.array(joined[key]))
        return joined
    @staticmethod
    def std(setName,optim):
        joined = BenchmarkAnalyzer.getAllData(setName,BenchmarkAnalyzer.getRunCount(setName),optim=optim)
        for key in joined.keys():
            joined[key] =  np.std(np.array(joined[key]))
        return joined
