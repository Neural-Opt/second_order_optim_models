import time
import subprocess
from benchmark.state import BenchmarkState
from utils.utils import MeanAggregator
import numpy as np
class Benchmark:
    _instance = None
    def __init__(self,state: BenchmarkState) -> None:
        self.averageStepTime = None
        self.averageMemory = None
        self.state = state
        pass
    def __getstate__(self) -> object:
        return self.state
    def stepStart(self,):
        self.averageStepTime = MeanAggregator(measure=lambda time:time)
        self.start_time = time.perf_counter()
        pass
    def stepEnd(self,):
        assert self.start_time != None, "Timer has not been started"
        diff =  time.perf_counter()  - self.start_time
        self.averageStepTime(diff)
        self.start_time = None
    def addTrainAcc(self,acc):
        acc_train =  self.state.get("acc_train")
        acc_train = acc_train if acc_train != None else []
        acc_train.append(acc)
        self.state.set("acc_train",acc_train)
    def addTrainLoss(self,loss):
        loss_train =  self.state.get("train_loss")
        loss_train = loss_train if loss_train != None else []
        loss_train.append(loss)

        self.state.set("train_loss",loss_train)
    def addTestAcc(self,acc):
        acc_test =  self.state.get("acc_test")
        acc_test = acc_test if acc_test != None else []
        acc_test.append(acc)
        self.state.set("acc_test",acc_test)

   
    def measureGPUMemUsage(self,rank):
        self.averageMemory =  MeanAggregator(measure=lambda mem:mem) if  self.averageMemory == None else self.averageMemory
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                                    stdout=subprocess.PIPE, text=True)
            
            memory_usages = result.stdout.strip().split('\n')
            memory_usages = [int(mem) for mem in memory_usages]
            
            if rank == "cuda":
                self.averageMemory(np.max(np.array(memory_usages)))
            else:
                self.averageMemory(int(memory_usages[int(rank)]))

        except Exception as e:
 
            return None
 
    def flush(self):
        tps =  self.state.get("tps")
        gpu_mem =  self.state.get("gpu_mem")

        tps = tps if tps != None else []
        gpu_mem = gpu_mem if gpu_mem != None else []

        tps.append(float(self.averageStepTime.get()))
        gpu_mem.append(float(self.averageMemory.get()))

        self.averageStepTime = None
        self.averageMemory = None

        self.state.set("tps",tps)
        self.state.set("gpu_mem",gpu_mem)

    @staticmethod
    def getInstance(state: BenchmarkState):
        if Benchmark._instance  == None:
            Benchmark._instance = Benchmark(state)
        return Benchmark._instance
    @staticmethod
    def cleanUp():
            Benchmark._instance = None

""" 
values to track
    time per step
    epochs till convergence
    train_set acc
    test_set acc
    memory consumtion
"""