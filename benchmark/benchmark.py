import time
import subprocess
from benchmark.postprocess import PostProcessor
from benchmark.state import BenchmarkState
from utils.utils import MeanAggregator
import numpy as np
import torch
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
        self.averageStepTime = MeanAggregator(measure=lambda time:time) if  self.averageStepTime == None else  self.averageStepTime
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
    def add(self,key,value):
        v = self.state.get(key)
        v = v if v != None else []
        v.append(value)
        self.state.set(key,v)
    def postProcess(self):
        PostProcessor(self.state)
                
    def measureGPUMemUsageStart(self,rank):
        self.averageMemory =  MeanAggregator(measure=lambda mem:mem) if  self.averageMemory == None else  self.averageMemory
        for i in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(device=f'cuda:{i}')
    def measureGPUMemUsageEnd(self,rank):
        memory_allocated = []
        for i in range(torch.cuda.device_count()):
            memory_allocated.append(torch.cuda.max_memory_allocated(device=f'cuda:{i}')/1024**2)
        self.averageMemory(np.mean(np.array(memory_allocated)))
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