import time
from benchmark.state import BenchmarkState
from utils.utils import CPUMemory
class Benchmark:
    _instance = None
    def __init__(self,state:BenchmarkState) -> None:
        self.state = state
        pass
    def __getstate__(self) -> object:
        return self.state
    def stepStart(self,):
        self.start_time = time.perf_counter()
        pass
    def stepEnd(self,):
        assert(self.start_time != None, "Timer has not been started")
        diff =  time.perf_counter()  - self.start_time
        tps =  self.state.get("tps")
        tps = tps if tps != None else []
        tps.append(diff)
        self.state.set("tps",tps)
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

    def measureGPUMemUsage(self,):
        gpu_mem =  self.state.get("gpu_mem")
        gpu_mem = gpu_mem if gpu_mem != None else []
        gpu_mem.append(CPUMemory())
        self.state.set("gpu_mem",gpu_mem)
        
    @staticmethod
    def getInstance(state:BenchmarkState):
        if not Benchmark._instance:
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