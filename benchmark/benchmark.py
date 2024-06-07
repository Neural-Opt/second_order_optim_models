import time
from benchmark.state import BenchmarkState
 
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
        tps.append(diff)
        self.state.set("tps",tps)
        self.start_time = None
    def addTrainAcc(self,acc):
        acc_train =  self.state.get("acc_train")
        acc_train.append(acc)
        self.state.set("acc_train",acc_train)
        
    def addTestAcc(self,acc):
        acc_test =  self.state.get("acc_test")
        acc_test.append(acc)
        self.state.set("acc_test",acc_test)

    @staticmethod
    def getInstance(state:BenchmarkState):
        if not Benchmark._instance:
            Benchmark._instance = Benchmark(state)
        return Benchmark._instance

""" 
values to track
    time per step
    epochs till convergence
    train_set acc
    test_set acc
    memory consumtion
"""