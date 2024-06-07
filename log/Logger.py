import os
import pickle
from config.loader import getConfig
from benchmark.benchmark import Benchmark,BenchmarkState
def createNewRun(base_dir):
    if not os.path.isdir(base_dir):
        os.makedirs(base_dir)
    dir_count = sum(os.path.isdir(os.path.join(base_dir, d)) for d in os.listdir(base_dir)) 
    os.makedirs( os.path.join(base_dir,  f"{dir_count + 1}"))
    return os.path.join(base_dir,  f"{dir_count + 1}")
    
class Logger:
    def __init__(self,test_set) -> None:
        conf = getConfig()
        self.test_name = test_set
        self.base = f"{conf["runs"]["dir"]}/{test_set}"
        self.currRun = createNewRun(self.base)

    def setup(self,optim):
        os.makedirs(f"{ self.currRun}/{optim}")
        self.benchmarkState = BenchmarkState(f"{self.currRun}/{optim}/benchmark.json")
        self.benchmark = Benchmark.getInstance(self.benchmarkState)

    def trash(self,):
        Benchmark.cleanUp()

    def getData(self,number_of_trial = 1):
        base_dir = f"{self.base}/{number_of_trial}"
        output = {}
        for d in os.listdir(base_dir):
            optim = os.path.dirname(d)
            state = BenchmarkState(f"{os.path.join(base_dir, d)}/benchmark.json")
            state.load()
            data = state.dump()
            output[optim] = data

        return output
            

