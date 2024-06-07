import os
import pickle
from config.loader import getConfig
from benchmark.benchmark import Benchmark,BenchmarkState
def createNewRun(base_dir):
    dir_count = sum(os.path.isdir(os.path.join(base_dir, d)) for d in os.listdir(base_dir)) 
    os.makedirs( os.path.join(base_dir,  f"{dir_count + 1}"))
    return os.path.join(base_dir,  f"{dir_count + 1}")
    
class Logger:
    def __init__(self,test_set) -> None:
        conf = getConfig()
        self.base = f"{conf["runs"]["dir"]}/{test_set}"
        self.currRun = createNewRun(self.base)
        self.benchmarkState = BenchmarkState(f"{self.currRun}/benchmark.json")
        self.benchmark = Benchmark.getInstance(self.benchmarkState)


        


