from benchmark.state import BenchmarkState
import numpy as np

class PostProcessor:
    def __init__(self,state:BenchmarkState) -> None:
        self.state = state
        self.calcTTC()
    def calcTTC(self):
        data = self.state['train_loss']
        diff = np.diff(data)
        std = np.std(diff)
        print(std)
        for i in range(len(data)-1,0,-1):
            min, max = np.min(data[i:len(data)]), np.max(data[i:len(data)])
            if abs(max-min) < 0.25*std:
                continue
            elif i > 0:
                self.state["ttc"]=i+1
                break
            else:
                self.state["ttc"]= -1
                break
        