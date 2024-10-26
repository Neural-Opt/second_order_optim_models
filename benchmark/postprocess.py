from benchmark.state import BenchmarkState
import numpy as np

class PostProcessor:
    def __init__(self,state:BenchmarkState) -> None:
        self.state = state
        self.calcTTC()
    def calcTTC(self):
        data = self.state['train_loss']
        ref = data[0]/100

        sigma = 5
        old_mean = np.mean(data[len(data)-sigma:])
        for i in range(len(data)-sigma,0,-1):
            new_mean = np.mean(data[i:i+sigma])
            if abs(new_mean-old_mean) < ref:
                continue
            else:
                self.state["ttc"]=[i+np.argmin(new_mean)]
                break

