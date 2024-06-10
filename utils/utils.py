import psutil

from typing import Callable, Any
import numpy as np


def CPUMemory():
    return psutil.virtual_memory().used / 1024**3
def GPUMemory():
    return psutil.virtual_memory().used / 1024**3


class AverageAggregator:
    def __init__(self,measure:Callable=lambda *args:(args[0].eq(args[1]).sum().item())) -> None:
        self.total = 0
        self.measure = measure
        self.measure_result = np.array([])
        pass
    def __call__(self, *args: Any,total = 0) -> Any:
        self.measure_result = np.append(self.measure_result,[self.measure(*args)])
        if total == 0:
            self.total += total
    def get(self):
        return np.mean(self.measure_result)