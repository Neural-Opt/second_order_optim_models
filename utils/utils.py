from typing import Callable, Any


class AverageAggregator:
    def __init__(self,measure:Callable=lambda *args:(args[0].eq(args[1]).sum().item())) -> None:
        self.total = 0
        self.measure = measure
        self.measure_result = 0
        pass
    def __call__(self, *args: Any,) -> Any:
        all = args[1]
        self.measure_result += self.measure(*args)
        self.total += all.size(0)
        pass
    def get(self):
        if self.total == 0:
            return 0
        return self.measure_result / self.total