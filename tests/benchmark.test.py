import unittest
from unittest.mock import MagicMock, patch
import time
import torch
import numpy as np
import sys
import os
sys.path.insert(0, './')
from benchmark.state import BenchmarkState
from benchmark.benchmark import Benchmark
from utils.utils import MeanAggregator

class TestBenchmark(unittest.TestCase):
    def setUp(self):
        self.mock_state = MagicMock(spec=BenchmarkState)
        self.benchmark = Benchmark(self.mock_state)

    def test_step_timing(self):
        self.benchmark.stepStart()
        time.sleep(0.01)  
        self.benchmark.stepEnd()

        self.assertIsNotNone(self.benchmark.averageStepTime)
        self.assertGreater(self.benchmark.averageStepTime.get(), 0)

    def test_add_train_accuracy(self):
        self.benchmark.addTrainAcc(0.95)
        self.mock_state.get.return_value = []
        self.benchmark.addTrainAcc(0.96)

        self.mock_state.set.assert_called_with("acc_train", [ 0.96])

    def test_add_train_loss(self):
        self.benchmark.addTrainLoss(0.1)
        self.mock_state.get.return_value = []
        self.benchmark.addTrainLoss(0.08)

        self.mock_state.set.assert_called_with("train_loss", [ 0.08])

    def test_add_test_accuracy(self):
        self.benchmark.addTestAcc(0.9)
        self.mock_state.get.return_value = []
        self.benchmark.addTestAcc(0.92)


        self.mock_state.set.assert_called_with("acc_test", [ 0.92])

    @patch('torch.cuda.device_count', return_value=1)
    @patch('torch.cuda.max_memory_allocated', return_value=1024**2 * 100) 
    @patch('torch.cuda.reset_peak_memory_stats')
    def test_gpu_memory_usage(self, mock_reset_peak_memory, mock_max_memory_allocated, mock_device_count):
        self.benchmark.measureGPUMemUsageStart(rank=0)
        self.benchmark.measureGPUMemUsageEnd(rank=0)
        self.assertIsNotNone(self.benchmark.averageMemory)
        self.assertAlmostEqual(self.benchmark.averageMemory.get(), 100.0, delta=0.1) 


    def tearDown(self):
        Benchmark.cleanUp()

if __name__ == '__main__':
    unittest.main()
