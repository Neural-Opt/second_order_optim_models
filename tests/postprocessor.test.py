import unittest
from unittest.mock import MagicMock
import numpy as np
import sys
import os
sys.path.insert(0, './')
from benchmark.postprocess import PostProcessor
from benchmark.state import BenchmarkState

class TestPostProcessor(unittest.TestCase):
    def setUp(self):
        self.mock_state = MagicMock(spec=BenchmarkState)
        self.mock_state.__getitem__.return_value =  np.array([10, 9, 8, 5, 3, 2.5, 2.1, 2, 1.9, 1.8, 1.75, 1.7]) 
        self.mock_state.dump.return_value = self.mock_state.__getitem__.return_value

    def test_calcTTC(self):
        processor = PostProcessor(self.mock_state)

        expected_ttc_index = 5
        self.mock_state.__setitem__.assert_called_with("ttc", [expected_ttc_index])

    def test_calcTTC_no_convergence(self):
        self.mock_state.__getitem__.return_value = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
        self.mock_state.dump.return_value = self.mock_state.__getitem__.return_value
        processor = PostProcessor(self.mock_state)

        self.mock_state.__setitem__.assert_called()  
        args, _ = self.mock_state.__setitem__.call_args
        self.assertEqual(args[0], "ttc")
        self.assertLess(args[1][0], len(self.mock_state['train_loss']))  
if __name__ == '__main__':
    unittest.main()
