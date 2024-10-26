import unittest
import numpy as np
import sys
import os
sys.path.insert(0, './')
from utils.utils import MeanAggregator, VarianceAggregator

class TestMeanAggregator(unittest.TestCase):
    def test_mean_aggregator_basic(self):
        aggregator = MeanAggregator()
        aggregator(5)
        aggregator(10)
        aggregator(15)
        self.assertAlmostEqual(aggregator.get(), 10.0) 

    def test_mean_aggregator_with_measure(self):
        aggregator = MeanAggregator(measure=lambda x: x**2)
        aggregator(2)
        aggregator(3)
        self.assertAlmostEqual(aggregator.get(), (2**2 + 3**2) / 2)  

    def test_mean_aggregator_with_total(self):
        aggregator = MeanAggregator()
        aggregator(5, total=1)
        aggregator(15, total=2)
        self.assertEqual(aggregator.total, 3)  

    def test_mean_aggregator_empty(self):
        aggregator = MeanAggregator()
        self.assertTrue(np.isnan(aggregator.get()))  

class TestVarianceAggregator(unittest.TestCase):
    def test_variance_aggregator_basic(self):
        aggregator = VarianceAggregator()
        aggregator(5)
        aggregator(10)
        aggregator(15)
        self.assertAlmostEqual(aggregator.get(), np.var([5,10,15])) 

    def test_variance_aggregator_with_measure(self):
        # Custom measure function that squares the input
        aggregator = VarianceAggregator(measure=lambda x: x**2)
        aggregator(2)
        aggregator(3)
        self.assertAlmostEqual(aggregator.get(), np.var([4, 9])) 

    def test_variance_aggregator_with_total(self):
        aggregator = VarianceAggregator()
        aggregator(5, total=1)
        aggregator(15, total=2)
        self.assertEqual(aggregator.total, 3)  

    def test_variance_aggregator_empty(self):
        aggregator = VarianceAggregator()
        self.assertTrue(np.isnan(aggregator.get())) 

if __name__ == '__main__':
    unittest.main()
