#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from linear_regression import LinearRegression
import unittest
import numpy as np

class TestLinearRegression(unittest.TestCase):
    def setUp(self):
        self.r = LinearRegression(order=2)

    def test_feeding_data(self):
        r = LinearRegression(order=2, target_index=0)
        r.feed([0, 0, 0, 0])
        r.feed([1, 1, 0, 0])
        r.feed([2, 0, 1, 0])
        r.feed([4, 0, 0, 1])

    def test_simple_regression(self):
        r = LinearRegression(order=1, stds=[100, 100, 100, 1000])
        coeffs = np.array([1, 10, 100, 1000])
        old_value = np.array([0, 0, 0, 0])
        for i in range(100):
            value = np.random.uniform(-10, 10, 4)
            value[0] = np.sum(old_value * coeffs)
            old_value = value
            r.feed(value)

        print("With penalty_range: {0}".format(r.penalty_range))
        print()
        print(r.means)
        print(r.stds)
        print()
