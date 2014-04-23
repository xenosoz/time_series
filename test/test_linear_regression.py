#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
from linear_regression import LinearRegression

class TestLinearRegression(unittest.TestCase):
    def setUp(self):
        self.r = LinearRegression(order=2)

    def test_feeding_data(self):
        r = LinearRegression(order=2, target_index=0)
        r.feed([0, 0, 0, 0])
        r.feed([1, 1, 0, 0])
        r.feed([2, 0, 1, 0])
        r.feed([4, 0, 0, 1])
