#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import deque
import random
import numpy as np
from operator import itemgetter

class LinearRegression:
    """
    Feeding data:

        r = LinearRegression(order=2, target_index=0)
        r.feed([a0, a1, a2, a3])
        r.feed([b0, b1, b2, b3])
        r.feed([c0, c1, c2, c3])
        r.feed([d0, d1, d2, d3])

    Fit:

        r.fit()

    What it does for fitting:

        c0 = linear_combination([b0, b1, b2, b3, a0, a1, a2, a3])
        d0 = linear_combination([c0, c1, c2, c3, b0, b1, b2, b3])

    Getting coefficients:

        r.means
        r.stds
        r.penalty_range
        r.ranking and r.ranking[0]  # r.ranking may be an empty list.
    """
    
    def __init__(self, order=1, pool_size=100, pick_rate=0.3, trials=100, target_index=0, noise=1e-12, means=None, stds=None, verbose=False):
        self.history = []
        self.order = order
        self.dimension = None
        self.pool_size = 100
        self.pick_rate = pick_rate
        self.trials = trials
        self.target_index = target_index
        self.verbose = verbose
        self.noise = noise
        self.ranking = []

        if means is None:
            self.means = None
        else:
            # XXX: has numpy way?
            self.means = np.array([means] * self.order)

        if stds is None:
            self.stds = None
        else:
            # XXX: has numpy way?
            self.stds = np.array([stds] * self.order)

    def clear_history(self):
        self.history = []

    def set_dimension(self, dimension):
        if self.dimension is not None:
            if self.dimension != dimension:
                raise ValueError("dimension mismatch {0} (given {1})".format(self.dimension, dimension))
            return
        self.dimension = dimension
        if self.means is None:
            self.means = np.zeros((self.order, self.dimension))
        if self.stds is None:
            self.stds = np.ones((self.order, self.dimension))

    def set_header(self, header):
        self.set_dimension(len(header))
        self.header = header

    def feed_values(self, values):
        self.set_dimension(len(values))
        self.history.append(values)

    def feed_line(self, line):
        line = line.strip()
        if not line:
            self.clear_history()
            return
        values = [float(x) for x in line.split()]
        self.feed_values(values)

    def feed(self, obj):
        if isinstance(obj, (list, tuple, np.ndarray)):
            self.feed_values(obj)
        elif isinstance(obj, str):
            self.feed_line(obj)
        else:
            raise ValueError("input type must be list, tuple, np.ndarray or str (given {0}).".format(type(obj)))

    def new_gene(self):
        # XXX: numpy-way for this?
        gene = np.zeros_like(self.means)
        for i in range(len(gene)):
            for j in range(len(gene[i])):
                gene[i][j] = np.random.normal(self.means[i][j], self.stds[i][j])

        return gene

    def penalty(self, gene):
        sum_of_penalty = 0
        for offset in range(len(self.history) - self.order):
            partial_data = np.array(self.history[offset:offset+self.order])
            target_value = self.history[offset+1][self.target_index]

            value_hat = np.sum(gene * partial_data)
            error = value_hat - target_value
            sum_of_penalty += error**2

        return sum_of_penalty

    def learn(self):
        if len(self.history) <= self.order:
            raise ValueError("Need more than {0} rows to fit (given {1}).".format(self.order, len(self.history)))

        # Reevalutate alive genes.
        for i, (_, gene) in enumerate(self.ranking):
            self.ranking[i] = (self.penalty(gene), gene)

        # Add new random genes.
        for i in range(self.pool_size - len(self.ranking)):
            gene = self.new_gene()
            self.ranking.append((self.penalty(gene), gene))
        self.ranking.sort(key=itemgetter(0))

        picks = max(1, int(len(self.ranking) * self.pick_rate))
        self.ranking = self.ranking[:picks]

        genes = [x[1] for x in self.ranking]
        self.means = np.mean(genes, axis=0)
        self.stds = np.std(genes, axis=0)
        self.stds += self.noise

        self.penalty_range = (self.ranking[0][0], self.ranking[-1][0])

        if self.verbose:
            print("With penalty_range: {0}".format(self.penalty_range))
            print()
            print(self.means)
            print(self.stds)
            print()

    def fit(self):
        for i in range(self.trials):
            self.learn()

if __name__ == '__main__':
    # XXX: quick and dirty.

    import sys
    r = LinearRegression(order=4, target_index=0, trials=100, verbose=True)
    line = sys.stdin.readline()
    while line:
        r.feed(line)
        line = sys.stdin.readline()

    r.fit()
    print(r.ranking[0])
