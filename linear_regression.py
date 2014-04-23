#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import deque
import random
import numpy as np

class LinearRegression:
    """
    Feeding data:

        r = LinearRegression(order=2, target_index=0)
        r.feed([a0, a1, a2, a3])
        r.feed([b0, b1, b2, b3])
        r.feed([c0, c1, c2, c3])
        r.feed([d0, d1, d2, d3])

    What it does for fitting:

        c0 = linear_combination([b0, b1, b2, b3, a0, a1, a2, a3])
        d0 = linear_combination([c0, c1, c2, c3, b0, b1, b2, b3])

    Getting coefficients:

        r.means
        r.stds
    """
    
    def __init__(self, order=1, trials=100, pick_rate=0.3, target_index=0):
        self.history = deque()
        self.order = order
        self.dimension = None
        self.trials = trials
        self.pick_rate = pick_rate
        self.picks = max(1, int(self.trials * self.pick_rate))
        self.target_index = target_index

    def clear_history(self):
        self.history = deque()

    def set_dimension(self, dimension):
        if self.dimension is not None:
            if self.dimension != dimension:
                raise ValueError("dimension mismatch {0} (given {1})".format(self.dimension, dimension))
            return
        self.dimension = dimension
        self.means = np.zeros((self.order, self.dimension))
        self.stds = np.ones((self.order, self.dimension))

    def set_header(self, header):
        self.set_dimension(len(header))
        self.header = header

    def feed_values(self, values):
        self.set_dimension(len(values))
        self.history.appendleft(values)
        if len(self.history) > self.order + 1:
            self.history.pop()
        self.learn()

    def feed_line(self, line):
        line = line.strip()
        if not line:
            self.clear_history()
            return
        values = [float(x) for x in line.split()]
        self.feed_values(values)

    def feed(self, obj):
        if isinstance(obj, (list, tuple)):
            self.feed_values(obj)
        elif isinstance(obj, str):
            self.feed_line(obj)
        else:
            raise ValueError("input type must be list, tuple or str (given {0}).".format(type(obj)))

    def new_gene(self):
        # XXX: numpy-way for this?
        gene = np.zeros_like(self.means)
        for i in range(len(gene)):
            for j in range(len(gene[i])):
                gene[i][j] = np.random.normal(self.means[i][j], self.stds[i][j])

        return gene

    def learn(self):
        if len(self.history) <= self.order:
            """Not enough data to learn."""
            return

        # Take old data only.
        history = np.array(self.history)[1:]
        value = self.history[0][0]

        ranking = []
        for i in range(self.trials):
            gene = self.new_gene()
            value_hat = np.sum(gene * history)
            error = value_hat - value
            ranking.append((error**2, gene))
        ranking.sort()

        genes = [x[1] for x in ranking[:self.picks]]
        self.means = np.mean(genes, axis=0)
        self.stds = np.std(genes, axis=0)

        min_penalty = ranking[0][0]
        max_penalty = ranking[self.picks-1][0]
        print("With penalty_range: ({0}, {1})".format(min_penalty, max_penalty))
        print()
        print(self.means)
        print(self.stds)
        print()
