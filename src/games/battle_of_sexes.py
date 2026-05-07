import numpy as np

class BattleOfSexes:
    def __init__(self):
        self.payoffs = np.array([
            [[2, 1], [0, 0]],
            [[0, 0], [1, 2]]
        ])
        self.strategies = ["Opera", "Football"]
