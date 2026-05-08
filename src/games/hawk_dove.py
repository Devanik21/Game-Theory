import numpy as np

class HawkDove:
    def __init__(self, v=10, c=15):
        self.payoffs = np.array([
            [(v-c)/2, v],
            [0, v/2]
        ])
        self.strategies = ["Hawk", "Dove"]
