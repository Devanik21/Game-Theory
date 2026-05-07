import numpy as np

class PrisonersDilemma:
    def __init__(self, cc=3, cd=0, dc=5, dd=1):
        self.payoffs = np.array([
            [[cc, cc], [cd, dc]],
            [[dc, cd], [dd, dd]]
        ])
        self.strategies = ["Cooperate", "Defect"]
