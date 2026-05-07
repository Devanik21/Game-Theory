import pytest
import numpy as np
from src.games.prisoners_dilemma import PrisonersDilemma

def test_prisoners_dilemma_payoffs():
    pd = PrisonersDilemma()
    assert pd.payoffs.shape == (2, 2, 2)
    assert pd.payoffs[0, 0, 0] == 3
    assert pd.payoffs[1, 1, 1] == 1
