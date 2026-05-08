import pytest
from src.solvers.lemke_howson import lemke_howson
from src.solvers.support_enumeration import support_enumeration

def test_lemke_howson_stub():
    assert lemke_howson(None, None) is None

def test_support_enumeration_stub():
    assert support_enumeration(None) is None
