import pytest
from src.mechanisms.auctions import VickreyAuction
from src.mechanisms.vcg import vcg_mechanism

def test_vickrey_auction():
    auction = VickreyAuction(bidders=[1, 2, 3], items=["item1"])
    bids = {1: 100, 2: 80, 3: 50}
    winner, price = auction.resolve(bids)
    assert winner == 1
    assert price == 80

def test_vcg_stub():
    assert vcg_mechanism([], set(), lambda x: 0) is None
