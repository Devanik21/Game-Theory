class Auction:
    def __init__(self, bidders, items):
        self.bidders = bidders
        self.items = items

    def resolve(self, bids):
        """Base method to resolve an auction."""
        raise NotImplementedError

class VickreyAuction(Auction):
    def resolve(self, bids):
        """Resolves a single-item second-price auction."""
        # Filter bids to only include authorized bidders
        valid_bids = {k: v for k, v in bids.items() if k in self.bidders}
        if not valid_bids:
            return None, 0
        
        sorted_bids = sorted(valid_bids.items(), key=lambda x: x[1], reverse=True)
        winner = sorted_bids[0][0]
        price = sorted_bids[1][1] if len(sorted_bids) > 1 else 0
        return winner, price
