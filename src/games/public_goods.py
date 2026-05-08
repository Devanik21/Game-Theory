class PublicGoodsGame:
    def __init__(self, num_players, endowment, multiplier):
        self.num_players = num_players
        self.endowment = endowment
        self.multiplier = multiplier

    def get_payoffs(self, contributions):
        if len(contributions) != self.num_players:
            raise ValueError(f"Expected {self.num_players} contributions, got {len(contributions)}")
        total_pot = sum(contributions) * self.multiplier
        return [self.endowment - c + (total_pot / self.num_players) for c in contributions]
