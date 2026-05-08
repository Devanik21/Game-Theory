# Arrow-d'Aspremont-Gérard-Varet (AGV) Mechanism

The AGV mechanism (also known as the "expected externality mechanism") is a prominent mechanism design tool developed by Kenneth Arrow, Claude d'Aspremont, and Louis-André Gérard-Varet. It is designed to overcome a key limitation of the VCG mechanism: the lack of budget balance.

## Main Properties

1. **Efficiency:** The AGV mechanism implements the efficient allocation (maximizes total social welfare).
2. **Budget Balance:** The mechanism is strictly budget-balanced. The sum of all transfers to/from the mechanism designer is exactly zero.
3. **Bayesian Incentive Compatibility (BIC):** Unlike VCG which is Dominant Strategy Incentive Compatible, AGV is only Bayesian Incentive Compatible. This means it's optimal for each agent to report their true type *assuming* all other agents also report truthfully, and given the common knowledge prior distribution of types.

## How it works

The core idea is to charge each agent an amount based on the *expected* externality they impose on others, where the expectation is taken over the prior distribution of other agents' types. This differs from VCG, which charges based on the *realized* externality.
