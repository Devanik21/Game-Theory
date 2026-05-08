# Myerson's Optimal Auction

Myerson's optimal auction design addresses the problem of maximizing the seller's expected revenue in an auction with risk-neutral bidders whose valuations are drawn from independent distributions.

## Key Features

- **Virtual Valuations:** The optimal auction allocates the item to the bidder with the highest *virtual valuation*, provided it is positive. The virtual valuation for a bidder with true value $v$ is given by $v - \frac{1 - F(v)}{f(v)}$, where $F$ is the cumulative distribution function and $f$ is the probability density function of the bidder's valuation.
- **Reserve Price:** The optimal auction includes a reserve price, which is determined by the point where the virtual valuation is zero.
- **Revenue Equivalence:** Myerson's theory relies on the Revenue Equivalence Theorem, showing that any two mechanisms that result in the same allocation rule and give zero expected utility to the lowest type will yield the same expected revenue for the seller.
