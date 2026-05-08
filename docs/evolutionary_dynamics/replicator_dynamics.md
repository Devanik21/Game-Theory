# Replicator Dynamics

Replicator dynamics is a fundamental continuous-time model of evolution in population biology and evolutionary game theory. It describes how the frequencies of different strategies in a population change over time.

## The Equation

Let $x_i$ be the proportion of the population playing strategy $i$. The replicator equation is:

$$ \frac{dx_i}{dt} = x_i (f_i(x) - \bar{f}(x)) $$

Where:
- $f_i(x)$ is the fitness (expected payoff) of strategy $i$ given the current population state $x$.
- $\bar{f}(x) = \sum_j x_j f_j(x)$ is the average fitness of the entire population.

## Intuition

The equation dictates that a strategy's share of the population grows if its fitness is strictly greater than the average fitness of the population, and it shrinks if its fitness is strictly less than the average.
