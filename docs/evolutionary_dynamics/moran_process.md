# Moran Process

The Moran process is a stochastic model in population genetics used to describe the evolution of a population with a constant size over time. It is a fundamental model for studying genetic drift and selection in finite populations.

## Mechanism

Consider a population of constant size $N$. At each discrete time step:
1.  **Reproduction:** One individual is chosen to reproduce, with a probability proportional to its fitness.
2.  **Death:** One individual is chosen uniformly at random to die and be replaced by the offspring from the reproduction step.

This ensures the population size remains strictly $N$.

## Fixation Probability

A key metric studied in the Moran process is the **fixation probability**—the probability that a newly introduced mutant allele will eventually spread to the entire population. For a mutant with relative fitness $r$ (compared to a resident fitness of $1$), the probability of fixation starting from 1 mutant is approximately:

$$ \rho = \frac{1 - 1/r}{1 - 1/r^N} $$
