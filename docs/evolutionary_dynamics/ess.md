# Evolutionarily Stable Strategy (ESS)

An Evolutionarily Stable Strategy (ESS) is a strategy that, if adopted by a population, cannot be invaded by any alternative strategy that is initially rare. It is a refinement of the Nash Equilibrium concept for biological populations.

## Formal Definition

Consider a symmetric two-player game with a payoff function $E(S, T)$ representing the payoff to strategy $S$ playing against strategy $T$.
A strategy $S^*$ is an ESS if, for every mutant strategy $T \neq S^*$, one of the following conditions holds:

1.  **Strictly higher payoff against itself:** $E(S^*, S^*) > E(T, S^*)$
2.  **Equal payoff against itself, but better against the mutant:** $E(S^*, S^*) = E(T, S^*)$ AND $E(S^*, T) > E(T, T)$

## Interpretation

The first condition says that the resident strategy $S^*$ is a strict best response to itself. If a small group of mutants $T$ appear, they will encounter mostly $S^*$ individuals and do strictly worse, thus dying out.
The second condition covers the case where mutants do just as well against the residents as the residents do against themselves. In this case, for $S^*$ to be stable, it must perform better when encountering the rare mutants than the mutants do when encountering each other.
