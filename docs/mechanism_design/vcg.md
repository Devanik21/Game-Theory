# Vickrey-Clarke-Groves (VCG) Mechanism

The VCG mechanism is a generalization of the Vickrey auction for multiple items. It is a direct quasilinear mechanism that achieves an efficient outcome (maximizes social welfare) while making truthful reporting a dominant strategy for all agents.

## Core Concepts

1.  **Efficient Allocation:** The mechanism chooses the allocation that maximizes the total reported value of all agents.
2.  **Payment Rule:** Each agent pays the "opportunity cost" they impose on the other agents. Specifically, the payment is equal to the total value that the other agents *would have had* if the agent were not present, minus the total value the other agents *actually* have with the agent present.

## Properties

-   **Dominant Strategy Incentive Compatible (DSIC):** Truth-telling is a dominant strategy.
-   **Allocative Efficiency:** Maximizes the sum of agents' true valuations.
-   **Not always budget-balanced:** The mechanism may run a deficit.
