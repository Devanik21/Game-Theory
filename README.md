# Game Theory

![Language](https://img.shields.io/badge/Language-Python-3776AB?style=flat-square) ![Stars](https://img.shields.io/github/stars/Devanik21/Game-Theory?style=flat-square&color=yellow) ![Forks](https://img.shields.io/github/forks/Devanik21/Game-Theory?style=flat-square&color=blue) ![Author](https://img.shields.io/badge/Author-Devanik21-black?style=flat-square&logo=github) ![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)

> Interactive game theory — implement, simulate, and visualise strategic equilibria, evolutionary dynamics, and mechanism design from first principles.

---

**Topics:** `reinforcement-learning` · `auction-theory` · `cooperative-game-theory` · `evolutionary-game-theory` · `mechanism-design` · `multi-agent-systems` · `nash-equilibrium` · `social-choice-theory` · `strategic-reasoning` · `zero-sum-games`

## Overview

This Game Theory project provides a computational exploration of strategic interaction: the mathematical
study of rational decision-making among multiple agents whose outcomes depend on each other's choices.
It implements the core solution concepts — Nash Equilibrium, Pareto Optimality, dominant strategy
identification, minimax — alongside a suite of classic games (Prisoner's Dilemma, Battle of the Sexes,
Stag Hunt, Public Goods, Ultimatum, Auction mechanisms) with simulation, visualisation, and analysis tools.

The project covers four progressively advanced areas. Strategic form games provide a matrix-based
framework for simultaneous-move games with Nash Equilibrium computation via support enumeration and
the Lemke-Howson algorithm. Extensive form games use tree representations for sequential games with
backward induction and Subgame Perfect Equilibrium identification. Evolutionary game theory implements
replicator dynamics and evolutionary stable strategy (ESS) computation for population-level strategic
evolution. Mechanism design covers auction theory (first-price, second-price, VCG) with revenue
equivalence theorem verification through simulation.

All solution concepts are implemented with both analytical solvers and Monte Carlo simulation verification —
allowing students to build intuition by simulating thousands of games before engaging with the formal
equilibrium mathematics.

---

## Motivation

Game theory is one of the most powerful and broadly applicable frameworks in social science and
computer science — from algorithmic mechanism design in internet advertising to evolutionary biology
to AI multi-agent systems. Yet most introductory treatments remain abstract and algebraic, providing
little computational intuition. This project was built to make game theory viscerally interactive:
simulate the Prisoner's Dilemma 10,000 times, watch replicator dynamics evolve toward ESS in real time,
and verify the Revenue Equivalence Theorem numerically before accepting it analytically.

---

## Architecture

```
Game Specification (players, strategies, payoffs)
        │
  ┌──────────────────────────────────────────────┐
  │  Strategic Form Engine:                      │
  │  Nash Equilibrium (support enumeration, L-H) │
  │  Dominated strategy elimination (IESDS)      │
  └──────────────────────────────────────────────┘
        │
  ┌──────────────────────────────────────────────┐
  │  Extensive Form Engine:                      │
  │  Backward induction, SPE computation         │
  └──────────────────────────────────────────────┘
        │
  ┌──────────────────────────────────────────────┐
  │  Evolutionary Dynamics:                      │
  │  Replicator dynamics ODE, ESS identification │
  └──────────────────────────────────────────────┘
        │
  Monte Carlo simulation verification
        │
  Streamlit / Matplotlib interactive visualisation
```

---

## Features

### Nash Equilibrium Solver
Exact Nash Equilibrium computation for 2-player strategic form games via support enumeration and the Lemke-Howson algorithm, handling both pure and mixed strategy equilibria.

### Classic Game Library
Pre-built implementations of 15+ classic games: Prisoner's Dilemma, Stag Hunt, Battle of the Sexes, Chicken, Hawk-Dove, Rock-Paper-Scissors, Matching Pennies, Coordination, Public Goods, and more.

### Iterated Elimination of Dominated Strategies
Step-by-step IESDS algorithm with visual highlighting of eliminated strategies, demonstrating the rationalizability concept interactively.

### Extensive Form Game Trees
Graphical representation of sequential games with information sets, backward induction visualisation, and Subgame Perfect Equilibrium path highlighting.

### Replicator Dynamics Simulation
Evolutionary game dynamics visualised as phase portraits (2-strategy simplex plots and 3-strategy simplex for symmetric games), showing convergence to evolutionary stable strategies.

### Auction Mechanism Simulator
First-price sealed-bid and second-price (Vickrey) auction simulations with configurable number of bidders and value distribution, verifying the Revenue Equivalence Theorem statistically.

### Monte Carlo Strategy Simulator
Run any game 10,000+ times with configurable strategy profiles to build empirical payoff distributions and verify analytical results computationally.

### Interactive Payoff Matrix Editor
Browser-based payoff matrix editor where students can input any 2×N strategic form game and immediately solve for Nash Equilibria and Pareto optimal outcomes.

---

## Tech Stack

| Library / Tool | Role | Why This Choice |
|---|---|---|
| **NumPy** | Linear algebra | Payoff matrix operations, mixed strategy probability vectors |
| **SciPy** | ODE solving / optimisation | Replicator dynamics integration, Nash computation |
| **NetworkX** | Extensive form trees | Game tree construction and traversal for backward induction |
| **Matplotlib / Plotly** | Visualisation | Phase portraits, simplex plots, auction revenue charts |
| **Streamlit** | Interactive interface | Payoff editor, game selector, simulation controls |
| **pandas** | Results analysis | Monte Carlo simulation result aggregation |

---

## Getting Started

### Prerequisites

- Python 3.9+ (or Node.js 18+ for TypeScript/JavaScript projects)
- A virtual environment manager (`venv`, `conda`, or equivalent)
- API keys as listed in the Configuration section

### Installation

```bash
git clone https://github.com/Devanik21/Game-Theory.git
cd Game-Theory
python -m venv venv && source venv/bin/activate
pip install numpy scipy networkx matplotlib plotly streamlit pandas
streamlit run app.py
```

---

## Usage

```bash
# Launch interactive platform
streamlit run app.py

# Solve a custom game
python solve.py --game prisoner_dilemma --method nash
python solve.py --payoff_matrix '[[3,0],[5,1]],[[3,5],[0,1]]'

# Run evolutionary dynamics
python evolution.py --game hawk_dove --population 1000 --generations 500

# Simulate auction mechanisms
python auction.py --type second_price --bidders 10 --trials 10000
```

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `DEFAULT_GAME` | `prisoner_dilemma` | Default game loaded on startup |
| `MONTE_CARLO_TRIALS` | `10000` | Default number of simulation trials |
| `EVO_POPULATION` | `1000` | Default evolutionary dynamics population size |
| `AUCTION_BIDDERS` | `5` | Default number of auction participants |

> Copy `.env.example` to `.env` and populate required values before running.

---

## Project Structure

```
Game-Theory/
├── README.md
├── requirements.txt
├── app.py
└── ...
```

---

## Roadmap

- [ ] Cooperative game theory: Shapley value computation for coalition games
- [ ] Network games: local interaction games on arbitrary graph structures
- [ ] AI agent training via reinforcement learning to find Nash Equilibria in complex games
- [ ] Behavioural game theory: prospect theory and bounded rationality variants
- [ ] Tournament mode: run different strategy agents (tit-for-tat, grim trigger, Pavlov) in repeated game competitions

---

## Contributing

Contributions, issues, and suggestions are welcome.

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-idea`
3. Commit your changes: `git commit -m 'feat: add your idea'`
4. Push to your branch: `git push origin feature/your-idea`
5. Open a Pull Request with a clear description

Please follow conventional commit messages and add documentation for new features.

---

## Notes

Nash Equilibrium computation is NP-hard in general for games with more than 2 players. The implemented algorithms are exact for 2-player games but may require approximation methods for larger games. The platform is designed for educational exploration, not for production mechanism design deployments.

---

## Author

**Devanik Debnath**  
B.Tech, Electronics & Communication Engineering  
National Institute of Technology Agartala

[![GitHub](https://img.shields.io/badge/GitHub-Devanik21-black?style=flat-square&logo=github)](https://github.com/Devanik21)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-devanik-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/devanik/)

---

## License

This project is open source and available under the [MIT License](LICENSE).

---

*Built with curiosity, depth, and care — because good projects deserve good documentation.*
