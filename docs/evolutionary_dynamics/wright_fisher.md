# Wright-Fisher Model

The Wright-Fisher model is another foundational stochastic model in population genetics. Unlike the Moran process which models overlapping generations (one birth/death at a time), the Wright-Fisher model assumes discrete, non-overlapping generations.

## Mechanism

Consider a population of constant size $N$ (or $2N$ chromosomes in a diploid population). To form the next generation:
1.  The adults produce an effectively infinite pool of gametes.
2.  The next generation of size $N$ is formed by drawing individuals independently and randomly (with replacement) from this gamete pool.

If there are two alleles, A and a, with allele A having a frequency $p$ in the current generation, the number of 'A' alleles in the next generation follows a binomial distribution:

$$ P(X = k) = \binom{2N}{k} p^k (1-p)^{2N-k} $$

## Genetic Drift

The Wright-Fisher model is the classic framework for illustrating genetic drift—the random fluctuations in allele frequencies from generation to generation purely due to sampling error.
