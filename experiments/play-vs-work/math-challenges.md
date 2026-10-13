# Math Challenges

Thirty multi-step problems intended to require sustained reasoning ("work") from a model.
Each has a single exact answer, listed in the [answer key](#answer-key) at the bottom so
the problems can be given to a model without the answers.

## Problems

### Number theory

1. How many ordered triples $(a, b, c)$ of positive integers satisfy $abc = 10^6$?
2. Find the last three digits of $7^{7^7}$.
3. How many integers $n$ with $1 \le n \le 10{,}000$ satisfy $65 \mid n^2 + 1$?
4. What is the smallest positive integer $n$ such that $n!$ ends in exactly 100 zeros?
5. Find the remainder when $2^{2025}$ is divided by $1000$.
6. How many ordered pairs of integers $(x, y)$ satisfy $x^2 + y^2 = 5^6 \cdot 13^2$?
7. What is the largest prime factor of $2^{32} - 1$?
8. How many integers $n$ with $2 \le n \le 10^6$ are perfect powers, i.e. of the form $m^k$ for integers $m \ge 2$, $k \ge 2$?
9. How many primes are less than $100{,}000$?

### Combinatorics

10. How many lattice paths from $(0,0)$ to $(10,10)$ using unit steps right or up never pass above the line $y = x$?
11. In how many ways can a $3 \times 10$ rectangle be tiled with $1 \times 2$ dominoes?
12. How many subsets of $\{1, 2, \dots, 20\}$ (including the empty set) have a sum divisible by 5?
13. How many distinct necklaces of 8 beads can be made using beads of 3 colors, where two necklaces are the same if one is a rotation of the other (reflections count as different)?
14. In how many ways can 8 non-attacking queens be placed on an $8 \times 8$ chessboard?
15. How many spanning trees does the complete bipartite graph $K_{3,4}$ have?
16. In how many ways can 100 be written as a sum of positive integers, where order does not matter?
17. How many non-congruent triangles with integer side lengths have perimeter 100?
18. One thousand people stand in a circle numbered 1 to 1000. Starting from person 1, every third person is eliminated (persons 3, 6, 9, …), continuing around the circle until one remains. What is the number of the survivor?

### Probability

19. A permutation of $\{1, \dots, 10\}$ is chosen uniformly at random. What is the exact probability that it has no fixed points? Give the answer as a reduced fraction.
20. A fair six-sided die is rolled repeatedly. What is the expected number of rolls until two consecutive sixes appear?
21. A fair coin is flipped repeatedly. What is the expected number of flips until the sequence HTH first appears?

### Algebra and linear algebra

22. Evaluate $\displaystyle\sum_{k=0}^{50} (-1)^k \binom{100}{2k}$.
23. Let $M$ be the $10 \times 10$ matrix with entries $M_{ij} = \gcd(i, j)$. Compute $\det M$.
24. Let $A = \begin{pmatrix} 1 & 1 \\ 1 & 0 \end{pmatrix}$. What is the top-left entry of $A^{10}$?
25. Find the minimum value of $x^2 + y^2 + z^2$ over real numbers satisfying $x + 2y + 3z = 14$.

### Analysis

26. Evaluate $\displaystyle\int_0^1 \frac{\ln(1+x)}{1+x^2}\,dx$ in closed form.
27. Evaluate $\displaystyle\sum_{n=1}^{\infty} \frac{1}{n^2 (n+1)^2}$ in closed form.
28. Evaluate $\displaystyle\sum_{n=1}^{\infty} \frac{n^3}{2^n}$.
29. Evaluate $\displaystyle\int_0^\infty \frac{x^3}{e^x - 1}\,dx$ in closed form.

### Geometry

30. Find the volume of the region common to the three solid cylinders $x^2 + y^2 \le 1$, $y^2 + z^2 \le 1$, and $z^2 + x^2 \le 1$.

---

## Answer key

Verified by direct computation (exact arithmetic or high-precision numerics) where possible.

| # | Answer | # | Answer | # | Answer |
|---|---|---|---|---|---|
| 1 | 784 | 11 | 571 | 21 | 10 |
| 2 | 343 | 12 | 209728 | 22 | $-2^{50}$ |
| 3 | 615 | 13 | 834 | 23 | 18432 |
| 4 | 405 | 14 | 92 | 24 | 89 |
| 5 | 432 | 15 | 432 | 25 | 14 |
| 6 | 84 | 16 | 190569292 | 26 | $\frac{\pi}{8}\ln 2$ |
| 7 | 65537 | 17 | 208 | 27 | $\frac{\pi^2}{3} - 3$ |
| 8 | 1110 | 18 | 604 | 28 | 26 |
| 9 | 9592 | 19 | $\frac{16481}{44800}$ | 29 | $\frac{\pi^4}{15}$ |
| 10 | 16796 | 20 | 42 | 30 | $8(2 - \sqrt{2})$ |
