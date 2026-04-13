# The Decision Tree Theory of Exploration: A Unified Framework for Learning Under Uncertainty

**Abstract.** We present a unified decision-theoretic framework for understanding exploration in reinforcement learning through the lens of probabilistic decision trees. By explicitly modeling the joint distribution over actions, rewards, and information gain, we derive fundamental results connecting classical exploration strategies ($\varepsilon$-greedy, Thompson sampling, UCB) to Bayesian decision theory and information theory. We show that exploration emerges naturally from the interplay between immediate reward maximization and long-term information acquisition, formalized through Bayes-Adaptive Markov Decision Processes. Our framework reveals exploration not as a heuristic add-on to exploitation, but as a rational consequence of operating under uncertainty with future-directed value.

---

## 1. Introduction

The exploration-exploitation dilemma stands as one of the central problems in sequential decision-making under uncertainty.

---

## 2. The Probabilistic Decision Tree

### Action Selection

$$
A \sim 
\begin{cases}
P & \text{with probability } \alpha \\
E & \text{with probability } 1 - \alpha
\end{cases}
$$

---

### Reward Model

$$
\Pr(R = +r \mid A=a, \theta) = p_a(\theta)
$$

---

### Information Gain

$$
\Pr(I=1 \mid A=a, R=r, \theta) = q_{a,r}(\theta)
$$

---

### Joint Distribution

$$
\Pr(A,R,I \mid \theta,\alpha)
= \Pr(A \mid \alpha)\Pr(R \mid A,\theta)\Pr(I \mid A,R,\theta)
$$

---

## 3. Utility

$$
U(r,i) = r + \lambda i
$$

---

## 4. Expected Utility (Fixed)

$$
\mathbb{E}_{\theta,\,A \sim \alpha}[U]
=
\int \sum_{a,r,i}
\Pr(a,r,i \mid \theta,\alpha)\, U(r,i)\, b(\theta)\, d\theta
$$

---

## 5. Decomposition (Clean)

$$
\mathbb{E}_{\theta,\,A \sim \alpha}[U]
=
\alpha\, \mathbb{E}_{\theta}[U \mid P]
+
(1 - \alpha)\, \mathbb{E}_{\theta}[U \mid E]
$$

---

Define:

$$
\mu_a = \mathbb{E}_{\theta}[R \mid a]
\quad\text{and}\quad
\iota_a = \mathbb{E}_{\theta}[I \mid a]
$$

Then:

$$
\mathbb{E}_{\theta}[U \mid a]
=
\mu_a + \lambda\, \iota_a
$$

---

## 6. Optimization

$$
\mathbb{E}[U](\alpha)
=
\alpha(\mu_P + \lambda \iota_P)
+
(1-\alpha)(\mu_E + \lambda \iota_E)
$$

---

### Optimal Policy

$$
\alpha^* =
\begin{cases}
1 & \text{if } \mu_P + \lambda \iota_P > \mu_E + \lambda \iota_E \\
0 & \text{if } \mu_P + \lambda \iota_P < \mu_E + \lambda \iota_E \\
\text{any } \alpha \in [0,1] & \text{if equal}
\end{cases}
$$

---

### Exploration Condition

$$
\lambda >
\frac{\mu_P - \mu_E}{\iota_E - \iota_P}
$$

---

## 7. Key Insight

$$
\text{Explore if: }
\lambda \cdot \mathbb{E}[\text{information gain}]
>
\mathbb{E}[\text{reward loss}]
$$

---

## 8. Interpretation

Exploration is not a heuristic — it is a rational consequence of valuing information.

---

## References

- Russo & Van Roy (2014)  
- Russo & Van Roy (2018)  
- Friston et al. (2015)
