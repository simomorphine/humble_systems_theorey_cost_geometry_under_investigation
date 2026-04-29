# The Two-Dimensional Agent

### Complex-Valued Reinforcement Learning: Geometry, Information, and Epistemic Equilibrium

**Mohamed Elwardi** · Independent Researcher, Casablanca
**Humble Systems Theory** ·
*March 2026 · Working Paper*

---

> *"No single weight in a linear combination captures the geometry of the Pareto frontier for all environments. When two objectives are orthogonal, encode them as orthogonal dimensions."*

---

## The Central Idea

Every agent navigating an uncertain world carries two burdens simultaneously — and classical reinforcement learning forces them into a single number.

The first burden is **real cost** $c$: energy, error, time, money — the measurable, immediate price of acting. The second is **information debt** $d$: the cost of operating with a miscalibrated model — the gap between what you know and what the world demands you know.

These two burdens are *geometrically orthogonal* in the following precise sense: the optimal weighting parameter $\mu$ in the scalarisation $c + \mu d$ is environment-dependent, so no fixed $\mu$ traces the full Pareto frontier across all environments. The right response to this parameter sensitivity is not to search for a universal $\mu$ — it is to embrace the geometry. Encode cost and debt as the real and imaginary parts of a complex number:

$$z = c + id \in \mathbb{C}$$

The modulus $|z| = \sqrt{c^2 + d^2}$ is the Euclidean distance from the origin in the cost-debt plane. Minimising $|z|$ is principled, parameter-free, and geometrically natural. The geometry selects a specific point on the Pareto frontier determined by the magnitudes of $c$ and $d$, not by an external choice. No weighting parameter. No tuning. The geometry does the work.

---

## What This Paper Does and Does Not Claim

| Status | Claim |
|:---|:---|
| **Proven** | The Bellman *evaluation* operator is a $\lambda$-contraction with a unique fixed point |
| **Proven** | The imaginary $Q$-value telescopes to a closed form under the potential difference assumption |
| **Proven** | Under the HST Equilibrium Axiom, $|Q|^2$ is a Lyapunov function and the fixed point lies in the upper half-plane |
| **Open** | The Bellman *optimality* operator is not yet proven to be a contraction |
| **Interpretive** | The mutual information reading requires a semantic assumption, stated explicitly |

All formal results in Sections 3 through 9 pertain to the modulus-of-expectation criterion $J^\pi(s) = |\mathbb{E}[G^\pi_t \mid S_t = s]|$, not the expected-modulus criterion $\tilde{J}^\pi(s) = \mathbb{E}[|G^\pi_t| \mid S_t = s]$. By Lemma 1.2, $J^\pi \leq \tilde{J}^\pi$; the expected-modulus criterion requires a distributional treatment and is an open problem.

The Probabilistic Decision Tree (PDT) framework appears as *motivating context*, not as proof.

---

## Contents

1. [Foundations](#1-foundations)
2. [The Complex MDP](#2-the-complex-mdp)
3. [The Bellman Evaluation Equation](#3-the-bellman-evaluation-equation)
4. [The Optimality Operator and Its Gap](#4-the-optimality-operator-and-its-gap)
5. [The Probabilistic Decision Tree](#5-the-probabilistic-decision-tree--a-motivating-bridge)
6. [The Potential Difference Form](#6-the-potential-difference-form)
7. [The Telescoping Identity](#7-the-telescoping-identity)
8. [Mutual Information Falls Out](#8-mutual-information-falls-out)
9. [Epistemic Equilibrium](#9-epistemic-equilibrium)
10. [Four Open Problems](#10-four-open-problems)
11. [Conclusion](#11-conclusion)
12. [References](#12-references)

---

## 1. Foundations

### 1.1 Complex Numbers

For $z = a + ib \in \mathbb{C}$, we write $|z| = \sqrt{a^2 + b^2}$ for the modulus, $\bar{z} = a - ib$ for the conjugate, and $\mathrm{Arg}(z) \in (-\pi, \pi]$ for the principal argument.

**Lemma 1.1** *(Reverse triangle inequality).* For any $z_1, z_2 \in \mathbb{C}$:

$$\big| |z_1| - |z_2| \big| \leq |z_1 - z_2|$$

**Lemma 1.2** *(Modulus of expectation).* For any $\mathbb{C}$-valued random variable $Z$ with $\mathbb{E}[|Z|] < \infty$:

$$|\mathbb{E}[Z]| \leq \mathbb{E}[|Z|]$$

Lemma 1.2 is quietly important: our primary criterion $|\mathbb{E}[G_t]|$ is a *lower bound* on the harder criterion $\mathbb{E}[|G_t|]$, making our problem tractable.

### 1.2 Function Spaces

Let $\mathcal{Q} = \mathcal{B}(\mathcal{S} \times \mathcal{A}, \mathbb{C})$ be the Banach space of bounded complex action-value functions with sup-norm $\|Q\|_ \infty = \max_{s,a} |Q(s,a)|$.

### 1.3 Standard MDPs (Recalled)

A finite MDP is a tuple $(\mathcal{S}, \mathcal{A}, p, r, \lambda)$ where $\lambda \in [0,1)$ is the **discount factor**. The scalar max operator satisfies:

$$\left| \max_a f(a) - \max_a g(a) \right| \leq \max_a |f(a) - g(a)|$$

This inequality is the cornerstone of classical convergence — and exactly what fails in $\mathbb{C}$.

---

## 2. The Complex MDP

**Definition 2.1** *(Complex MDP).* A cMDP is a tuple $\mathcal{M} = (\mathcal{S}, \mathcal{A}, p, z, \lambda)$ where the immediate utility decomposes as:

$$z(s, a, s') = c(s, a, s') + i \cdot d(s, a, s')$$

with $c \geq 0$ a real cost and $d \in \mathbb{R}$ a real information debt (sign-unrestricted). Write $Z_{\max} = \max_{s,a,s'} |z(s,a,s')|$.

The agent *minimises* the modulus. Reducing information debt is a gain. Increasing it is penalised through the modulus.

**Definition 2.2** *(Complex return and performance).* Under a deterministic policy $\pi$:

$$G^\pi_t = \sum_{k=0}^\infty \lambda^k \cdot z(S_{t+k}, A_{t+k}, S_{t+k+1}), \qquad A_{t+k} = \pi(S_{t+k})$$

The sum converges absolutely since $|z| \leq Z_{\max}$ and $\sum_{k=0}^\infty \lambda^k = (1-\lambda)^{-1} < \infty$.

The **performance** of $\pi$ at state $s$ is:

$$J^\pi(s) = \left| \mathbb{E}^\pi\bigl[G^\pi_t \mid S_t = s\bigr] \right|$$

The **optimal performance** is $J^*(s) = \inf_\pi J^\pi(s)$.

> **Two distinct criteria.** The modulus-of-expectation $J^\pi(s) = |\mathbb{E}[G^\pi_t \mid S_t = s]|$ yields a tractable linear Bellman equation and governs all formal results in this paper. The expected-modulus $\tilde{J}^\pi(s) = \mathbb{E}[|G^\pi_t| \mid S_t = s]$ requires a distributional treatment and is an open problem. By Lemma 1.2, $J^\pi \leq \tilde{J}^\pi$.

**Definition 2.3** *(Modulus-greedy policy).* Given $Q \in \mathcal{Q}$, define:

$$\pi_Q(s) = \arg\min_{a \in \mathcal{A}} |Q(s, a)|$$

with ties broken by a fixed deterministic rule.

---

## 3. The Bellman Evaluation Equation

**Theorem 3.1** *(Evaluation contraction — proven).* For any deterministic policy $\pi$, the complex action-value function $Q^\pi(s,a) = \mathbb{E}^\pi[G^\pi_t \mid S_t = s, A_t = a]$ is the *unique* solution in $\mathcal{Q}$ of:

$$Q^\pi(s, a) = \sum_{s' \in \mathcal{S}} p(s' \mid s, a) \Bigl[ z(s, a, s') + \lambda \cdot Q^\pi(s', \pi(s')) \Bigr]$$

The evaluation operator $T^\pi$ is a $\lambda$-contraction on $(\mathcal{Q}, \|\cdot\|_\infty)$.

*Proof.* The Bellman equation follows from the Markov property and dominated convergence. For contraction, take any $Q_1, Q_2 \in \mathcal{Q}$:

$$|(T^\pi Q_1)(s,a) - (T^\pi Q_2)(s,a)|$$

$$= \lambda \left| \sum_{s'} p(s' \mid s,a) \bigl[ Q_1(s', \pi(s')) - Q_2(s', \pi(s')) \bigr] \right|$$

$$\leq \lambda \sum_{s'} p(s' \mid s,a) \cdot |Q_1(s', \pi(s')) - Q_2(s', \pi(s'))| \leq \lambda \cdot \|Q_1 - Q_2\|_\infty$$

Banach's fixed-point theorem gives existence and uniqueness. $\square$

**Corollary 3.2.** $\|Q^\pi\|_ \infty \leq Z_{\max} / (1-\lambda)$.

**Structural decomposition.** Because the evaluation equation is linear over $\mathbb{C}$, the real and imaginary parts satisfy independent Bellman equations:

$$Q^\pi_R(s,a) = \mathbb{E}^\pi\left[\sum_{k=0}^\infty \lambda^k \cdot c(S_{t+k}, A_{t+k}, S_{t+k+1})\right]$$

$$Q^\pi_I(s,a) = \mathbb{E}^\pi\left[\sum_{k=0}^\infty \lambda^k \cdot d(S_{t+k}, A_{t+k}, S_{t+k+1})\right]$$

The two components are decoupled in *evaluation* but coupled in *action selection* via:

$$\arg\min_a |Q(s,a)| = \arg\min_a \sqrt{Q_R^2 + Q_I^2}$$

This coupling is the source of every technical difficulty that follows.

---

## 4. The Optimality Operator and Its Gap

**Definition 4.1** *(Bellman optimality operator).* Define $T : \mathcal{Q} \to \mathcal{Q}$ by:

$$(TQ)(s,a) = \sum_{s'} p(s' \mid s, a) \Bigl[ z(s,a,s') + \lambda \cdot Q(s', \pi_Q(s')) \Bigr]$$

where $\pi_Q(s') = \arg\min_{a'} |Q(s', a')|$.

### The Obstruction

We want $\|TQ_1 - TQ_2\|_ \infty \leq \lambda \cdot \|Q_1 - Q_2\|_\infty$. The proof requires:

$$|Q_1(s', a_1(s')) - Q_2(s', a_2(s'))| \leq \|Q_1 - Q_2\|_\infty$$

In $\mathbb{R}$ this holds because the real line is totally ordered. In $\mathbb{C}$, no such implication holds for $\arg\min |\cdot|$, and the inequality **fails in general**. The obstruction has three faces.

**Face 1 — No total order.** The identity $a_1 = \arg\min |Q(a)|$ says nothing about the complex relationship between $Q(a_1)$ and $Q(a_2)$; the two values can point in opposite directions while having equal modulus.

**Face 2 — Discontinuity of the selector.** The map $Q \mapsto \arg\min_a |Q(a)|$ is discontinuous at ties. An infinitesimal perturbation can switch the selected action, causing a discrete jump in the selected value.

**Face 3 — Phase cancellation.** Two complex numbers can be close in modulus while pointing in nearly opposite directions. The selected values can then be nearly antipodal, making their difference large relative to $\|Q_1 - Q_2\|_\infty$.

### A Partial Positive Result

**Theorem 4.2** *(Scalar modulus contraction — proven).* Define T̂ : B(S, ℝ₊) → B(S, ℝ₊) by:

$$(\hat{T}V)(s) = \min_{a \in \mathcal{A}} \sum_{s'} p(s' \mid s, a) \Bigl[ |z(s,a,s')| + \lambda \cdot V(s') \Bigr]$$

Then $\hat{T}$ is a $\lambda$-contraction with a unique fixed point $V^\dagger$, and value iteration converges at rate $\lambda^n$.

> **What this buys and what it loses.** This confirms the optimal *magnitude* of the value function can be computed by a convergent iteration. What it discards is the *phase* — the cost-debt decomposition. The full complex framework is needed to recover phase information and identify which actions minimise cost versus debt.

---

## 5. The Probabilistic Decision Tree — A Motivating Bridge

*This section does not prove anything. Its role is to make the algebraic choices in Sections 6 through 8 feel inevitable rather than arbitrary.*

### Setup

An agent faces an environment with unknown parameters $\theta \in \Theta$. At each step, uncertainty nests in three levels.

**Level 1 — Action selection.**

$$A = \begin{cases} \text{exploit} & \text{with probability } \alpha \\ \text{explore} & \text{with probability } 1 - \alpha \end{cases}$$

**Level 2 — Reward realisation.** $R \mid A = a, \theta \sim p_\theta(\cdot \mid a)$.

**Level 3 — Information gain.** Model information gain as $I \in \{0,1\}$:

$$P(I = 1 \mid A = a, R = r, \theta) = q_{a,r}(\theta)$$

### The Free Parameter Problem

A natural utility is $U(r, i) = r + \mu \cdot i$ for some value-of-information $\mu > 0$. Single-step optimisation gives: *explore if and only if*

$$\mu \cdot (\iota_E - \iota_P) > \mu_P - \mu_E$$

The decision rule is correct in spirit — but $\mu$ is a free parameter that traces a different point on the Pareto frontier for each value, and the optimal $\mu$ is environment-dependent.

### How Complex Modulus Eliminates the Free Parameter

The modulus criterion $|c + id|$ achieves precisely what the PDT aims for **without any free parameter**. The $L^2$ geometry of $\mathbb{C}$ selects a specific point on the Pareto frontier determined by the magnitudes of cost and information debt, not by an external choice.

Two things carry forward: information gain *belongs* in the utility, and it is *entropy-like* — $I = 1$ corresponds to entropy reduction. These are the hints. The algebra provides the payoff.

---

## 6. The Potential Difference Form

**Definition 6.1** *(Epistemic potential).* A function $\phi : \mathcal{S} \to \mathbb{R}$ is an **epistemic potential**. The imaginary component $d$ takes the **potential difference form** if:

$$d(s, a, s') = \phi(s') - \phi(s)$$

for all $(s, a, s') \in \mathcal{S} \times \mathcal{A} \times \mathcal{S}$.

The information debt of a transition is the difference in epistemic potential between destination and origin. The telescoping identity of Section 7 holds for *any* choice of $\phi : \mathcal{S} \to \mathbb{R}$, regardless of what $\phi$ represents. The semantic interpretation of $\phi$ as a function of entropy is introduced separately in Section 8 and should not be pre-supposed here.

> **Important caveat.** The potential difference form is a structural condition enabling clean algebra — not a restriction derived from BAMDP or Bayesian reasoning. The semantic interpretation is introduced in Section 8.

**Assumption 6.2** *(Learning assumption).* The epistemic potential is non-decreasing in expectation along any policy trajectory:

$$\mathbb{E}^\pi[\phi(S_{t+1}) \mid S_t = s, A_t = a] \geq \phi(s) \qquad \forall (s, a, \pi)$$

In a Bayesian setting with $\phi(s) = -H(\text{belief at } s)$, this holds because Bayesian updates never increase entropy on average, so $-H$ is non-decreasing in expectation. This makes $\phi$ a **sub**martingale along any policy trajectory under Bayesian learning.

---

## 7. The Telescoping Identity

This is the algebraic heart of the paper.

**Theorem 7.1** *(Telescoping identity).* Let $d$ take the potential difference form. For any deterministic policy $\pi$, state $s$, and action $a$:

$$Q^\pi_I(s, a) = -\phi(s) + (1 - \lambda) \cdot \mathbb{E}^\pi\left[\sum_{j=0}^\infty \lambda^j \cdot \phi(S_{t+j+1}) \;\Bigg|\; S_t = s, A_t = a\right]$$

*Proof (six steps of pure algebra).*

**Step 1.** Start from the definition:

$$Q^\pi_I(s,a) = \mathbb{E}^\pi\left[\sum_{k=0}^\infty \lambda^k \bigl(\phi(S_{t+k+1}) - \phi(S_{t+k})\bigr)\right]$$

**Step 2.** Split the sum:

$$= \mathbb{E}^\pi\left[\sum_{k=0}^\infty \lambda^k \phi(S_{t+k+1}) - \sum_{k=0}^\infty \lambda^k \phi(S_{t+k})\right]$$

**Step 3.** Reindex the second sum with $j = k+1$:

$$\sum_{k=0}^\infty \lambda^k \phi(S_{t+k+1}) = \sum_{j=1}^\infty \lambda^{j-1} \phi(S_{t+j})$$

**Step 4.** Separate the $k=0$ term and recombine for $k \geq 1$:

$$Q^\pi_I = \mathbb{E}^\pi\left[\sum_{k=1}^\infty \lambda^{k-1} \phi(S_{t+k}) - \phi(S_t) - \sum_{k=1}^\infty \lambda^k \phi(S_{t+k})\right]$$

**Step 5.** Factor the coefficient of $\phi(S_{t+k})$ for $k \geq 1$:

$$\lambda^{k-1} - \lambda^k = \lambda^{k-1}(1 - \lambda)$$

**Step 6.** Shift index $j = k-1$ to obtain the stated result. $\square$

The algebraic engine is Step 3: reindexing creates the factor $(\lambda^{k-1} - \lambda^k)$. The initial potential $\phi(s)$ survives because $S_t = s$ is fixed — it has no cancellation partner in the reindexed sum.

**Corollary 7.2** *(Non-negativity).* Under Assumption 6.2, $Q^\pi_I(s,a) \geq 0$ for all $(s, a, \pi)$.

*Proof.* Assumption 6.2 makes $\phi$ a submartingale along any policy trajectory, so the discounted future average on the right is at least $\phi(s)$, and the expression is non-negative. $\square$

**Geometric reading.** The modulus decomposes as:

$$|Q^\pi(s,a)|^2 = Q^\pi_R(s,a)^2 + Q^\pi_I(s,a)^2$$

The Pareto frontier between cost and epistemic potential reduction is traced by circles in $\mathbb{C}$. The modulus criterion selects the $L^2$ point on that frontier. The phase $\mathrm{Arg}(Q^\pi(s,a))$ gives the **instantaneous exploration-exploitation angle**: purely real means pure exploitation; purely imaginary means pure exploration.

---

## 8. Mutual Information Falls Out

The telescoping identity holds for *any* $\phi$, algebraically. Now we give $\phi$ a semantic interpretation and watch what follows.

**Assumption 8.1** *(Semantic interpretation of $\phi$).* There exists a latent variable $\Theta$ such that:

$$\phi(s) = - H(\Theta \mid S_t = s)$$

where $H(\Theta \mid S_t = s)$ is the conditional entropy of $\Theta$ given state $s$.

> This assumption is *semantic*, not mathematical. It attaches a meaning to $\phi$. The PDT framework motivates this choice; the algebra does not require it.

**Proposition 8.2** *(One-step imaginary utility equals mutual information).* Under Assumption 8.1:

$$\mathbb{E}_{S_{t+1}}\bigl[d(s, a, S_{t+1})\bigr] = I(S_{t+1};\, \Theta \mid S_t = s, A_t = a)$$

*Proof.*

$$\mathbb{E}_{S_{t+1}}[d(s,a,S_{t+1})] = \mathbb{E}_{S_{t+1}}[\phi(S_{t+1})] - \phi(s)$$

$$= -\mathbb{E}_{S_{t+1}}[H(\Theta \mid S_{t+1}, s, a)] + H(\Theta \mid s, a)$$

$$= I(S_{t+1};\, \Theta \mid s, a) \qquad \square$$

**Corollary 8.3** *(Imaginary Q-value as discounted cumulative mutual information).* Under Assumption 8.1:

$$Q^\pi_I(s,a) = \sum_{k=0}^\infty \lambda^k \cdot I\left(S_{t+k+1};\, \Theta \;\Big|\; S_{t+k}, A_{t+k}\right)$$

The imaginary Q-value is the discounted sum of *all future mutual information gains* along the trajectory under policy $\pi$. Note that this is consistent with Corollary 7.2: mutual information is non-negative, so $Q^\pi_I \geq 0$ as required.

> **What was not assumed.** We did not choose mutual information as the measure of exploration. We wrote $d = \phi(s') - \phi(s)$ and imposed $\phi = -H(\Theta \mid \cdot)$. Mutual information *emerged* from taking expectations because mutual information *is* expected entropy reduction by definition. The complex framework did not import information theory. Information theory fell out of it.

---

## 9. Epistemic Equilibrium

**Axiom 9.1** *(HST Equilibrium Axiom).* Every Information Processing System evolves toward epistemic equilibrium. Formally:

$$\lim_{t \to \infty} Q^{\pi^*}_I(S_t, A_t) = 0 \quad \text{almost surely}$$

> This axiom is not derived. It is the foundational claim of Humble Systems Theory: all information processors seek the state of minimum epistemic tension.

**Proposition 9.2** *(Fixed point in the upper half-plane).* Under Assumption 6.2, if the complex Bellman operator $T$ has a fixed point $Q^*$, then:

$$Q^*(s,a) \in \{z \in \mathbb{C} : \mathrm{Im}(z) \geq 0\} \qquad \forall (s,a)$$

Moreover, under Axiom 9.1 and for identifiable environments, $\mathrm{Im}(Q^*(s,a)) \to 0$ as the system approaches equilibrium.

*Proof.* The first statement follows from Corollary 7.2: $Q^*_I \geq 0$ at the fixed point. The second follows from Axiom 9.1 and the telescoping identity. $\square$

> **Remark 9.3** *(Heuristic: the contraction gap at equilibrium).* The contraction gap of Section 4 arises when two complex values selected by $\arg\min |\cdot|$ point in nearly opposite directions. Proposition 9.2 places $Q^*$ in the upper half-plane. As $Q^*_I \to 0$, all values approach the positive real axis and the maximum phase difference approaches zero. This suggests the contraction gap may vanish in the neighbourhood of the fixed point that Axiom 9.1 guarantees the system is converging toward. This remains a heuristic observation, not a proof: establishing a uniform contraction bound near the fixed point requires a separate argument and is part of OP1.

**Theorem 9.4** ($|Q|^2$ is a Lyapunov function). Define $\mathcal{L}(s,a) = |Q^*(s,a)|^2 = Q^{*2}_R(s,a) + Q^{*2}_I(s,a)$. Then:

- $Q^*_R(s,a)$ measures distance from the cost optimum
- $Q^*_I(s,a) \geq 0$ measures epistemic tension remaining at $(s,a)$
- $\mathcal{L}(s,a)$ decomposes total tension into its cost and epistemic components

Under Axiom 9.1, the epistemic component $Q^*_ I \to 0$. Whether the cost component $Q^*_ R$ also vanishes depends on the structure of the MDP: in a zero-cost optimal trajectory $Q^*_R$ would vanish, but in general the real part of the fixed point is the discounted accumulated real cost under the optimal policy, which need not be zero.

**Proposition 9.5** *(Epistemic equilibrium condition).* A policy $\pi$ is in epistemic equilibrium at $(s,a)$ if and only if:

$$\phi(s) = (1-\lambda) \cdot \mathbb{E}^\pi\left[\sum_{k=0}^\infty \lambda^k \cdot \phi(S_{t+k+1}) \;\Bigg|\; S_t = s, A_t = a\right]$$

At equilibrium, $Q_I = 0$ and the policy acts as a pure cost minimiser. The agent transitions from exploration to exploitation **automatically** — no exploration schedule, no decay parameter, no external switching criterion.

---

## 10. Four Open Problems

*Ordered by dependency. Resolving OP1 would resolve OP3. OP2 and OP4 are independent.*

---

### OP1 — Contraction of the Optimality Operator

**Status: Open — Critical**

**(a)** Is $T$ a $\lambda$-contraction in $(\mathcal{Q}, \|\cdot\|_\infty)$ for all $\lambda \in [0,1)$?

**(b)** If not, what is the effective contraction modulus $\kappa^* = \inf\{\kappa \geq 0 : \|TQ_1 - TQ_2\|_\infty \leq \kappa \|Q_1 - Q_2\|_\infty\}$? Is $\kappa^* < 1$ for some non-trivial class of cMDPs?

**(c)** Does there exist an alternative norm on $\mathcal{Q}$ under which $T$ is a $\lambda$-contraction?

**Candidate resolutions.** A *cone restriction* would bound the maximum phase difference: if $Q^*(s,a) \in \{z \in \mathbb{C} : |\mathrm{Arg}(z)| \leq \theta\}$ for some $\theta < \pi/4$, the cross-action term can be controlled (Remark 9.3 shows $\theta \to 0$ at equilibrium). A *phase-weighted norm* $\|Q\|_{\infty,\varphi} = \max_{s,a} e^{\beta |\mathrm{Arg}(Q(s,a))|} |Q(s,a)|$ for $\beta > 0$ may allow a direct contraction proof. For existence without contraction: *Schauder's fixed-point theorem* applies if $T$ maps a bounded, convex, closed subset of $\mathcal{Q}$ into itself and is **compact** (maps bounded sets to relatively compact sets) — compactness must be verified separately and is not automatic in the infinite-dimensional space $\mathcal{Q}$; in a finite-state MDP, Brouwer's theorem applies directly. Uniqueness then requires a separate argument. A *smoothed operator* with $\pi_\tau(a \mid s) \propto \exp(-|Q(s,a)|/\tau)$ is Lipschitz and its limit as $\tau \to 0$ recovers the greedy selector.

---

### OP2 — Existence and Uniqueness of the Fixed Point

**Status: Open — Partially independent of OP1**

**(a)** Does $Q^* = TQ^*$ have at least one solution in $\mathcal{Q}$?

**(b)** Is the solution unique?

**(c)** If multiple solutions exist, what is the structure of the fixed-point set?

For uniqueness: showing $Q_1 = Q_2$ when both are fixed points requires a well-defined notion of optimality in the complex setting — which requires establishing that $J^* = \inf_\pi J^\pi$ is attained.

---

### OP3 — Almost-Sure Convergence of Complex Q-Learning

**Status: Open — Conditional on OP1**

The complex Q-learning update:

$$Q_{t+1}(s_t, a_t) \leftarrow Q_t(s_t, a_t) + \alpha_t(s_t, a_t) \Bigl[z_t + \lambda \cdot Q_t(s_{t+1}, \pi_{Q_t}(s_{t+1})) - Q_t(s_t, a_t)\Bigr]$$

is well-defined with a martingale difference noise term of bounded variance. Under the Borkar-Meyn stochastic approximation theorem, convergence follows if $T$ is a $\lambda$-contraction.

**(a)** Without the contraction hypothesis, does $\{Q_t\}$ converge?

**(b)** Does $\{|Q_t(s,a)|\}$ converge even if $\{Q_t(s,a)\}$ does not?

**(c)** If $\kappa^* > 1$, can a modified algorithm converge?

---

### OP4 — Characterisation of Valid Epistemic Potentials

**Status: Open — Independent**

**(a)** What is the minimal set of conditions on $\phi$ under which the modulus-greedy policy achieves $J^*(s)$ for all $s$?

**(b)** Does the choice of $\phi$ affect the Pareto frontier, or only which point on the frontier is selected?

**(c)** Under what conditions does the mutual information interpretation (Corollary 8.3) hold?

**(d)** Is there a canonical choice of $\phi$ for a given MDP, analogous to Shannon entropy in the Bayesian setting?

---

## 11. Conclusion

### What Is Established

The **evaluation operator** $T^\pi$ is a proven $\lambda$-contraction for all $\lambda \in [0,1)$, with a unique fixed point $Q^\pi$.

The **imaginary Q-value**, under the potential difference form $d(s,a,s') = \phi(s') - \phi(s)$ and Assumption 6.2 ($\phi$ a submartingale), telescopes to a closed-form expression and is non-negative. This is *pure algebra*: it holds for any $\phi$ satisfying the submartingale condition, regardless of interpretation.

Under the **semantic assumption** $\phi = -H(\Theta \mid \cdot)$, the telescoped quantity equals the discounted cumulative mutual information between future states and $\Theta$, with a positive sign consistent with $Q^\pi_I \geq 0$. This is an *interpretation*: the algebra rewards the choice but does not force it.

Under the **HST Equilibrium Axiom**, the fixed point $Q^*$ lives in the upper half-plane and $|Q|^2$ is a Lyapunov function decomposing total tension into cost and epistemic components. The epistemic component vanishes at equilibrium by Axiom 9.1; the cost component depends on MDP structure and need not vanish.

### What Is Not Established

The $\lambda$-contraction of the **optimality operator** $T$ remains open. The geometric obstruction is identified: no total order on $\mathbb{C}$, discontinuity of the modulus-greedy selector, and phase cancellation between nearly-optimal complex values. The heuristic argument of Remark 9.3 suggests the gap vanishes near equilibrium, but this is not a proof. Resolving this gap — in either direction — is the single most important open problem in this research programme.

### The Final Picture

No weighting parameter appears at any step. The geometry of $\mathbb{C}$ provides the exploration-exploitation trade-off for free.

The **phase** of $Q^*(s,a)$ is the instantaneous angle between exploitation and exploration. The **modulus** is the total distance from equilibrium. The **Lyapunov function** $|Q|^2$ is the natural measure of how far the agent is from the state it is — by axiom — trying to reach.

> *The agent that encodes both cost and information debt as a single complex number does not need to be told how to balance them. The geometry already knows.*

---

## 12. References

- Altman, E. (1999). *Constrained Markov Decision Processes.* Chapman & Hall/CRC.
- Bellemare, M. G., Dabney, W., & Munos, R. (2017). A distributional perspective on reinforcement learning. *ICML 2017*, 449–458.
- Borkar, V. S. (2000). The ODE method for convergence of stochastic approximation and reinforcement learning. *SIAM Journal on Control and Optimization*, 38(2):447–469.
- Borkar, V. S. (2008). *Stochastic Approximation: A Dynamical Systems View.* Cambridge University Press.
- Cover, T. M. & Thomas, J. A. (2006). *Elements of Information Theory*, 2nd ed. Wiley.
- Elwardi, M. (2024). Humble Systems Theory: Foundations, ComplexIPS Architecture, and Bitopological Equilibrium. *HST Working Paper.*
- Elwardi, M. (2025). PDT-BAMDP: Probabilistic Decision Trees for Bayes-Adaptive MDPs with Selective Belief Updates. *HST Working Paper.*
- Elwardi, M. (2026a). Complex-Valued Reinforcement Learning with Modulus Minimisation. *HST Working Paper v2.0.*
- Elwardi, M. (2026b). Complex Bayes-Adaptive MDPs: Entropy, Mutual Information, and the HST Equilibrium Axiom. *HST Working Paper v1.0.*
- Hayes, C. F. et al. (2022). A practical guide to multi-objective reinforcement learning and planning. *AAMAS*, 36(1):1–59.
- Hirose, A. (2012). *Complex-Valued Neural Networks*, 2nd ed. Springer-Verlag.
- Liu, C., Xu, X., & Hu, D. (2015). Multiobjective reinforcement learning: A comprehensive overview. *IEEE Trans. SMC*, 45(3):385–398.
- Puterman, M. L. (2014). *Markov Decision Processes.* Wiley.
- Russo, D. & Van Roy, B. (2018). Learning to optimise with information-directed sampling. *Operations Research*, 66(1):230–252.
- Sutton, R. S. & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*, 2nd ed. MIT Press.
- Szepesvari, C. (2010). Algorithms for reinforcement learning. *Synthesis Lectures on AI and ML.* Morgan & Claypool.
- Trabelsi, C. et al. (2018). Deep complex networks. *ICLR 2018.*
- Watkins, C. J. C. H. & Dayan, P. (1992). Q-learning. *Machine Learning*, 8(3–4):279–292.
- Wirtinger, W. (1927). Zur formalen Theorie der Funktionen von mehr komplexen Veranderlichen. *Mathematische Annalen*, 97(1):357–375.
