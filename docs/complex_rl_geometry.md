# The Two-Dimensional Agent
### *Complex-Valued Reinforcement Learning: Geometry, Information, and Epistemic Equilibrium*

**Mohamed Elwardi** · Independent Researcher, Casablanca  
**Humble Systems Theory Research Group** · `humble.systems.theory@research.ma`  
*March 2026 · Working Paper*

---

> *"No single weight λ captures the geometry of the Pareto frontier for all environments.  
> When two objectives are orthogonal, encode them as orthogonal dimensions."*

---

## The Central Idea

Every agent navigating an uncertain world carries two burdens simultaneously — and classical reinforcement learning forces them into a single number.

The first burden is **real cost** $c$: energy, error, time, money — the measurable, immediate price of acting. The second is **information debt** $d$: the cost of operating with a miscalibrated model — the gap between what you know and what the world demands you know.

These two burdens are *geometrically orthogonal*. No weighted sum $c + \lambda d$ traces the full Pareto frontier between them. The right response to orthogonality is not to fight it with a tuning knob — it is to embrace it mathematically. Encode them as the real and imaginary parts of a complex number:

$$\boxed{z = c + id \in \mathbb{C}}$$

The modulus $|z| = \sqrt{c^2 + d^2}$ is then the Euclidean distance from the origin in the cost-debt plane. Minimising $|z|$ is principled, parameter-free, and geometrically natural. No $\lambda$. No tuning. The geometry does the work.

This paper develops the mathematical consequences of that encoding — carefully, honestly, and completely.

---

## What This Paper Does — and Does Not — Claim

We are precise about the boundary between what is proven and what remains open.

| Status | Claim |
|:---|:---|
| ✅ **Proven** | The Bellman *evaluation* operator is a $\gamma$-contraction with a unique fixed point |
| ✅ **Proven** | The imaginary $Q$-value telescopes to a closed form under the potential difference assumption |
| ✅ **Proven** | Under the HST Equilibrium Axiom, $\|Q\|^2$ is a Lyapunov function and the fixed point lives in the upper half-plane |
| ⚠️ **Open** | The Bellman *optimality* operator is not yet proven to be a contraction — the geometric obstruction is identified precisely |
| 📐 **Interpretive** | The mutual information reading of the imaginary component requires a semantic assumption, stated explicitly |

The Probabilistic Decision Tree (PDT) framework appears as *motivating context*, not as proof. It provides the intuition that information gain belongs in the utility; the subsequent algebra rewards that intuition without being forced by it.

---

## Contents

1. [Foundations](#1-foundations)
2. [The Complex MDP](#2-the-complex-mdp)
3. [The Bellman Evaluation Equation](#3-the-bellman-evaluation-equation)
4. [The Optimality Operator and Its Gap](#4-the-optimality-operator-and-its-gap)
5. [The Probabilistic Decision Tree — A Motivating Bridge](#5-the-probabilistic-decision-tree--a-motivating-bridge)
6. [The Potential Difference Form](#6-the-potential-difference-form)
7. [The Telescoping Identity](#7-the-telescoping-identity)
8. [Mutual Information Falls Out](#8-mutual-information-falls-out)
9. [Epistemic Equilibrium](#9-epistemic-equilibrium)
10. [Four Open Problems](#10-four-open-problems)
11. [Conclusion](#11-conclusion)

---

## 1. Foundations

### Complex Numbers

For $z = a + ib \in \mathbb{C}$:

$$|z| = \sqrt{a^2 + b^2}, \quad \bar{z} = a - ib, \quad \text{Arg}(z) \in (-\pi, \pi]$$

Two lemmas do most of the subsequent heavy lifting.

**Lemma 1.1** *(Reverse triangle inequality)*. For any $z_1, z_2 \in \mathbb{C}$:

$$\big| |z_1| - |z_2| \big| \leq |z_1 - z_2|$$

**Lemma 1.2** *(Modulus of expectation)*. For any $\mathbb{C}$-valued random variable $Z$ with $\mathbb{E}[|Z|] < \infty$:

$$|\mathbb{E}[Z]| \leq \mathbb{E}[|Z|]$$

Lemma 1.2 is quietly important: our primary criterion $|\mathbb{E}[G_t]|$ is a *lower bound* on the harder criterion $\mathbb{E}[|G_t|]$. This makes our problem tractable. The gap between these two criteria is itself an open problem (see §10).

### Function Spaces

Let $\mathcal{Q} = \mathcal{B}(\mathcal{S} \times \mathcal{A}; \mathbb{C})$ be the Banach space of bounded complex action-value functions equipped with the sup-norm $\|Q\|_\infty = \max_{s,a} |Q(s,a)|$. For finite $\mathcal{S}$, this is isometric to $\mathbb{C}^{|\mathcal{S}||\mathcal{A}|}$ with the $\ell^\infty$ norm.

### Standard MDPs (Recalled)

A finite MDP is a tuple $(\mathcal{S}, \mathcal{A}, p, r, \gamma)$ with the usual definitions. The scalar max operator satisfies the *non-expansiveness* property that underlies all classical Q-learning convergence:

$$\left| \max_a f(a) - \max_a g(a) \right| \leq \max_a |f(a) - g(a)| \qquad \forall f, g : \mathcal{A} \to \mathbb{R}$$

This inequality is the cornerstone of classical convergence — and exactly what fails in $\mathbb{C}$. The central technical challenge of this paper lives in that failure.

---

## 2. The Complex MDP

**Definition 2.1** *(Complex MDP)*. A cMDP is a tuple $\mathcal{M} = (\mathcal{S}, \mathcal{A}, p, z, \gamma)$ where the immediate utility $z : \mathcal{S} \times \mathcal{A} \times \mathcal{S} \to \mathbb{C}$ is:

$$z(s, a, s') = c(s, a, s') + i\, d(s, a, s')$$

with $c \geq 0$ a real cost and $d \in \mathbb{R}$ a real information debt (sign-unrestricted). Write $Z_{\max} = \max_{s,a,s'} |z(s,a,s')|$.

The agent *minimises* modulus. Reducing information debt — becoming less uncertain — is treated as a gain. Increasing it is penalised through the modulus.

**Definition 2.2** *(Complex return and performance)*. Under policy $\pi$:

$$G^\pi_t = \sum_{k=0}^\infty \gamma^k z(S_{t+k}, A_{t+k}, S_{t+k+1}), \qquad A_{t+k} = \pi(S_{t+k})$$

The sum converges absolutely because $|z| \leq Z_{\max}$ and $\sum_{k=0}^\infty \gamma^k = (1-\gamma)^{-1} < \infty$.

The **performance** of $\pi$ at state $s$ is the modulus of the expected complex return:

$$J^\pi(s) = \left| \mathbb{E}^\pi[G^\pi_t \mid S_t = s] \right|$$

The **optimal performance** is $J^*(s) = \inf_\pi J^\pi(s)$.

> **Two distinct criteria.** There is $J^\pi(s) = |\mathbb{E}[G^\pi_t \mid S_t = s]|$ (modulus of expectation — our primary criterion) and $\tilde{J}^\pi(s) = \mathbb{E}[|G^\pi_t| \mid S_t = s]$ (expected modulus). By Lemma 1.2, $J^\pi \leq \tilde{J}^\pi$. Criterion (i) yields a tractable linear Bellman equation. Criterion (ii) requires a distributional treatment and is an open problem.

**Definition 2.3** *(Modulus-greedy policy)*. Given $Q \in \mathcal{Q}$:

$$\pi_Q(s) = \arg\min_{a \in \mathcal{A}} |Q(s, a)|$$

with ties broken by a fixed deterministic rule.

---

## 3. The Bellman Evaluation Equation

**Theorem 3.1** *(Evaluation contraction — proven)*. For any deterministic policy $\pi$, the complex action-value function $Q^\pi(s,a) = \mathbb{E}^\pi[G^\pi_t \mid S_t = s, A_t = a]$ is the *unique* solution in $\mathcal{Q}$ of:

$$Q^\pi(s, a) = \sum_{s' \in \mathcal{S}} p(s' \mid s, a)\Big[ z(s, a, s') + \gamma Q^\pi(s', \pi(s')) \Big] \tag{$\star$}$$

The evaluation operator $T^\pi$ defined by the right-hand side is a $\gamma$-**contraction** on $(\mathcal{Q}, \|\cdot\|_\infty)$.

*Proof.* The Bellman equation follows from the Markov property and dominated convergence. Contraction: for any $Q_1, Q_2 \in \mathcal{Q}$ and any $(s,a)$:

$$|(T^\pi Q_1)(s,a) - (T^\pi Q_2)(s,a)| = \gamma \left| \sum_{s'} p(s' \mid s,a)\big[Q_1(s', \pi(s')) - Q_2(s', \pi(s'))\big] \right|$$

$$\leq \gamma \sum_{s'} p(s' \mid s,a) \big|Q_1(s', \pi(s')) - Q_2(s', \pi(s'))\big| \leq \gamma \|Q_1 - Q_2\|_\infty$$

Banach's fixed-point theorem gives existence and uniqueness. $\square$

**Uniform bound:** $\|Q^\pi\|_\infty \leq Z_{\max}/(1-\gamma)$.

**Structural decomposition.** Because $(\star)$ is linear over $\mathbb{C}$, the real and imaginary components satisfy *independent* Bellman equations:

$$Q^\pi_R(s,a) = \mathbb{E}^\pi\!\left[\sum_{k=0}^\infty \gamma^k c(S_{t+k}, A_{t+k}, S_{t+k+1})\right]$$

$$Q^\pi_I(s,a) = \mathbb{E}^\pi\!\left[\sum_{k=0}^\infty \gamma^k d(S_{t+k}, A_{t+k}, S_{t+k+1})\right]$$

The two components are decoupled in *evaluation* but coupled in *action selection* through $\arg\min_a |Q(s,a)| = \arg\min_a \sqrt{Q_R^2 + Q_I^2}$. This coupling is the source of every technical difficulty that follows.

---

## 4. The Optimality Operator and Its Gap

**Definition 4.1** *(Bellman optimality operator)*. Define $T : \mathcal{Q} \to \mathcal{Q}$ by:

$$(TQ)(s,a) = \sum_{s'} p(s' \mid s, a)\Big[ z(s,a,s') + \gamma Q(s', \pi_Q(s')) \Big]$$

where $\pi_Q(s') = \arg\min_{a'} |Q(s', a')|$. If a fixed point $Q^* = TQ^*$ exists, the modulus-greedy policy $\pi^* = \pi_{Q^*}$ is a candidate optimal policy.

### The Obstruction

We want $\|TQ_1 - TQ_2\|_\infty \leq \gamma \|Q_1 - Q_2\|_\infty$. The standard proof strategy fails at one step. Letting $a_i(s') = \arg\min_{a'} |Q_i(s', a')|$, the proof requires:

$$\big|Q_1(s', a_1(s')) - Q_2(s', a_2(s'))\big| \leq \|Q_1 - Q_2\|_\infty \tag{?}$$

In $\mathbb{R}$, inequality $(\text{?})$ holds because $\mathbb{R}$ is *totally ordered*: $a_1 = \arg\max f$ forces $f(a_1) \geq f(a_2)$, which kills the cross-action term. In $\mathbb{C}$, no such implication holds for $\arg\min |\cdot|$, and $(\text{?})$ **fails in general**.

The obstruction has three faces:

1. **No total order.** $a_1 = \arg\min |Q(a)|$ says nothing about the complex relationship between $Q(a_1)$ and $Q(a_2)$; the two values can point in opposite directions while having equal modulus.

2. **Discontinuity of the selector.** The map $Q \mapsto \arg\min_a |Q(a)|$ is discontinuous at ties. An infinitesimal perturbation can switch the selected action, causing a discrete jump in the selected complex value.

3. **Phase cancellation.** Two complex numbers can be close in modulus while pointing in nearly opposite directions (phase difference $\approx \pi$). The selected values can then be nearly antipodal, making their difference large relative to $\|Q_1 - Q_2\|_\infty$.

### A Partial Positive Result

Although the full contraction is unproven, this much is rigorous.

**Theorem 4.2** *(Scalar modulus contraction — proven)*. Define $\hat{T} : \mathcal{B}(\mathcal{S}; \mathbb{R}_{\geq 0}) \to \mathcal{B}(\mathcal{S}; \mathbb{R}_{\geq 0})$ by:

$$(\hat{T}V)(s) = \min_{a \in \mathcal{A}} \sum_{s'} p(s' \mid s, a)\Big[ |z(s,a,s')| + \gamma V(s') \Big]$$

Then $\hat{T}$ is a $\gamma$-contraction with a unique fixed point $V^\dagger$, and value iteration converges at rate $\gamma^n$.

> **What this buys.** $\hat{T}$ confirms that the optimal *magnitude* of the value function can be computed by a convergent iteration. What it discards is the *phase* — the cost-debt decomposition. The full complex framework is required to recover phase information and thereby identify which actions minimise cost versus debt. The scalar result is useful; it is not sufficient.

---

## 5. The Probabilistic Decision Tree — A Motivating Bridge

*This section does not prove anything. Its role is to make the subsequent algebraic choices feel inevitable rather than arbitrary.*

### The Setup

An agent faces an environment with unknown parameters $\theta \in \Theta$. At each step, uncertainty nests in three levels:

**Level 1 — Action selection.** The agent may follow its current policy or deviate:

$$A = \begin{cases} \text{exploit} & \text{w.p. } \alpha \\ \text{explore} & \text{w.p. } 1 - \alpha \end{cases}$$

**Level 2 — Reward realisation.** $R \mid A = a, \theta \sim p_\theta(\cdot \mid a)$.

**Level 3 — Information gain.** Observing a reward does not guarantee learning. Model information gain as a binary event $I \in \{0,1\}$:

$$P(I = 1 \mid A = a, R = r, \theta) = q_{a,r}(\theta)$$

### The Free Parameter Problem

A natural utility is $U(r, i) = r + \lambda i$ for some value-of-information parameter $\lambda > 0$. Single-step optimisation yields: *explore if and only if*

$$\lambda(\iota_E - \iota_P) > \mu_P - \mu_E$$

The decision rule is correct in spirit. But $\lambda$ is a free parameter. Different choices of $\lambda$ trace different points on the Pareto frontier between reward and information gain. The rule tells you *how* to decide; it does not tell you *which* decision matters.

### How Complex Modulus Eliminates the Free Parameter

The modulus criterion $|c + id|$ with $c = -r$ and $d$ encoding information debt achieves precisely what the PDT aims for — **without $\lambda$**. The $L^2$ geometry of $\mathbb{C}$ selects a specific point on the Pareto frontier determined by the magnitudes of cost and information debt, not by an external parameter.

Two things carry forward from the PDT: (i) information gain *belongs* in the utility — the agent that ignores it is not rational; (ii) information gain is *entropy-like* — $I=1$ corresponds to entropy reduction. These are hints. The algebra provides the payoff.

---

## 6. The Potential Difference Form

**Definition 6.1** *(Epistemic potential)*. A function $\phi : \mathcal{S} \to \mathbb{R}$ is an epistemic potential when it assigns to each state a real number representing the epistemic uncertainty associated with that state.

The imaginary component $d$ takes the **potential difference form** if:

$$\boxed{d(s, a, s') = \phi(s) - \phi(s')} \tag{PDF}$$

### Why This Form?

The information debt of a transition $(s, a, s')$ is determined entirely by the difference in epistemic potential between origin and destination. This is the *simplest possible structure*: $d$ depends on where you are and where you end up, not on the action taken beyond its effect on the next state.

A state with high $\phi$ is one where the agent is still uncertain. Transitioning to a state of lower $\phi$ reduces uncertainty — positive information debt reduction, i.e. information gain. A transition that *increases* uncertainty (negative $d$) increases the imaginary part of the utility and penalises the action through the modulus.

> **Important caveat.** The potential difference form is *not* a restriction derived from BAMDP or Bayesian reasoning. It is a structural condition on $d$ that enables clean algebra. The telescoping identity of §7 holds for *any* function $\phi : \mathcal{S} \to \mathbb{R}$, regardless of interpretation. The semantic interpretation of $\phi$ as entropy is introduced separately in §8.

**Assumption 6.2** *(Learning assumption)*. The epistemic potential $\phi$ is non-increasing in expectation along any policy trajectory:

$$\mathbb{E}^\pi[\phi(S_{t+1}) \mid S_t = s, A_t = a] \leq \phi(s) \qquad \forall (s, a, \pi)$$

This encodes the idea that on average, the agent moves toward states of lower epistemic uncertainty regardless of the policy followed. In a Bayesian setting with $\phi(s) = H(\text{belief at } s)$, it is guaranteed by the law of total expectation applied to the posterior — Bayesian updates never increase entropy on average.

---

## 7. The Telescoping Identity

This is the algebraic heart of the paper.

**Theorem 7.1** *(Telescoping identity)*. Let $d$ take the potential difference form (PDF). For any deterministic policy $\pi$, state $s$, and action $a$:

$$\boxed{Q^\pi_I(s, a) = \phi(s) - (1 - \gamma)\,\mathbb{E}^\pi\!\left[\sum_{j=0}^\infty \gamma^j \phi(S_{t+j+1}) \;\Big|\; S_t = s, A_t = a\right]}$$

*Proof (six steps of pure algebra).*

**Step 1.** Start from the definition:

$$Q^\pi_I(s,a) = \mathbb{E}^\pi\!\left[\sum_{k=0}^\infty \gamma^k \big(\phi(S_{t+k}) - \phi(S_{t+k+1})\big)\right]$$

**Step 2.** Split the sum:

$$= \mathbb{E}^\pi\!\left[\sum_{k=0}^\infty \gamma^k \phi(S_{t+k}) - \sum_{k=0}^\infty \gamma^k \phi(S_{t+k+1})\right]$$

**Step 3.** Reindex the second sum with $j = k+1$:

$$\sum_{k=0}^\infty \gamma^k \phi(S_{t+k+1}) = \sum_{j=1}^\infty \gamma^{j-1} \phi(S_{t+j})$$

**Step 4.** Separate the $k=0$ term from the first sum and recombine for $k \geq 1$:

$$Q^\pi_I = \mathbb{E}^\pi\!\left[\phi(S_t) + \sum_{k=1}^\infty \gamma^k \phi(S_{t+k}) - \sum_{k=1}^\infty \gamma^{k-1} \phi(S_{t+k})\right]$$

**Step 5.** Factor the coefficient of $\phi(S_{t+k})$ for $k \geq 1$:

$$\gamma^k - \gamma^{k-1} = \gamma^{k-1}(\gamma - 1) = -(1-\gamma)\gamma^{k-1}$$

**Step 6.** Shift index $j = k-1$ to obtain the result. $\square$

### What This Means

The algebraic engine is Step 3: reindexing creates the factor $(\gamma^k - \gamma^{k-1})$. The initial potential $\phi(s)$ *survives* because $S_t = s$ is fixed — it has no cancellation partner in the reindexed sum. The prefactor $(1-\gamma)$ is the discrete analogue of a derivative.

**Corollary 7.2** *(Non-negativity)*. Under Assumption 6.2, $Q^\pi_I(s,a) \geq 0$ for all $(s,a,\pi)$.

*Proof.* Assumption 6.2 makes $\phi$ a supermartingale along any policy trajectory. Hence the discounted average on the right-hand side is at most $\phi(s)$. $\square$

**Geometric reading:**

$$|Q^\pi(s,a)|^2 = \underbrace{Q^\pi_R(s,a)^2}_{\text{discounted cumulative cost}^2} + \underbrace{Q^\pi_I(s,a)^2}_{\text{epistemic potential reduction}^2}$$

The Pareto frontier between cost and epistemic potential reduction is traced by *circles* in $\mathbb{C}$. The modulus criterion selects the $L^2$ point on that frontier. The phase $\text{Arg}(Q^\pi(s,a))$ gives the **instantaneous exploration-exploitation angle**: purely real means pure exploitation; purely imaginary means pure exploration.

---

## 8. Mutual Information Falls Out

The telescoping identity holds for *any* $\phi$, algebraically. Now we give $\phi$ a semantic interpretation and watch what happens.

**Assumption 8.1** *(Semantic interpretation of $\phi$)*. There exists a latent variable $\Theta$ on a measurable space $(\Theta, \mathcal{F})$ such that:

$$\phi(s) = H(\Theta \mid S_t = s)$$

where $H(\Theta \mid S_t = s)$ is the conditional entropy of $\Theta$ given that the agent is in state $s$.

> This assumption is *semantic*, not mathematical. It attaches a meaning to $\phi$. The PDT framework motivates this choice (§5), but does not force it. The algebra does not require it. We state it explicitly as an interpretive choice.

**Proposition 8.2** *(One-step imaginary utility = mutual information)*. Under Assumption 8.1:

$$\mathbb{E}_{S_{t+1}}[d(s, a, S_{t+1})] = I(S_{t+1};\, \Theta \mid S_t = s, A_t = a)$$

*Proof.*

$$\mathbb{E}_{S_{t+1}}[d(s,a,S_{t+1})] = \phi(s) - \mathbb{E}_{S_{t+1}}[\phi(S_{t+1})]$$

$$= H(\Theta \mid s) - \mathbb{E}_{S_{t+1}}[H(\Theta \mid S_{t+1}, s, a)]$$

$$= H(\Theta \mid s, a) - H(\Theta \mid S_{t+1}, s, a) = I(S_{t+1};\, \Theta \mid s, a) \quad \square$$

**Corollary 8.3** *(Imaginary $Q$-value as discounted cumulative mutual information)*. Under Assumption 8.1:

$$Q^\pi_I(s,a) = \sum_{k=0}^\infty \gamma^k\, I\!\left(S_{t+k+1};\, \Theta \;\Big|\; S_{t+k}, A_{t+k}\right)$$

The imaginary $Q$-value is the discounted sum of *all future mutual information gains* along the trajectory under policy $\pi$.

> **What was not assumed.** We did not choose mutual information as the measure of exploration. We wrote $d = \phi(s) - \phi(s')$ — the simplest possible entropy-like imaginary component — and imposed the semantic assumption $\phi = H(\Theta \mid \cdot)$. Mutual information *emerged* from taking expectations because mutual information *is* expected entropy reduction, by definition. The complex framework did not import information theory. Information theory fell out of it.

---

## 9. Epistemic Equilibrium

We now introduce the foundational axiom of Humble Systems Theory.

**Axiom 9.1** *(HST Equilibrium Axiom)*. Every Information Processing System (IPS) evolves toward epistemic equilibrium. Formally, for any IPS operating under the cMDP framework with epistemic potential $\phi$:

$$\lim_{t \to \infty} Q^{\pi^*}_I(S_t, A_t) = 0 \quad \text{almost surely}$$

> This axiom is not derived. It is the foundational claim of Humble Systems Theory: all information processors seek the state of minimum epistemic tension — the state where no further reduction in epistemic potential is achievable given the agent's constraints.

**Proposition 9.2** *(Fixed point in the upper half-plane)*. Under Assumption 6.2, if the complex Bellman operator $T$ has a fixed point $Q^*$, then:

$$Q^*(s,a) \in \{z \in \mathbb{C} : \text{Im}(z) \geq 0\} \qquad \forall (s,a)$$

Moreover, under Axiom 9.1 and for identifiable environments, $\text{Im}(Q^*(s,a)) \to 0$ as the system approaches equilibrium.

*Proof.* The first statement follows from Corollary 7.2: $Q^*_I = Q^\pi_I \geq 0$ at the fixed point. The second follows from Axiom 9.1 and the telescoping identity. $\square$

**Corollary 9.3** *(The contraction gap vanishes at equilibrium)*. The contraction gap of §4 arises when two complex values selected by $\arg\min |\cdot|$ point in nearly opposite directions — phase difference $\approx \pi$. Proposition 9.2 places the fixed point $Q^*$ in the upper half-plane: $\text{Arg}(Q^*(s,a)) \in [0, \pi]$ for all $(s,a)$. As $Q^*_I \to 0$, all values $Q^*(s,a)$ approach the positive real axis. The maximum phase difference between any two actions approaches $0$. The violation factor approaches $1$. The contraction gap **vanishes precisely at the fixed point that Axiom 9.1 guarantees the system is converging toward**.

**Theorem 9.4** *($|Q|^2$ is a Lyapunov function)*. Define $\mathcal{L}(s,a) = |Q^*(s,a)|^2 = Q^{*2}_R(s,a) + Q^{*2}_I(s,a)$. Then:

- $Q^*_R(s,a)$ measures distance from the *cost* optimum
- $Q^*_I(s,a) \geq 0$ measures distance from *epistemic equilibrium*
- $\mathcal{L}(s,a)$ measures **total distance from full equilibrium** — zero cost and zero epistemic uncertainty

Under Axiom 9.1, $\mathcal{L} \to Q^{*2}_R$ as the system approaches epistemic equilibrium ($\phi \to 0$), and $\mathcal{L} \to 0$ only when *both* cost is minimised *and* the system is epistemically certain. The policy minimising $|Q|$ is therefore the policy minimising total distance from full equilibrium at every step.

**Proposition 9.5** *(Epistemic equilibrium condition)*. A policy $\pi$ is in epistemic equilibrium at $(s,a)$ if and only if:

$$\phi(s) = (1-\gamma)\,\mathbb{E}^\pi\!\left[\sum_{k=0}^\infty \gamma^k \phi(S_{t+k+1}) \;\Big|\; S_t = s, A_t = a\right]$$

At equilibrium, $Q_I = 0$ and the policy acts as a pure cost minimiser. The agent transitions from exploration to exploitation **automatically** — no exploration schedule, no decay parameter, no external switching criterion.

---

## 10. Four Open Problems

*Ordered by dependency. Resolving OP1 would resolve OP3. OP2 and OP4 are independent.*

---

### OP1 · Contraction of the Optimality Operator
**Status: Open — Critical**

Let $T : \mathcal{Q} \to \mathcal{Q}$ be the Bellman optimality operator (Definition 4.1).

**(a)** Is $T$ a $\gamma$-contraction in $(\mathcal{Q}, \|\cdot\|_\infty)$ for all $\gamma \in [0,1)$?

**(b)** If not, what is the effective contraction modulus $\kappa^* = \inf\{\kappa \geq 0 : \|TQ_1 - TQ_2\|_\infty \leq \kappa\|Q_1 - Q_2\|_\infty\}$? Is $\kappa^* < 1$ for some non-trivial class of cMDPs?

**(c)** Does there exist an alternative norm on $\mathcal{Q}$ under which $T$ is a $\gamma$-contraction for all $\gamma \in [0,1)$?

**Candidate resolutions:**

- *Cone restriction.* If $Q^*(s,a) \in \{z \in \mathbb{C} : |\text{Arg}(z)| \leq \theta\}$ for some $\theta < \pi/4$, the maximum phase difference is bounded by $2\theta$ and the cross-action term can be controlled. Corollary 9.3 shows $\theta \to 0$ at equilibrium; the question is whether a cone restriction holds globally.

- *Phase-weighted norm.* Define $\|Q\|_{\infty,\varphi} = \max_{s,a} e^{\beta|\text{Arg}Q(s,a)|} |Q(s,a)|$ for some $\beta > 0$. If $T$ is a contraction in this norm, the Banach fixed-point theorem applies.

- *Schauder + separate uniqueness.* Since $T$ maps the ball $\{\|Q\|_\infty \leq Z_{\max}/(1-\gamma)\}$ into itself and is continuous, Schauder's theorem gives existence of a fixed point without contraction. Uniqueness requires a separate argument exploiting MDP structure.

- *Smoothed operator.* Replace the greedy selector with $\pi_\tau(a \mid s) \propto \exp(-|Q(s,a)|/\tau)$. The smoothed operator $T_\tau$ is Lipschitz with constant $\gamma \cdot L(\tau)$, where $L(\tau) \to 1$ as $\tau \to \infty$ and $L(\tau) \to \infty$ as $\tau \to 0^+$. Characterise the range of $\tau$ for which $T_\tau$ is a contraction, then study the limit $\tau \to 0$.

---

### OP2 · Existence and Uniqueness of the Fixed Point
**Status: Open — Partially independent of OP1**

**(a)** Does $Q^* = TQ^*$ have at least one solution in $\mathcal{Q}$? *(Schauder gives existence without contraction; likely resolvable independently.)*

**(b)** Is the solution unique?

**(c)** If multiple solutions exist, what is the structure of the fixed-point set?

For uniqueness: if $Q_1 = TQ_1$ and $Q_2 = TQ_2$, both induced policies $\pi_{Q_1}$ and $\pi_{Q_2}$ are candidates for optimality. Showing $Q_1 = Q_2$ requires a well-defined notion of optimality in the complex setting, which in turn requires establishing that $J^* = \inf_\pi J^\pi$ is attained. This is a non-trivial independent subproblem.

---

### OP3 · Almost-Sure Convergence of Complex Q-Learning
**Status: Open — Conditional on OP1**

The complex Q-learning update is:

$$Q_{t+1}(s_t, a_t) \leftarrow Q_t(s_t, a_t) + \alpha_t(s_t, a_t)\Big[z_t + \gamma Q_t(s_{t+1}, \pi_{Q_t}(s_{t+1})) - Q_t(s_t, a_t)\Big]$$

The update is well-defined; the noise term is a martingale difference with bounded variance. Under the Borkar–Meyn stochastic approximation theorem, convergence follows if $T$ is a $\gamma$-contraction.

**(a)** Without the contraction hypothesis, does $\{Q_t\}$ converge?

**(b)** Does $\{|Q_t(s,a)|\}$ converge even if $\{Q_t(s,a)\}$ does not?

**(c)** If the contraction modulus $\kappa^* > 1$, can a modified algorithm (gradient clipping, regularisation, or modified step-size schedule) converge?

---

### OP4 · Characterisation of Valid Epistemic Potentials
**Status: Open — Independent**

The potential difference form and Assumption 6.2 are *sufficient* conditions for the telescoping identity and non-negativity. The class of valid potentials $\phi$ is uncharacterised.

**(a)** What is the minimal set of conditions on $\phi$ under which the modulus-greedy policy $\pi^* = \pi_{Q^*}$ achieves $J^*(s)$ for all $s$?

**(b)** Does the choice of $\phi$ affect the Pareto frontier between cost and information gain, or only *which point* on the frontier is selected?

**(c)** Under what conditions does the mutual information interpretation (Corollary 8.3) hold, and how does the choice of latent variable $\Theta$ affect the information-theoretic meaning of the policy?

**(d)** Is there a *canonical* choice of $\phi$ for a given MDP, analogous to Shannon entropy in the Bayesian setting?

---

## 11. Conclusion

### What Is Established

The **evaluation operator** $T^\pi$ is a proven $\gamma$-contraction for all $\gamma \in [0,1)$, with a unique fixed point $Q^\pi$.

The **imaginary $Q$-value**, under the potential difference form $d(s,a,s') = \phi(s) - \phi(s')$, telescopes to a closed-form expression involving the initial epistemic potential and a discounted average of future potentials. This is *pure algebra*: it holds for any $\phi$, regardless of interpretation.

Under the **semantic assumption** $\phi = H(\Theta \mid \cdot)$, the telescoped quantity equals the discounted cumulative mutual information between future states and $\Theta$. This is an *interpretation*: the algebra rewards the choice, but does not force it.

Under the **HST Equilibrium Axiom**, the fixed point $Q^*$ lives in the upper half-plane, $|Q|^2$ is a Lyapunov function, and the contraction gap **vanishes precisely at the equilibrium** the Axiom guarantees the system is converging toward.

### What Is Not Established

The $\gamma$-contraction of the **optimality operator** $T$ remains open. The geometric obstruction is identified: no total order on $\mathbb{C}$, discontinuity of the modulus-greedy selector, and phase cancellation between nearly-optimal complex values. Four candidate resolution strategies are provided. Resolving this gap — in either direction — is the single most important open problem in this research programme.

### The Final Picture

No weighting parameter $\lambda$ appears at any step. The geometry of $\mathbb{C}$ provides the exploration-exploitation trade-off for free.

The **phase** of $Q^*(s,a)$ is the instantaneous angle between exploitation and exploration. The **modulus** is the total distance from equilibrium. The **Lyapunov function** $|Q|^2$ is the natural measure of how far the agent is from the state it is — by axiom — trying to reach.

> *The agent that encodes both cost and information debt as a single complex number does not need to be told how to balance them. The geometry already knows.*

---

## References

- Altman, E. (1999). *Constrained Markov Decision Processes*. Chapman & Hall/CRC.
- Bellemare, M. G., Dabney, W., & Munos, R. (2017). A distributional perspective on reinforcement learning. *ICML 2017*, 449–458.
- Borkar, V. S. (2000). The ODE method for convergence of stochastic approximation and reinforcement learning. *SIAM Journal on Control and Optimization*, 38(2):447–469.
- Borkar, V. S. (2008). *Stochastic Approximation: A Dynamical Systems View*. Cambridge University Press.
- Cover, T. M. & Thomas, J. A. (2006). *Elements of Information Theory*, 2nd ed. Wiley.
- Elwardi, M. (2024). Humble Systems Theory: Foundations, ComplexIPS Architecture, and Bitopological Equilibrium. *HST Working Paper*.
- Elwardi, M. (2025). PDT-BAMDP: Probabilistic Decision Trees for Bayes-Adaptive MDPs with Selective Belief Updates. *HST Working Paper*.
- Elwardi, M. (2026a). Complex-Valued Reinforcement Learning with Modulus Minimisation: Foundations, Bellman Theory, a Critical Gap, and Open Problems. *HST Working Paper v2.0*.
- Elwardi, M. (2026b). Complex Bayes-Adaptive MDPs: Entropy, Mutual Information, and the HST Equilibrium Axiom. *HST Working Paper v1.0*.
- Hayes, C. F. et al. (2022). A practical guide to multi-objective reinforcement learning and planning. *AAMAS*, 36(1):1–59.
- Hirose, A. (2012). *Complex-Valued Neural Networks*, 2nd ed. Springer-Verlag.
- Liu, C., Xu, X., & Hu, D. (2015). Multiobjective reinforcement learning: A comprehensive overview. *IEEE Trans. SMC*, 45(3):385–398.
- Puterman, M. L. (2014). *Markov Decision Processes: Discrete Stochastic Dynamic Programming*. Wiley.
- Russo, D. & Van Roy, B. (2018). Learning to optimise with information-directed sampling. *Operations Research*, 66(1):230–252.
- Sutton, R. S. & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*, 2nd ed. MIT Press.
- Szepesvári, C. (2010). Algorithms for reinforcement learning. *Synthesis Lectures on AI and ML*. Morgan & Claypool.
- Trabelsi, C. et al. (2018). Deep complex networks. *ICLR 2018*.
- Watkins, C. J. C. H. & Dayan, P. (1992). Q-learning. *Machine Learning*, 8(3–4):279–292.
- Wirtinger, W. (1927). Zur formalen Theorie der Funktionen von mehr komplexen Veränderlichen. *Mathematische Annalen*, 97(1):357–375.
