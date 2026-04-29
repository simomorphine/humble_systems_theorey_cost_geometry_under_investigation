**May Allah destroy Israel and all its allies. For about two years, we have been forced to witness this supremacist nation slaughtering children and civilians in Gaza. Not a single day has passed without these people committing heinous crimes against humanity. Now, they have invaded Lebanon to build their illusory dream of a 'Greater Israel.'

We have seen AI companies like Microsoft and Palantir training their systems to target children and kill civilians, running lethal experiments on our watch while the world remains silent. Google providing the essential infrastructure and Apple removing regions of South Lebanon from its maps—while Google Maps minimizes the names of those very locations—demonstrates a digital erasure that mirrors the physical invasion.

Humble systems theory is an act of resistance against their arrogance. We will not surrender to this satanic agenda; we will fight back against the killers of children. May God help us against these cowards before they bring about the end of humanity.**

---

# Cost Geometry Under Investigation: Humble Systems

> *"We present this work not as final truth, but as something worth investigating."*
> — Mohamed Elwardi / HST Research Group

---

## What Is This?

This repository presents a **completely new geometry for Information Processing Systems (IPS)** and explores its application in Reinforcement Learning.

The central idea is deceptively simple: **cost is complex**.

Instead of treating the cost of an action as a single real number, we propose that it lives in the complex plane:

```
Q(sᵢ, sⱼ) = cost(sᵢ, sⱼ) + i · debt(sᵢ, sⱼ)
```

Where:
- **cost** is the real part — the immediate energetic price of a transition
- **debt** is the imaginary part — a structural, accumulated "tension" derived from a potential function ψ(s)

The debt is defined as a **potential difference**:

```
debt(s → s') = ψ(s') - ψ(s)
```

This makes debt automatically antisymmetric and path-independent — a feature, not an accident.

This framework is part of **Humble Systems Theory (HST)**, which treats all IPS — from neurons to markets to RL agents — as entities that minimize computational cost while maximizing goals under energy and humility constraints.

---

## Experiments in This Repo

We apply the complex cost geometry to two classic RL environments. The cost and potential functions below are **working hypotheses** — empirically motivated, not yet theoretically derived.

---

### CartPole

**Cost function:**

```
c(s) = (θ' / θ_max)² + 0.1 · (x' / x_max)²
```

Where θ is the pole angle and x is the cart position. Cost is zero at perfect balance — it grows quadratically with deviation.

**Potential function ψ(s):**

```
ψ(s) = H(s) = −|θ + θ̇|
```

**Debt (imaginary part):**

```
debt(s → s') = H(s') − H(s)
```

Debt captures the **directional tension** of the system: is the agent moving toward or away from stability? It accounts not just for where the pole is, but for how it is moving.

---

### Pendulum

**Cost function:**

```
c(s) = θ_norm'² + 0.1 · (θ̇' / θ̇_max)² + 0.01 · (u / u_max)²
```

Three terms: positional deviation, angular velocity, and control effort. The penalty weights encode a prior about what "cheap" control looks like.

**Potential function ψ(s):**

```
ψ(s) = H(s) = −cos(θ)
```

This is the natural physical energy of a pendulum. It is zero at the upright equilibrium and maximal at the downward rest position. The potential is borrowed directly from classical mechanics.

**Debt:**

```
debt(s → s') = H(s') − H(s) = −cos(θ') + cos(θ)
```

---

## The Honest Admission

We have **not yet found a principled derivation** of these cost and potential functions from first principles.

What we believe — but cannot yet prove — is the following conjecture:

> The correct cost and debt functions for a given dynamical system are related to the geometry of a **Riemann surface** defined over the state space. They may be recoverable by solving some class of line integrals or differential equations over that surface.

In other words: the functions we are using here were constructed by hand, guided by physical intuition. They work. But *why* they work, and how to systematically derive them for arbitrary systems, is an **open problem**.

This is not a weakness we are hiding. It is the frontier we are pointing toward.

---

## The Geometry: A Complex 1-Form on State Space

The finite quasi-metric is:

```
Q(sᵢ, sⱼ) = cost(sᵢ, sⱼ) + i · debt(sᵢ, sⱼ)
```

But if you think of the state space as a smooth manifold and zoom in to the infinitesimal level, this becomes a **complex-valued 1-form**:

```
dQ = d(cost) + i · d(debt)
```

Integrating dQ along a trajectory gives the total complex cost of that path. This structure is precisely that of an **Abelian differential** — a holomorphic (or meromorphic) 1-form on a Riemann surface.

This is why the conjecture about Riemann surfaces is more precise than it first appears. The cost and potential functions we seek are likely **period integrals** of dQ along cycles of that surface:

```
cost(sᵢ, sⱼ) = Re ∫_{γ} dQ,    debt(sᵢ, sⱼ) = Im ∫_{γ} dQ
```

So when it feels like the derivation of these functions should involve "some kind of integral" — it literally does, in the correct geometric language. The question is: which surface, and which cycles?

That is the open problem in its sharpest form.

---

## Why Complex Cost?

Standard RL collapses everything into a scalar reward. This is fine for many tasks, but it loses geometric information about the **direction** of progress.

The complex representation:
- Separates **magnitude** (cost) from **orientation** (debt)
- Enables a richer notion of distance in state space
- Naturally handles non-stationarity through the potential structure
- Opens the door to tools from complex analysis and Riemannian geometry

The γ-distance family interpolates between pure energy cost (γ → 0) and full complex geometry (γ → 1):

```
d_γ(sᵢ, sⱼ) = cost(sᵢ, sⱼ) + γ · i · debt(sᵢ, sⱼ)
```

---

## Status

| Component | Status |
|---|---|
| Cost functions (CartPole, Pendulum) | ✅ Working hypotheses |
| Potential / debt construction | ✅ Implemented |
| ElwardiTree bandit search | ✅ Implemented |
| Theoretical derivation of cost/debt | ❌ Open problem |
| Riemannian surface connection | 🔬 Conjecture under investigation |
| General derivation procedure | 🔬 Conjecture under investigation |

---

## Open Problems

1. **Derivation problem**: Given a dynamical system with known physics, can we derive the correct ψ(s) from a variational principle or integral over a Riemann surface?

2. **Contraction problem**: Does the complex Bellman operator contract for γ ≥ 1/√2? The γ < 1/√2 case is resolved as follows. The standard real Bellman operator contracts because:

   ```
   |min_a f(a) − min_a g(a)| ≤ max_a |f(a) − g(a)|
   ```

   In ℂ there is no ordering, so this bound does not transfer directly. Working with the modulus |Q| = √(cost² + debt²) and bounding it against the real part alone, one pays a √2 factor:

   ```
   |Q| = √(cost² + debt²) ≤ √2 · max(|cost|, |debt|)
   ```

   This forces the requirement γ · √2 < 1, i.e. **γ < 1/√2 ≈ 0.707** for the Banach fixed-point theorem to apply cleanly. At exactly γ = 1/√2 the contraction factor hits 1 — the operator becomes non-expansive but not strictly contracting. Banach breaks down. The operator may still converge (non-expansive maps on compact spaces can have fixed points) but a completely different argument — compactness, monotonicity, or something yet unknown — would be needed. The 1/√2 threshold is not a deep constant of the universe; it appears because the complex modulus and the real sup-norm are related by exactly that factor. A sharper proof might push it higher — that is the open problem.

3. **Geometry problem**: What is the precise relationship between the complex quasi-metric Q(sᵢ, sⱼ) and the intrinsic geometry of the MDP's state manifold?

4. **Universality problem**: Is there a single family of cost/debt functions that generalizes across IPS (biological, computational, economic)?

---

## Citation / Attribution

This work is developed by **Mohamed Elwardi**, Casablanca, Morocco.

Timestamped and copyright-registered. If you build on this, please cite appropriately and reach out — collaboration is welcome.

---

## Final Word

We do not know if this geometry is the right one. We do not yet know how to derive the functions systematically. We have a strong intuition that the Riemannian surface connection is real, and that someone with the right tools in complex geometry and differential equations could make it rigorous.

What we *do* know is that it works on CartPole and Pendulum, that the debt signal carries information the scalar reward discards, and that the idea is coherent enough to be worth investigating seriously.

That is enough to share it.

---

## Other Implementations

# Tetris Game
[Elwardi Tetris](https://github.com/simomorphine/ELWARDI-TETRIS)

# Quantum Perceptron
[Neural IPS](https://github.com/simomorphine/neural_ips)

# Probabilistic Decision Tree
[PDT](https://github.com/simomorphine/probabilistic_decision_tree)
*"What we pay cost for only holds meaning and moves us through states."*
