# Belief Space Geometry: Energy, Debt, and Equilibrium in Information Processing Systems

**Mohamed Elwardi**

---

## Abstract

We introduce a mathematical framework for characterizing the belief space of information processing systems by coupling energy costs with informational debt. By defining a complex-valued quasi-metric $Q(b_i, b_j) = \text{cost}(b_i, b_j) + i \cdot \text{debt}(b_i, b_j)$, where $\text{cost}$ behave like an energy quasi-metric and $\text{debt}$ is derived from a potential function, we uncover a rich geometric structure. The resulting family of metrics, parameterized by $\gamma \in [0, 1]$, continuously interpolates between pure thermodynamic (energy-only) and energy-debt geometries.

We further study how the asymmetry of this gamma distance induces a natural **bitopological structure** on belief space, generating forward and backward topologies $(\tau_+, \tau_-)$ and a hierarchy of equilibrium concepts:

$$E_{\text{avg}} \subseteq E_\vee \subseteq E_\rightarrow \cap E_\leftarrow$$

This framework reveals that information processing systems inhabit a richer mathematical space than previously recognized, with profound implications for computational complexity, optimization, and the thermodynamics of computation.

---

## 1. The Belief Space and Energy Quasi-Metric
**motivation:** Imagine a moving agent evolving in a belief space. The agent aims to transition from a belief state $A$ to a belief state $B$. We observe several key properties: moving from $A$ to $B$ requires an energy cost, while remaining in place—i.e., not updating the belief—incurs no cost. Moreover, this energy cost is always positive.

The central question motivating this work is: what conditions are required for this energy cost to form a quasi-metric (triangular inequality holds)? The answer is subtle. The energy cost may or may not satisfy the properties of a quasi-metric. However, even when it fails to do so, we propose that a *regulation debt* naturally emerges.

Formally, assume that with probability $\alpha$, the cost behaves as a quasi-metric, and with probability (1 - $\alpha$), a regulation debt is generated instead. This leads us to focus on the complex-valued quantity:

$$
\text{cost} + i \cdot \text{debt}.
$$

This object captures both the geometric structure (when quasi-metric properties hold) and the corrective or compensatory dynamics (when they fail), offering a richer framework for understanding motion in belief spaces.

**Definition 1.1 (Belief Space).** Let $\mathcal{B}$ denote the belief space of an information processing system, consisting of all configurations reachable with bounded energy cost. Each belief $b \in \mathcal{B}$ represents a complete specification of the system's configuration.

**Definition 1.2 (Energy Quasi-Metric).** An energy quasi-metric is a function $d : \mathcal{B} \times \mathcal{B} \to \mathbb{R}_{\geq 0}$ satisfying:

1. $d(b_i, b_i) = 0$ for all $b_i \in \mathcal{B}$
2. $d(b_i, b_j) \geq 0$ for all $b_i, b_j \in \mathcal{B}$
3. $d(b_i, b_k) \leq d(b_i, b_j) + d(b_j, b_k)$ for all $b_i, b_j, b_k \in \mathcal{B}$ *(triangle inequality)*

Note that symmetry is **not** required: $d(b_i, b_j)$ may differ from $d(b_j, b_i)$, reflecting the potential irreversibility of computational processes.

---

## 2. The Debt Function

### 2.1 Definition and Motivation

Energy cost alone does not capture the full computational burden of belief transitions. We introduce a complementary notion of **informational debt**.

**Definition 2.1 (Potential Function and Debt).** Let $\psi : \mathcal{B} \to \mathbb{R}$ be a potential function on belief space. The debt function is defined as:

$$\text{debt}(b_i, b_j) = \psi(b_j) - \psi(b_i)$$

The potential $\psi(b)$ quantifies the "computational obligation" carried by belief $b$. A transition from $b_i$ to $b_j$ incurs debt equal to the change in potential: positive debt represents taking on new obligations; negative debt represents paying down existing ones.

### 2.2 Mathematical Properties

**Proposition 2.2 (Properties of Debt).** The debt function satisfies:

1. **Identity:** $\text{debt}(b_i, b_i) = 0$
2. **Additivity (Telescoping):** $\text{debt}(b_i, b_k) = \text{debt}(b_i, b_j) + \text{debt}(b_j, b_k)$
3. **Antisymmetry:** $\text{debt}(b_i, b_j) = -\text{debt}(b_j, b_i)$
4. **Cycle Invariance:** For any closed path $b_1 \to b_2 \to \cdots \to b_n \to b_1$:

$$\sum_{k=1}^{n} \text{debt}(b_k, b_{k+1}) = 0 \quad (b_{n+1} = b_1)$$

5. **Gauge Invariance:** For any constant $c \in \mathbb{R}$, if $\psi'(b) = \psi(b) + c$ for all $b$, then:

$$
\text{debt}_{\psi'}(b_i, b_j) = \text{debt}_ \psi(b_i, b_j)
$$

*Proof.* All properties follow directly from the definition as a potential difference. Additivity: $\text{debt}(b_i, b_j) + \text{debt}(b_j, b_k) = [\psi(b_j) - \psi(b_i)] + [\psi(b_k) - \psi(b_j)] = \psi(b_k) - \psi(b_i)$. Cycle invariance follows by telescoping. Gauge invariance: $[\psi(b_j) + c] - [\psi(b_i) + c] = \psi(b_j) - \psi(b_i)$. $\square$

These properties establish that debt is a **conservative quantity**, fundamentally different from the energy quasi-metric $d$. While $d$ measures irreversible costs, debt is path-independent and reversible.

---

## 3. The Complex Quasi-Metric

**Definition 3.1 (Complex Quasi-Metric).** The complex quasi-metric on belief space is:

$$|Q(b_i, b_j)| = | cost(b_i, b_j) + i \cdot \text{debt}(b_i, b_j) |$$

where $i = \sqrt{-1}$. This packages energy (real part, non-negative) and debt (imaginary part, signed) into a single complex number. The complex structure naturally accommodates their different characters: energy is constrained to the non-negative real axis, while debt ranges over the full imaginary axis.

**Proposition 3.2 (Properties of $Q$).** The complex-valued function $Q$ satisfies:

1. $Q(b_i, b_i) = 0$
2. $\text{Re}(Q(b_i, b_j)) \geq 0$
3. A generalized triangle inequality in $\mathbb{C}$

In polar form, each transition corresponds to a point:

$$Q(b_i, b_j) = |Q(b_i, b_j)|e^{i\theta} , \quad \theta = \arctan\left(\frac{\text{debt}(b_i, b_j)}{cost(b_i, b_j)}\right)$$

where $\theta$ represents the **debt-to-cost ratio** — a natural operating point in the energy-debt trade-off space.

---

## 4. The $\gamma$-Distance Family

### 4.1 Definition

**Definition 4.1 ($\gamma$-Distance).** For $\gamma \in [0, 1]$, the $\gamma$-distance is:

$$d_\gamma(b_i, b_j) = \sqrt{cost(b_i, b_j)^2 + \gamma^2 \cdot \text{debt}(b_i, b_j)^2}$$

At the extremes:
- $\gamma = 0$: $d_\gamma(b_i, b_j) = cost(b_i, b_j)$ — *pure energy metric*
- $\gamma = 1$: $d_\gamma(b_i, b_j) = |Q(b_i, b_j)|$ — *full energy-debt metric*

### 4.2 Metric Properties

**Theorem 4.2 (Triangle Inequality for $d_\gamma$).** For any $\gamma \in [0, 1]$, the $\gamma$-distance satisfies:

$$d_\gamma(b_i, b_k) \leq d_\gamma(b_i, b_j) + d_\gamma(b_j, b_k)$$

*Proof sketch.* For $\gamma = 1$:

$$d_1(b_i, b_k)^2 = cost(b_i, b_k)^2 + \text{debt}(b_i, b_k)^2 \leq [cost(b_i, b_j) + cost(b_j, b_k)]^2 + [\text{debt}(b_i, b_j) + \text{debt}(b_j, b_k)]^2$$

By the Minkowski inequality in $\mathbb{R}^2$, this is bounded by $[d_1(b_i, b_j) + d_1(b_j, b_k)]^2$. The result extends to all $\gamma \in [0, 1]$ by continuity. $\square$

### 4.3 Geometric Interpretation

As $\gamma$ varies from $0$ to $1$, the metric structure of belief space continuously deforms:

- **Neighborhoods:** Metric balls $B_r(b) = \{b' : d_\gamma(b, b') < r\}$ expand or contract differently for different beliefs
- **Geodesics:** Shortest paths between beliefs change as $\gamma$ increases
- **Distances:** Beliefs that are close in the $\gamma = 0$ geometry may be far apart for $\gamma > 0$ if their debt difference is large

**Proposition 4.3 (Topology Dependence).** The metric topology induced by $d_\gamma$ depends continuously on $\gamma$. For $\gamma_1 < \gamma_2$, there exist beliefs $b_i, b_j$ such that $d_{\gamma_1}(b_i, b_j) < d_{\gamma_2}(b_i, b_j)$ if and only if $\text{debt}(b_i, b_j) \neq 0$.

---

## 5. The Asymmetry of $d_\gamma$ and Bitopological Structure

### 5.1 Two Topologies from One Distance

The asymmetry $d_\gamma(b_i, b_j) \neq d_\gamma(b_j, b_i)$ naturally generates **two distinct topologies** on $\mathcal{B}$.

**Definition 5.1 (Forward Topology $\tau_+$).** Basis:

$$B^+(b, \varepsilon) = \{ x \in \mathcal{B} : d_\gamma(b, x) < \varepsilon \} $$

The forward ball $B^+(b, \varepsilon)$ contains beliefs *reachable from* $b$ with cost less than $\varepsilon$.

**Definition 5.2 (Backward Topology $\tau_-$).** Basis:

$$B^-(b, \varepsilon) = \{x \in \mathcal{B} : d_\gamma(x, b) < \varepsilon\}$$

The backward ball $B^-(b, \varepsilon)$ contains beliefs *from which* $b$ *is reachable* with cost less than $\varepsilon$.

**Definition 5.3 (Bitopological Space).** The triple $(\mathcal{B}, \tau_+, \tau_-)$ is the **bitopological space** of the system. We write $(\mathcal{B}, \rightarrow)$ for $(\mathcal{B}, \tau_+)$ and $(\mathcal{B}, \leftarrow)$ for $(\mathcal{B}, \tau_-)$.

**Proposition 5.4.** If $\text{debt} \equiv 0$, then $\tau_+ = \tau_-$. Otherwise, neither topology is generally finer than the other.

**Definition 5.5 (Join Topology).** The join topology $\tau_+ \vee \tau_-$ is the coarsest topology containing both $\tau_+$ and $\tau_-$, with basis:

$$\mathcal{B}^\vee = \{B^+(b, \varepsilon) \cap B^-(b, \delta) : b \in \mathcal{B},\; \varepsilon, \delta > 0\}$$

A neighborhood in the join topology requires both forward and backward proximity simultaneously.

---

## 6. Equilibrium Structure

### 6.1 Symmetrization and the Average Distance

**Definition 6.1 (Average Distance).**

$$d_{\text{avg}}(b, x) = d_\gamma(b, x) + d_\gamma(x, b)$$

This measures the **round-trip cost**: going from $b$ to $x$ and back.

**Proposition 6.2.** The average distance $d_{\text{avg}}$ is a genuine metric on $\mathcal{B}$.

*Proof.* Symmetry is immediate. Triangle inequality: $d_{\text{avg}}(b, y) \leq d_{\text{avg}}(b, x) + d_{\text{avg}}(x, y)$ follows from applying the triangle inequality of $d_\gamma$ to each direction separately. $\square$

**Definition (Max Distance)**

$$d_{\text{max}}(b, x) = max{d_\gamma(b, x) , d_\gamma(x, b)}$$

**Proposition**

$d_\gamma$ and $d_{\text{max}}$ are equivalent distances, and they define the same topology on $B$.

*Proof.* 

$$d_\gamma \leq d_{\text{max}} \leq 2d_\gamma$$

**Proposition 6.3 (Intersection Property).**

$$B^+(b, \varepsilon) \cap B^-(b, \varepsilon) \subseteq B_{\text{avg}}(b, 2\varepsilon)$$

Beliefs in the intersection $B^+(b, \varepsilon) \cap B^-(b, \varepsilon)$ are *bidirectionally close* — they are both reachable from $b$ and can reach $b$ with low cost. These are the beliefs in approximate equilibrium with $b$.

**Theorem 6.4 (Topological Hierarchy).**

$$\tau_{\text{avg}} \subseteq \tau_+ \vee \tau_-$$

The average topology is coarser (fewer open sets) than the join topology.

### 6.2 Four Equilibrium Concepts

**Definition 6.5 (Pre-equilibrium Set).**

$$U_{\text{pre}} = \bigcup_{b \in \mathcal{B}} B_{\text{avg}}(b, \varepsilon_b)$$

for thresholds $\varepsilon_b > 0$.

Taking the closure of $U_{\text{pre}}$ in different topologies yields four distinct equilibrium concepts:

| Concept | Definition | Interpretation |
|---|---|---|
| $E_{\text{avg}} = \overline{U_{\text{pre}}}^{\tau_{\text{avg}}}$ | Average equilibrium | Debt asymmetry vanishes locally |
| $E_\vee = \overline{U_{\text{pre}}}^{\tau_+ \vee \tau_-}$ | Join equilibrium | Bidirectional accessibility holds |
| $E_\rightarrow = \overline{U_{\text{pre}}}^{\tau_+}$ | Forward equilibrium | Forward-reachable from balanced beliefs |
| $E_\leftarrow = \overline{U_{\text{pre}}}^{\tau_-}$ | Backward equilibrium | Can trace backward to balanced beliefs |

### 6.3 The Equilibrium Hierarchy

**Theorem 6.6 (Equilibrium Hierarchy).**

$$E_{\text{avg}} \subseteq E_\vee \subseteq E_\rightarrow \cap E_\leftarrow$$

*Proof.* Since $\tau_{\text{avg}} \subseteq \tau_+ \vee \tau_-$, a set closed in the join topology is also closed in the average topology, giving $E_{\text{avg}} \subseteq E_\vee$. Since $\tau_+ \vee \tau_-$ is coarser than both $\tau_+$ and $\tau_-$ individually, $E_\vee \subseteq E_\rightarrow \cap E_\leftarrow$. $\square$

### 6.4 Characterization via Vanishing Asymmetry

**Theorem 6.7.** A belief $b$ is in $E_{\text{avg}}$ if and only if:

$$\liminf_{x \to b} \frac{|d_\gamma(b, x) - d_\gamma(x, b)|}{d_{\text{avg}}(b, x)} = 0$$

The relative asymmetry in $\gamma$-distance vanishes locally around beliefs in $E_{\text{avg}}$. In terms of debt, since $d_\gamma(b, x) - d_\gamma(x, b) \approx \frac{2\,\text{debt}(b, x)}{d_\gamma}$ when $d$ is fixed and debt changes sign, near $E_{\text{avg}}$ the debt component is small relative to total distance.

**Proposition 6.8 (Fixed Points).** Every fixed point $b^*$ (where $Q(b^*, b^*) = 0$) belongs to all four equilibrium subspaces.

---

## 7. Optimization in $\gamma$-Space

**Definition 7.1 (Optimal Path).** For given $\gamma$, an optimal path from $b_i$ to $b_j$ is a sequence $b_i = b_0 \to b_1 \to \cdots \to b_n = b_j$ minimizing:

$$\sum_{k=0}^{n-1} d_\gamma(b_k, b_{k+1})$$

As $\gamma$ increases from $0$ to $1$, optimal paths shift. An energy-efficient but debt-accumulating path becomes disfavored; a path that pays down debt may become preferred despite higher energy cost.

This gives rise to three algorithm classes:

- **Energy-efficient:** Minimize $d(b_i, b_j)$ regardless of debt
- **Debt-efficient:** Minimize $|\text{debt}(b_i, b_j)|$ regardless of energy  
- **$\gamma$-efficient:** Minimize $d_\gamma(b_i, b_j)$ for a specific $\gamma$

---

## 8. Examples

### 8.1 Compression System

Consider data compression with beliefs $b$ representing data structures.

- $d(b, x)$: computational cost to transform $b$ to $x$
- $\text{debt}(b, x)$: information loss (negative if losing information)

Lossy compression has $\text{debt}(b_{\text{full}}, b_{\text{compressed}}) < 0$. Decompression has $\text{debt}(b_{\text{compressed}}, b_{\text{full}}) > 0$. Lossless schemes have $\text{debt} \approx 0$ and lie in $E_{\text{avg}}$.

### 8.2 Gradient Flow

For a function $J : \mathcal{B} \to \mathbb{R}$:

- $d(b, x) = \|b - x\|$
- $\text{debt}(b, x) = J(x) - J(b)$

Then $d_\gamma(b, x) = \sqrt{\|b - x\|^2 + (J(x) - J(b))^2}$ measures distance plus change in $J$. Critical points of $J$ (where $\nabla J = 0$) satisfy $\text{debt}(b, x) \approx 0$ locally and lie in $E_{\text{avg}}$.

### 8.3 Thermodynamic System

- $d(b, x)$: energy cost of transition
- $\text{debt}(b, x)$: entropy production (negative if spontaneous)

The second law: spontaneous processes have $\text{debt} < 0$. Equilibrium beliefs (maximum entropy) have $\text{debt} \approx 0$ for nearby perturbations.

---

## 9. Open Questions

1. **Explicit potentials:** Can we construct explicit $\psi$ for concrete systems (neural networks, reversible computers, quantum circuits)?
2. **Determining $\gamma$:** What determines the effective $\gamma$ for a given system? Is it fundamental or tunable?
3. **Hamiltonian formulation:** Does the complex structure of $Q$ suggest a natural Hamiltonian dynamics on belief space?
4. **Geometry of $E_{\text{avg}}$:** Is $E_{\text{avg}}$ dense in $\mathcal{B}$, connected, or a manifold?
5. **Conservation laws:** Are there conserved quantities along trajectories in $Q$-space analogous to energy-momentum conservation?
6. **Statistical mechanics:** Can we define a "$\gamma$-temperature" and formulate statistical mechanics in the $\gamma$-geometry?
7. **Information-theoretic bounds:** Can Landauer's principle and holographic bounds be reformulated using the debt function?
8. **Convergence dynamics:** Under what dynamics on $\mathcal{B}$ do trajectories converge to $E$?
9. **Biological systems:** Do biological information processors operate at characteristic $\gamma$ values?
10. **Strict inclusion:** When is $E_{\text{avg}} \subsetneq E_\vee$? What properties of the debt function determine whether equilibrium concepts coincide?

---

## 10. Conclusion

We have developed a unified mathematical framework extending the geometry of information processing systems beyond pure energy considerations. The key objects are:

- A **potential-derived debt function** $\text{debt}(b_i, b_j) = \psi(b_j) - \psi(b_i)$ with conservative, antisymmetric, and cycle-invariant properties
- A **complex quasi-metric** $Q(b_i, b_j) = d(b_i, b_j) + i \cdot \text{debt}(b_i, b_j)$ packaging energy and debt into a single geometric object
- A **one-parameter family** $d_\gamma$ interpolating between pure energy ($\gamma = 0$) and full energy-debt ($\gamma = 1$) geometries
- A **bitopological space** $(\mathcal{B}, \tau_+, \tau_-)$ arising from the asymmetry of $d_\gamma$, generating a natural hierarchy of equilibrium concepts

The hierarchy $E_{\text{avg}} \subseteq E_\vee \subseteq E_\rightarrow \cap E_\leftarrow$ captures distinct meanings of balance, from perfect local symmetry ($E_{\text{avg}}$) to directional reachability ($E_\rightarrow, E_\leftarrow$). This structure emerges naturally whenever a system optimizes simultaneously in two directions — it is not imposed, but discovered.

---

## References

1. R. Landauer, "Irreversibility and heat generation in the computing process," *IBM Journal of Research and Development*, vol. 5, no. 3, pp. 183–191, 1961.
2. C. H. Bennett, "The thermodynamics of computation — a review," *International Journal of Theoretical Physics*, vol. 21, no. 12, pp. 905–940, 1982.
3. S. Amari, *Information Geometry and Its Applications*. Springer, 2016.
4. J. C. Kelly, "Bitopological spaces," *Proceedings of the London Mathematical Society*, 3(1):71–89, 1963.
5. P. Fletcher and W. F. Lindgren, *Quasi-Uniform Spaces*. Marcel Dekker, New York, 1982.
6. R. Kopperman, "Asymmetry and duality in topology," *Topology and its Applications*, 66(1):1–39, 1995.
