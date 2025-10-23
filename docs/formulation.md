


## Problem data and notation
- Grid size: $M \times N$ with both M and N being odd.
- Cell coordinates: rows $r \in \{0, \dots, M-1\}$ and columns $c \in \{0, \dots, N-1\}$.
- Each cell is either a **number** (fixed digit $v_{r,c} \in \mathbb{Z}_{\ge 0})$, a **fixed arrow** (direction in $\{'N','E','S','W'\}$), or a **hidden arrow** to be determined.
- By its layout rules, numbered cells occupy one color of a checkerboard; therefore no two numbered cells are Manhattan distance 1.
- **Line‑of‑sight rule:** An arrow contributes **+1** to **every** numbered cell strictly along its pointing ray until the edge.
- **Edge rule:** Arrows in boundary cells may not point out of the grid (i.e., a boundary arrow must contribute to at least one numbered cell).

Define the set of numbered cells $\mathcal{T} \subseteq \{0, \dots, M-1 \} \times \{0, \dots, N-1\}$ and the set of arrow cells $\mathcal{P}$ (fixed or hidden). For a numbered cell $t=(r,c)$, define the four *visibility sets*
```math
	\begin{align}
	L_t &= \{(r,j) \in \mathcal{P}: j<c\}, \\
	R_t &= \{(r,j) \in \mathcal{P}: j>c\}, \\
	U_t &= \{(i,c) \in \mathcal{P}: i<r\}, \\
	D_t &= \{(i,c) \in \mathcal{P}: i>r\}
	\end{align}
```
These are all arrow cells in the same row/column and on the corresponding side of $t$.

---

## Decision variables
For each arrow cell $p \in \mathcal{P}$, introduce four binary variables
```math
	X_{p}^{N},\; X_{p}^{E},\; X_{p}^{S},\; X_{p}^{W} \in \{0,1\},
```
interpreted as “cell $p$ points in that direction.” If a direction is **impossible by border** (e.g., a top‑edge cell cannot point (N) if that contributes to no number), we remove that literal (or later forbid it via a constraint below).
**Exactly‑one direction** at every arrow cell:
```math
    X_{p}^{N} + X_{p}^{E} + X_{p}^{S} + X_{p}^{W} = 1 \qquad \forall \,p \in \mathcal{P}.\tag{A1}
```
If an arrow is pre‑fixed at $p$ to direction $d$, set $X_{p}^{d}=1$ and all other $X_{p}^{(\ast)}=0$.

---

## Number (balance) constraints
A numbered cell $t=(r,c)$ must receive exactly $v_t$ incoming arrows from all four directions. Using visibility sets and the line‑of‑sight rule, the _incoming_ literals for $t$ are
```math
	A_t = \{X_{p}^{E}: p \in L_{t}\} \; \cup \; \{X_{p}^{W}: p \in R_{t}\} \; \cup \; \{ X_{p}^{S}: p \in U_{t} \} \; \cup \; \{X_{p}^{N}: p \in D_{t}\}
```
The balance equality is then
```math
	\sum_{x \in A_t} x \; = \; v_t \qquad \forall \, t \in \mathcal{T}.\tag{B1}
```
These are linear "= =" cardinality constraints and can be encoded directly in ILP/CP‑SAT (e.g., via sequential counters or sorting networks if using SAT).

---

## Edge feasibility (no “off‑grid” arrows)
A border cell may not point _away_ from the grid when that would fail to see any number. Let $\mathrm{vis}(p,d)$ be the set of numbered cells seen by $p$ when pointing in direction $d$. The rule becomes
```math
	X_{p}^{d} = 0 \quad \text{whenever} \quad \mathrm{vis}(p,d) = \varnothing.\tag{E1}
```
Equivalently: remove those literals from the model.

---

## Complete ILP/CP/SAT model
- **Variables:** $X_p^{N}, X_p^{E}, X_p^{S}, X_p^{W} \in \{0,1\}$ for each $p \in \mathcal{P}$ and allowed direction.
- **Constraints:** `(A1)` for each $p$; `B1` for each $t$; and `E1` for all forbidden border directions.
- **Objective:** none (feasibility).

This is an exact formulation with no greedy steps and no heuristics required to be correct. Modern CP‑SAT/ILP solvers typically solve such instances immediately thanks to strong propagation on (A1) and (B1).

> **Observation (useful in implementation).**
For a position $t=(r,c)$, the equality (B1) splits naturally into horizontal and vertical flows:
```math
v_t \;=\; \underbrace{\sum_{p \,\in \, L_t} X_p^{E}  +  \sum_{p \, \in \, R_t} X_p^{W}}_{\text{row contributions}} \; + \; \underbrace{\sum_{p \, \in \, U_t} X_p^{S} + \sum_{p \in D_t} X_p^{N}}_{\text{column contributions}} \tag{B2}
```
This additivity enables powerful propagation (see the DP/transfer section) and optional row/column prefix logic.

---

## Strong propagation rules (solver‑independent)

While rule-based propagation of constraints already solves most boards, you can push far using generic equality filtering:
- For each number $t$ with candidate literal set $A_t$, maintain lower/upper bounds on each literal ($0/1$ if already fixed). Let
```math
 \begin{align} L_t &= \sum_{x \in A_t} \mathrm{lb}(x) \quad \text{and} \\ U_t &= \sum_{x \in A_t} \mathrm{ub}(x) \end{align}
```
- Enforce $L_t \le v_t \le U_t$. If $L_t = v_t$, set all unfixed literals in $A_t$ to 0. If $U_t=v_t$, set all to 1. Also, for each $x \in A_t$,
    - If $U_t - \mathrm{ub}(x) < v_t$, force $x=1$.
    - If $L_t - \mathrm{lb}(x) > v_t - 1$, force $x=0$.
- Mirror changes into (A1): once one direction at $p$ is set to 1, the other three become 0; if a direction at $p$ becomes 0 and only one remains, that remaining one becomes 1.
- Apply (E1) early to prune illegal border directions.

These rules are sufficient/complete for unit propagation on a SAT/CP encoding and often finish the puzzle without branching.


# Dynamic Programming (Transfer‑Matrix) Approach

We also implement a DP formulation that avoids global searches by sweeping the grid. It's exact and non‑greedy, while complexity is pseudo‑polynomial in the maximum digit and exponential only in a thin frontier whose width is controlled by the checkerboard spacing.

### Key cumulative identity

For any cell position $t=(r,c)$, the total vertical incoming arrows is given by
```math
\underbrace{S^{\uparrow}(r,c)}_{count\text{(S arrows) for } i < r  \text{ in column } c} \; + \;
\underbrace{N^{\downarrow}(r,c)}_{count\text{(N arrows) for } i > r \text{ in column } c}
```
while the horizontal incoming is
```math
\underbrace{E^{\leftarrow}(r,c)}_{count\text{(E arrows) for } j < c \text{ in row } r} \;+\;
\underbrace{W^{\rightarrow}(r,c)}_{count\text{(S arrows) for } j > c \text{ in row } r}
```
Thus (B2) can be written as
```math
	E^{\leftarrow}(r,c) + W^{\rightarrow}(r,c) + S^{\uparrow}(r,c) + N^{\downarrow}(r,c) = v_{r,c} \tag{T1}
```
Each term is a prefix/suffix counter in its row/column.

### Sweep order and DP state
Sweep rows from top to bottom. Let the set of "active" columns be those that contain numbers. By spacing, there are at most $\lceil N/2 \rceil$ such columns.

At the start of row $r$, maintain a state vector
```math
\mathbf{s}_r = \big( s_c \big)_{c \in \mathcal{C}},\qquad s_c = \text{number of S arrows placed so far above } (r,c)
```
This captures all **future** vertical contributions that will hit every number below in column $c$. Arrows pointing N affect previous numbers and must be checked on the fly to avoid overshooting.

Inside row $r$, process cells left to right. For each arrow cell at $(r,j)$:
- Choosing $S$ increments $s_j$ and affects all numbers not yet seen below $(r,j)$.
- Choosing $N$ immediately pays $+1$ into every numbered cell above in column $j$. Because numbers are spaced, these have already been passed in the sweep. We need to ensure none become over‑satisfied. This is checked with stored cumulative sums per column.
- Choosing $E/W$ affects only numbers in the current row. We handle them with a 1D sub‑DP across the row segments between consecutive numbers: for each segment, we maintain prefix counts $E^{\leftarrow}$ and suffix counts $W^{\rightarrow}$ to satisfy the in‑row equations induced by `T1`.

When we encounter a numbered cell $(r,c)$ during the left‑to‑right pass, we evaluate its balance:
```math
	E^{\leftarrow}(r,c) + W^{\rightarrow}(r,c) + s_c + \underbrace{N^{\downarrow}(r,c)}_{\text{current unknown}} = v_{r,c}.
```
This yields a required remainder for the still‑unseen future $N$-arrows below in column $c$:
```math
	\rho_{r,c} \;=\; v_{r,c} - \big(E^{\leftarrow}(r,c) + W^{\rightarrow}(r,c) + s_c \big).\tag{T2}
```
Feasibility demands $0 \le \rho_{r,c} \le N^{\downarrow}(r,c)$. Store $\rho_{r,c}$ as a per‑column target that must be met by later $N$ choices in that column. During the sweep, keep per‑column counters
```math
    \nu_c = \text{number of (N) arrows chosen so far below the last visited number in column }c,
```
and enforce at every step that $\nu_c \le \rho_{r,c}$, and by the time you leave the last row you must have $\nu_c = \rho_{r,c}$ for every column c.

### State size and transitions
- Frontier:
	- Because numbers are equally-spaced, the number of active columns is at most $\lceil N/2 \rceil$, giving pseudo‑polynomial complexity in $v_{\max}$.
	- for each active column $c$, the pair $(s_c \,, \rho_c)$, with bounds $0 \le s_c, \rho_c \le v_{\max}$ and $s_c + \nu_c \le \rho_c + v_{max}$.
- Within each row, the horizontal sub‑DP runs independently on each segment between consecutive numbers and realizes all feasible $E/W$ patterns that meet the exact in‑row portions of (T1). The sub‑DP state is just the pair of prefix/suffix counts hitting the bordering numbers, which are bounded by those numbers’ digits.

### Complexity
Let $C=\lceil N/2 \rceil$ be active columns, and let $v_{\max}$ be the largest digit. The DP has $O(M \cdot v_{\max}^{\,O(C)})$ states in the worst case, but in practice is much smaller because
	1. many columns share tight bounds
	2. segments are short
	3. edge tautologies prune heavily

This is typically competitive with CP‑SAT but stays purely combinatorial and non‑heuristic.
