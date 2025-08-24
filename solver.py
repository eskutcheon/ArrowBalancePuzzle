
from typing import Dict, List, Optional, Tuple, Callable, Union
# importing classes used to define the puzzle structure
from structs import Pos, Direction, Puzzle
from utils import get_allowed_directions


CellDirection = Tuple[Pos, Direction]  # a single boolean variable "pos has arrow in direction ___"


def count_solutions(grid: List[List[str]], limit: int = 2) -> int:
    """ Use the existing solver to count up to 'limit' solutions for the given puzzle.
        Early-outs once 'limit' is reached. Helpful for uniqueness tests.
    """
    puz = Puzzle.from_grid(grid)
    solver = ArrowCSP(puz)
    # Shallow wrapper over ArrowCSP.search with early cutoff.
    count = 0
    def dfs() -> bool:
        nonlocal count
        if not solver._propagate_to_fixpoint():
            return True  # dead branch; keep going
        if solver._is_decided():
            if not solver._verify_numbers_match():
                return True  # invalid solution, discard and keep searching
            count += 1
            return count < limit  # stop exploring when we hit the cap
        p = solver._pick_branch_cell()
        # if p is None:
        #     # Should not happen (check with `is_decided` handled above), but to be safe:
        #     print("CHECKING IF THIS EVER RUNS")
        #     count += 1
        #     return count < limit
        snapshot = solver._snapshot_domains()
        for lit in solver.arrow_dirs[p]:
            if solver.lb[lit] == 0 and solver.ub[lit] == 1:
                solver._assume_literal_true(lit)
                if not dfs():
                    return False  # cutoff
                solver._restore_domains(snapshot)
        return True
    dfs()
    return count


def solve_grid(grid: Union[List[List[str]], Puzzle]) -> Optional[List[List[str]]]:
    """ Solve the puzzle described by the grid and return a completed grid (same shape)
        whose arrow cells are filled with 'N','E','S','W'. Returns None if unsatisfiable.
    """
    puz = Puzzle.from_grid(grid) if not isinstance(grid, Puzzle) else grid
    solver = ArrowCSP(puz)
    sol = solver.solve()
    if sol is None:
        return None
    # Rebuild the grid with directions
    R, C = puz.rows, puz.cols
    out = [[grid[r][c] for c in range(C)] for r in range(R)]
    for p in puz.arrow_cells:
        out[p.r][p.c] = sol[p].value
    return out


class ArrowCSP:
    """ Constraint model and propagator for the 'count incoming arrows' puzzle
        Variables:
            - For each arrow cell p and direction d in {N,E,S,W} we keep a boolean domain:
                lb[p,d] in {0,1}, ub[p,d] in {0,1} with lb <= ub and lb==ub when decided.
        Constraints:
            1) Exactly-one per arrow cell: sum_d X_{p,d} = 1
            2) For each numbered cell t:   sum_{(p,d) sees t} X_{p,d} = value(t)
        Propagation:
            - Exact-one filtering per cell
            - Per-number linear-equality bounds filtering (GAC-style)
    """
    def __init__(self, puzzle: Puzzle):
        self.puz = puzzle
        # Boolean variable domains: lb/ub maps (naming conventions from linear programming)
        self.lb: Dict[CellDirection, int] = {}
        self.ub: Dict[CellDirection, int] = {}
        # TODO: rename later to reflect deprecation of "literal" terminology
        self.lit_to_numbers: Dict[CellDirection, List[Pos]] = {}
        # Indexing helpers
        self.arrow_dirs: Dict[Pos, List[CellDirection]] = {}     # pos -> 4 CellDirections (p,d)
        self.num_candidates: Dict[Pos, List[CellDirection]] = {} # number pos -> inbound CellDirections
        # Work queue for incremental propagation
        self._queue_numbers: List[Pos] = []
        self._queue_arrows: List[Pos] = []
        # TODO: rename function - Literal used to refer to CellDirection but that's a terrible idea with typing.Literal being ubiquitous
        self._build_literals()
        self._index_visibility()
        self._seed_fixed()
        # (Optional future extension) row/col runs:
        self._maybe_build_runs()

    ############################### public API endpoint ###############################

    def solve(self) -> Optional[Dict[Pos, Direction]]:
        """ Run propagation to fixpoint, then (if needed) a tiny backtracking search.
            Returns a mapping of every arrow cell to its decided Direction, or None if unsat.
        """
        if not self._propagate_to_fixpoint():
            return None
        #& UPDATE: now verifying numbers match before returning for extra redundancy
        if self._is_decided():
            if self._verify_numbers_match():
                return self._extract_solution()
            return None
        # Tiny, guided search (kept small thanks to strong propagation)
        return self._search()

    ############################### building & indexing ###############################

    # TODO: rename function - literals used to refer to CellDirection
    def _build_literals(self) -> None:
        """ Initialize lb/ub for each CellDirection and the arrow->CellDirections index. """
        for p in self.puz.arrow_cells:
            li = []
            for d in Direction.all():
                # TODO: need to update to exclude invalid directions on edges
                lit = (p, d)
                self.lb[lit] = 0
                # self.ub[lit] = 1 if d in allowed directions else 0
                self.ub[lit] = int(d.value in get_allowed_directions(p.r, p.c, self.puz.rows, self.puz.cols))
                li.append(lit)
            self.arrow_dirs[p] = li


    # TODO: extract from this class and generalize to a utility function that also performs validation
        # UPDATE: added the function `loop_in_direction` to utils.py which can handle arbitrary index-based loop conditions and callback actions
        # doesn't really account for the current use of `Pos` and class variables in this function but _index_visibility should be able to be adapted
        # honestly kind of hating the way CellDirection is used here, as well as the unnecessary complexity of using the Enum for now
    def _index_visibility(self) -> None:
        """ For each numbered cell t, compute the set of inbound CellDirections that can see t:
            - left of t, the E CellDirections
            - right of t, the W CellDirections
            - above t, the S CellDirections
            - below t, the N CellDirections
        """
        R, C = self.puz.rows, self.puz.cols
        for t, _ in self.puz.numbers.items():
            inbound: List[CellDirection] = []
            # search left of position (same row, c'<t.c), arrows must point East
            c = t.c - 1
            while c >= 0:
                p = Pos(t.r, c)
                if p in self.arrow_dirs:
                    inbound.append((p, Direction.E))
                c -= 1
            # search right (arrows must point West)
            c = t.c + 1
            while c < C:
                p = Pos(t.r, c)
                if p in self.arrow_dirs:
                    inbound.append((p, Direction.W))
                c += 1
            # search above (arrows must point South)
            r = t.r - 1
            while r >= 0:
                p = Pos(r, t.c)
                if p in self.arrow_dirs:
                    inbound.append((p, Direction.S))
                r -= 1
            # search below (arrows must point North)
            r = t.r + 1
            while r < R:
                p = Pos(r, t.c)
                if p in self.arrow_dirs:
                    inbound.append((p, Direction.N))
                r += 1
            self.num_candidates[t] = inbound
            #& UPDATE: populate the reverse index for (CellDirection -> numbers) to reduce enqueue cost to O(1)
            # after building inbound list per t, also populate reverse map:
            for lit in inbound:
                self.lit_to_numbers.setdefault(lit, []).append(t)

    def _seed_fixed(self) -> None:
        """ Apply pre-placed arrows and enforce '0' numbers as immediate bans, enqueue affected. """
        # Fixed arrows
        for p, d in self.puz.fixed_arrows.items():
            for lit in self.arrow_dirs[p]:
                # TODO: need to update to exclude invalid directions on edges
                self._set_lb(lit, int(lit[1] == d))
                self._set_ub(lit, int(lit[1] == d))
            self._enqueue_arrow(p)
        # For any '0' numbered cell, forbid all inbound CellDirections
        for t, v in self.puz.numbers.items():
            if v == 0:
                for lit in self.num_candidates[t]:
                    self._set_ub(lit, 0)
                self._enqueue_number(t)

    def _maybe_build_runs(self) -> None:
        """ Hook for row/column 'run' tautologies discussed in the design doc.
            Currently a no-op (the core propagation is enough for now), but added it for drop-in difference-equality constraints
            without touching the rest of the solver.
        """
        pass

    ############################### propagation engine ###############################

    def _propagate_to_fixpoint(self) -> bool:
        """ Process queues until stable. Returns False if a contradiction is found. """
        # Initially, everything is "dirty"
        if not self._queue_numbers and not self._queue_arrows:
            self._queue_numbers = list(self.puz.numbers.keys())
            self._queue_arrows = list(self.puz.arrow_cells)
        while self._queue_numbers or self._queue_arrows:
            while self._queue_numbers:
                t = self._queue_numbers.pop()
                if not self._propagate_number(t):
                    return False
            while self._queue_arrows:
                p = self._queue_arrows.pop()
                if not self._propagate_arrow_exactly_one(p):
                    return False
        return True

    def _propagate_arrow_exactly_one(self, p: Pos) -> bool:
        """ Maintain sum_d X_{p,d} = 1 via domain filtering, e.g. if three directions are 0, the last must be 1 """
        lits = self.arrow_dirs[p]
        lbs = [self.lb[l] for l in lits]
        ubs = [self.ub[l] for l in lits]
        if sum(lbs) > 1 or sum(ubs) < 1:
            return False  # contradiction
        # if one direction is already lb=1, others must be ub=0
        if sum(lbs) == 1:
            for lit, lb in zip(lits, lbs):
                if lb == 0 and self.ub[lit] != 0:
                    self._set_ub(lit, 0)
                    self._enqueue_neighbors(lit)
        # if three directions have ub=0, the remaining must be lb=1
        if ubs.count(0) == 3:
            idx = ubs.index(1)
            lit = lits[idx]
            if self.lb[lit] == 0:
                self._set_lb(lit, 1)
                self._enqueue_neighbors(lit)
        return True

    def _propagate_number(self, t: Pos) -> bool:
        """ Per-number equality filtering on sum of inbound CellDirections and per-CellDirection necessity/impossibility checks
            ```sum lb <= target <= sum ub```
        """
        def set_cand_if_changed(lit: CellDirection, v: int, is_lb: bool, condition: Callable[[None], bool]) -> bool:
            """ Helper to reduce repetition - Sets lb or ub of lit to v if condition() is true, enqueues neighbors, and returns True if changed. """
            if condition():
                if is_lb:
                    self._set_lb(lit, v)
                else:
                    self._set_ub(lit, v)
                self._enqueue_neighbors(lit)
                return True
            return False
        # set target and candidates and begin filtering
        target = self.puz.numbers[t]
        cand = self.num_candidates[t]
        if not cand:
            # If target is nonzero but nothing can see it, it's a contradiction
            return target == 0
        sum_lb = sum(self.lb[l] for l in cand)
        sum_ub = sum(self.ub[l] for l in cand)
        if sum_lb > target or sum_ub < target:
            return False
        changed = False
        # Tightness cases
        if sum_lb == target:
            # everything else must be 0
            for l in cand:
                changed |= set_cand_if_changed(l, 0, False, lambda: self.lb[l] == 0 and self.ub[l] == 1)
        if sum_ub == target:
            # all undecided must be 1
            for l in cand:
                changed |= set_cand_if_changed(l, 1, True, lambda: self.lb[l] == 0 and self.ub[l] == 1)
        # Per-CellDirection test: try lb=0 and lb=1 hypotheticals on bounds
        for l in cand:
            if self.lb[l] == self.ub[l]:
                continue  # already fixed
            # if setting l=0 would make sum_ub - 1 < target, then l must be 1
            if set_cand_if_changed(l, 1, True, lambda: (sum_ub - 1) < target):
                changed = True
                # update sums for subsequent checks
                sum_lb += 1 # increment lb sum; ub sum unchanged
            # if setting l=1 would make sum_lb + 1 > target, then l must be 0
            elif set_cand_if_changed(l, 0, False, lambda: (sum_lb + 1) > target):
                changed = True
                sum_ub -= 1 # increment ub sum; lb sum unchanged
        if changed:
            # Re-enqueue this number because its sums changed
            self._enqueue_number(t)
        return True


    def _search(self) -> Optional[Dict[Pos, Direction]]:
        """ Minimal backtracking - branch on an arrow cell with the smallest number of remaining directions """
        p = self._pick_branch_cell()
        if p is None:
            # Should be decided; just extract.
            return self._extract_solution()
        # Capture current domains for backtracking
        snapshot = self._snapshot_domains()
        # try each available direction
        for lit in self.arrow_dirs[p]:
            if self.lb[lit] == 0 and self.ub[lit] == 1:
                # assume lit = 1, siblings 0
                self._assume_literal_true(lit)
                if self._propagate_to_fixpoint():
                    if self._is_decided():
                        #? NOTE: not sure if I should add the verification step here or not - adds a lot of overhead but could lead to more efficient branching
                        sol = self._extract_solution()
                    else:
                        sol = self._search()
                    if sol is not None:
                        return sol
                # backtrack
                self._restore_domains(snapshot)
        return None

    # TODO: rename function - literals used to refer to CellDirection types
    def _assume_literal_true(self, lit: CellDirection) -> None:
        """ Force a CellDirection literal to true and all siblings of its cell to false; enqueue effects. """
        p, d = lit
        for l in self.arrow_dirs[p]:
            if l == lit:
                if self.lb[l] == 0:
                    self._set_lb(l, 1)
                # NOTE: checking if self.ub[l] == 0 is a contradiction that should be caught in propagation
            elif self.ub[l] == 1:
                self._set_ub(l, 0)
                #& UPDATED: numbers that depended on this sibling need rechecking, so enqueue number neighbors before moving on
                self._enqueue_neighbors(l)
        self._enqueue_arrow(p)
        self._enqueue_neighbors(lit)

    ############################### utilities ###############################
    # TODO: think I should toss some of these that aren't used often and move some others to utils.py

    def _enqueue_number(self, t: Pos) -> None:
        self._queue_numbers.append(t)

    def _enqueue_arrow(self, p: Pos) -> None:
        self._queue_arrows.append(p)

    def _enqueue_neighbors(self, lit: CellDirection) -> None:
        """ When a CellDirection changes, recheck:
            - the arrow cell's exact-one,
            - every numbered cell that uses this CellDirection.
        """
        p, _ = lit
        self._enqueue_arrow(p)
        # for t, cands in self.num_candidates.items():
        #     if lit in cands:
        #         self._enqueue_number(t)
        #& UPDATE: reduce enqueue cost to O(1) by checking the new reverse index mapping
        for t in self.lit_to_numbers.get(lit, []):
            self._enqueue_number(t)

    def _set_lb(self, lit: CellDirection, v: int) -> None:
        if self.lb[lit] != v:
            self.lb[lit] = v

    def _set_ub(self, lit: CellDirection, v: int) -> None:
        if self.ub[lit] != v:
            self.ub[lit] = v

    def _is_decided(self) -> bool:
        """ True if every arrow cell has exactly one direction fixed (lb==ub==1). """
        for p, lits in self.arrow_dirs.items():
            if sum(self.lb[l] for l in lits) != 1:
                return False
            if any(self.lb[l] != self.ub[l] for l in lits):
                return False
        return True

    def _verify_numbers_match(self) -> bool:
        """ Debugging utility: verify that every number's inbound count matches its target. """
        for t, candidates in self.num_candidates.items():
            count = sum(self.lb[l] for l in candidates)
            if count != self.puz.numbers[t]:
                return False # indicating the solver should discard and keep searching
        return True

    def _extract_solution(self) -> Optional[Dict[Pos, Direction]]:
        """ Return the final mapping p -> Direction, or None if any cell undecided. """
        sol: Dict[Pos, Direction] = {}
        for p, lits in self.arrow_dirs.items():
            chosen = [d for (pp, d) in lits if self.lb[(pp, d)] == 1 and self.ub[(pp, d)] == 1]
            if len(chosen) != 1:
                return None
            sol[p] = chosen[0]
        return sol

    def _pick_branch_cell(self) -> Optional[Pos]:
        """ Choose an arrow cell with the fewest remaining options (>1), to keep search tiny. """
        best_p: Optional[Pos] = None
        best_k = 5  # more than 4
        for p, lits in self.arrow_dirs.items():
            k = sum(1 for l in lits if self.lb[l] == 0 and self.ub[l] == 1) + \
                sum(1 for l in lits if self.lb[l] == 1 and self.ub[l] == 1)  # if already 1, k=1
            if k == 1:
                continue
            if k < best_k:
                best_k = k
                best_p = p
        return best_p

    def _snapshot_domains(self) -> Dict[CellDirection, Tuple[int, int]]:
        """ Deep-copy lb/ub for backtracking """
        return {l: (self.lb[l], self.ub[l]) for l in self.lb.keys()}

    def _restore_domains(self, snap: Dict[CellDirection, Tuple[int, int]]) -> None:
        """ Restore lb/ub; clear queues (and enqueue relevant sets next) """
        for l, (lo, hi) in snap.items():
            self.lb[l] = lo
            self.ub[l] = hi
        self._queue_numbers.clear()
        self._queue_arrows.clear()
        # After restoring, we need to re-enqueue all constraints touched by any change.
        # For simplicity, re-enqueue everything (still very fast on these sizes).
        self._queue_numbers = list(self.puz.numbers.keys())
        self._queue_arrows = list(self.puz.arrow_cells)


