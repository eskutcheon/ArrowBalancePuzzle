# src/solver/solver_dp.py
from collections import defaultdict
import heapq
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Iterable, Union
# from itertools import groupby
from ..structs import Puzzle, Direction, RowSegment
from ..utils import get_allowed_directions

# TODO: need to add these types to a single common location, otherwise there could be a inconsistency problems between files later
Pos = Tuple[int,int]
PuzzleType = Union[List[List[str]], Puzzle]
CellDirectionDict = Dict[Pos, Direction]  # a single boolean variable "pos has arrow in direction ___"


# class DPSolverError(Exception):
#     """ Custom exception for DP solver errors """
#     pass

# class DPSolverTimeout(Exception):
#     """ Custom exception for DP solver timeouts """
#     pass

class DPPuzzleSolver:
    """ Transfer / sweep DP solver for Arrow-Balance puzzles
        Public API:
            - DPPuzzleSolver(puzzle).solve() -> Optional[List[List[str]]]
        Notes:
        - Minimal state - only per-run sweep structures are kept; helpers are static/functional where practical
        - Falls back to CSP when no horizontal plan exists for a row or if it detects an infeasible frontier
    """
    def __init__(
            self,
            puzzle: Puzzle,
            *,
            # rng_seed: Optional[int] = None,
            max_frontier: Optional[int] = None,   # optional limit on (num active columns) to keep DP bounded
            prefer_south: bool = True             # vertical tiebreak to stabilize sweep
        ) -> None:
            self.puz = puzzle
            self.R, self.C = puzzle.rows, puzzle.cols
            #!!! FIXME: always ends up using the fallback at the moment - need to revise the default max_frontier logic
            self.max_frontier = max_frontier if max_frontier is not None else self.C #(self.C + 1) // 2
            self.prefer_south = prefer_south
            # sweep state (reset every solve)
            self._chosen: CellDirectionDict = {}
            # per-column vertical accumulators: s_c (#S above current row), n_cap_c (residual capacity for N below)
            self._s_above: List[int] = [0] * self.C
            self._n_cap: List[int] = [10**9] * self.C   # conservative cap; tighten when crossing numbers


    def solve(self) -> Optional[List[List[str]]]:
        """ return fully-solved grid (strings) or None. Falls back to CSP on stubborn rows/frontiers """
        # quick guard on frontier size (columns that actually contain numbers)
        active_cols = self._col_active_numbers(self.puz)
        if len(active_cols) > self.max_frontier:
            return self._fallback_csp(warning_msg="Number of active columns too large. Falling back to CSP solver...")
        for r in range(self.R):
            segments = self._row_segments_with_numbers(self.puz, r)
            # enumerate EW assignments per segment and assemble one valid row-wide plan
            row_plan = self._pick_row_ew_plan(segments)
            if row_plan is None:
                return self._fallback_csp()
            # commit the horizontal EW choices for this row
            for p, d in row_plan.items():
                self._chosen[p] = d
            # fill remaining arrow cells in row r with vertical choices that respect column state
            if not self._fill_vertical_row(r):
                return self._fallback_csp()
            # after finishing row r, update column residual info when we *pass* numbers in row r
            self._update_column_residuals_on_row_exit(r)
        # build the solved grid and validate via CSP to be safe
        out = [row[:] for row in self.puz.to_grid()]
        for (rr, cc), d in self._chosen.items():
            out[rr][cc] = d.value
        # if DP assigned everything, CSP should accept immediately; otherwise CSP will fill in whatever's left
        # solution = self._fallback_csp(out)
        solution = self._fallback_csp(self.puz)
        self._reset()
        return solution


    def _fallback_csp(self, puzzle: PuzzleType = None, warning_msg: Optional[str] = None) -> Optional[List[List[str]]]:
        # Defer import to avoid hard dependency; keep existing naming compatibility with your project structure.
        if warning_msg is not None:
            print(f"[WARNING] {warning_msg}")
        puz = puzzle if puzzle is not None else self.puz
        from .solver_csp import solve_grid
        return solve_grid(puz)

    #? NOTE: not sure how I want to use this yet, but it'll help avoid confusing state across multiple solve() calls
    def _reset(self) -> None:
        """ Reset sweep state """
        self._chosen = {}
        self._s_above = [0] * self.C
        self._n_cap = [10**9] * self.C
        # TODO: may want to reset the puzzle itself, but still need to think about how to handle multiple solve() calls

    # horizontal (EW) planning for a row

    def _pick_row_ew_plan(self, segments: List[RowSegment]) -> Optional[CellDirectionDict]:
        """ Build EW choices for all segments in a row, pruning by legality only (i.e., doesn't yet push the full equality for bordering numbers)
            - might implement row-difference equalities later
        """
        # Enumerate EW options per segment and do a small DFS across segments.
        per_seg_options: List[List[CellDirectionDict]] = [self._enumerate_ew_options(seg.run) for seg in segments]
        # trivial fast path
        if all(len(opts) == 1 for opts in per_seg_options):
            merged: CellDirectionDict = {}
            for opts in per_seg_options:
                merged.update(opts[0])
            return merged
        merged: CellDirectionDict = {}

        @lru_cache(maxsize=None)
        def dfs(i: int) -> bool:
            if i == len(per_seg_options):
                return True
            for plan in per_seg_options[i]:
                merged.update(plan)
                if dfs(i + 1):
                    return True
                for p in plan:
                    merged.pop(p, None)
            return False

        return merged if dfs(0) else None

    def _enumerate_ew_options(self, run: Iterable[Pos]) -> List[CellDirectionDict]:
        """ Return all EW-only assignments for the short run (checkerboard => short) """
        run = tuple(run)
        if not run:
            return [dict()]
        out: List[CellDirectionDict] = []
        # iterative stack avoids recursion overhead
        stack: List[Tuple[int, CellDirectionDict]] = [(0, {})]
        while stack:
            i, cur = stack.pop()
            if i == len(run):
                out.append(cur.copy())
                continue
            p = run[i]
            r, c = p
            allowed = get_allowed_directions(r, c, self.R, self.C)
            for d in (Direction.E, Direction.W):
                if d.value in allowed:
                    cur[p] = d
                    stack.append((i + 1, cur.copy()))
                    cur.pop(p, None)
        return out

    # functions for vertical placement on a row

    def _fill_vertical_row(self, r: int) -> bool:
        """ For the current row r, assign N/S to all remaining arrow cells not already set by EW plan
            Simple greedy approach respecting:
            - border feasibility
            - column running counts (s_above, n_cap)
        """
        for c in range(self.C):
            p = (r, c)
            if not (p in self.puz.arrow_cells and p not in self._chosen):
                continue
            allowed = get_allowed_directions(r, c, self.R, self.C)
            # prefer S early (push contribution to numbers below), then N; fallback to EW if vertical blocked
            prefs = (Direction.S, Direction.N) if self.prefer_south else (Direction.N, Direction.S)
            placed = False
            for d in prefs:
                if d.value in allowed and self._can_place_vertical(p, d):
                    self._place_vertical(p, d)
                    placed = True
                    break
            if not placed: # fallback to any allowed E-W (should be rare)
                for d in (Direction.E, Direction.W):
                    if d.value in allowed:
                        self._chosen[p] = d
                        placed = True
                        break
            if not placed:
                return False
        return True

    def _can_place_vertical(self, p: Pos, d: Direction) -> bool:
        if d in [Direction.S, Direction.N]:
            # always OK to increase s_above; updated in _place_vertical
            return self._n_cap[p[1]] > 0 if d == Direction.N else True
        return False

    def _place_vertical(self, p: Pos, d: Direction) -> None:
        if d == Direction.S:
            self._s_above[p[1]] += 1
        elif d == Direction.N:
            self._n_cap[p[1]] -= 1
        self._chosen[p] = d

    # function for column residual updates on row exit
    def _update_column_residuals_on_row_exit(self, r: int) -> None:
        """ After finishing row r, pass any numbers located in row r
            For each t = (r,c), compute remaining 'N below' needed and tighten _n_cap[c].
            - basically a conservative version of the canvas equality: E_left + W_right + s_above[c] + N_below == v_t
        """
        for c in range(self.C):
            t = (r, c)
            if t not in self.puz.numbers:
                continue
            v_t = self.puz.numbers[t]
            # compute horizontal E_left/W_right already committed within row r
            e_left = sum(1 for cc in range(0, c) if self._chosen.get((r, cc), None) == Direction.E)
            w_right = sum(1 for cc in range(c + 1, self.C) if self._chosen.get((r, cc), None) == Direction.W)
            used = e_left + w_right + self._s_above[c]
            need_n_below = max(0, v_t - used)
            # tighten cap for N below in this column
            if need_n_below < self._n_cap[c]:
                self._n_cap[c] = need_n_below


    # static layout helper functions (small, fast, testable)

    @staticmethod
    def _row_segments_with_numbers(puz: Puzzle, r: int) -> List[RowSegment]:
        """ return RowSegment entries from left edge to right edge for row r (numbers or None at segment boundaries) """
        C = puz.cols
        nums = [c for (rr, c) in puz.numbers.keys() if rr == r]
        nums.sort()
        # boundaries are (None, num1), (num1, num2), ..., (numk, None)
        idxs = [None] + nums + [None]
        arrow_positions = [c for c in range(C) if (r, c) in puz.arrow_cells]
        segments: List[RowSegment] = []
        for i in range(len(idxs) - 1):
            L, R = idxs[i], idxs[i + 1]
            run_cols = [c for c in arrow_positions if (L is None or c > L) and (R is None or c < R)]
            segments.append(
                RowSegment(
                    row=r,
                    left_num=(r, L) if L is not None else None,
                    run=tuple((r, c) for c in run_cols),
                    right_num=(r, R) if R is not None else None,
                )
            )
        return segments

    @staticmethod
    def _col_active_numbers(puz: Puzzle) -> Dict[int, List[int]]:
        """ for each column, return a min-heap of row indices that contain numbers """
        cols: Dict[int, List[int]] = defaultdict(list)
        for (r, c), _ in puz.numbers.items():
            heapq.heappush(cols[c], r)
        return cols