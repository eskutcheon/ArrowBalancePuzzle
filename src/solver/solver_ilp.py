# src/solver/solver_ilp.py
import random
from typing import Dict, List, Optional, Tuple, Iterable
from abc import ABC, abstractmethod
# importing classes used to define the puzzle structure
from ..structs import Direction, Puzzle
from ..utils import get_allowed_directions
from .solver_csp import ArrowCSP

Pos = Tuple[int, int]  # (row, col)
CellDirection = Tuple[Pos, Direction]  # a single boolean variable "pos has arrow in direction ___"



class ILPSolverBase(ABC):
    """ Abstract base class for ILP solvers for Arrow Puzzles. Prototypes abstract methods for solving numerical grids or arrows-only grids. """
    def __init__(self, rng_seed: Optional[int] = None, puzzle: Optional[Puzzle] = None):
        self.rng = random.Random(rng_seed)
        self.puzzle: Optional[Puzzle] = None
        self.dims = (puzzle.rows, puzzle.cols) if puzzle is not None else None

    @abstractmethod
    def fallback_solver(self) -> Optional[Dict[Pos, Direction]]:
        """ Fallback solver method to be implemented by subclasses. """
        raise NotImplementedError("Subclasses must implement fallback_solver method.")

    @abstractmethod
    def compare_lp_sum(self, lp_sum, pos: Pos) -> int:
        """ Comparison function for ILP constraints to be implemented by subclasses. """
        raise NotImplementedError("Subclasses must implement compare_lp_sum method.")

    def is_direction_allowed(self, p: Pos, d: Direction) -> bool:
        """ Check if a direction is allowed for a given position in the puzzle dimensions. """
        if self.dims is None:
            raise ValueError("Puzzle dimensions are not set.")
        return d.value in get_allowed_directions(p[0], p[1], *self.dims)

    def set_puzzle_instance(self, puz: Puzzle):
        """ Set the internal puzzle instance for the solver. """
        if self.puzzle is None:
            self.puzzle = puz
        elif self.puzzle != puz:
            print("[WARNING] ILPSolverBase instance puzzle does not match the provided puzzle; overwriting internal puzzle reference.")
            self.puzzle = puz
        self.dims = (puz.rows, puz.cols)

    def solve_ilp(
        self,
        # TODO: might want to remove this argument and exclusively use self.puzzle internally; will need more careful state management then
        puz: Puzzle,
        number_positions: Iterable[Pos], # will vary between List[Post] for arrows-only and Dict[Pos, int] for numbered puzzles
        lp_title: str = "ArrowPuzzle"
    ) -> Optional[Dict[Pos, Direction]]:
        """ General ILP solver method using the subclass-defined comparison and fallback methods. """
        self.set_puzzle_instance(puz)
        try:
            import pulp
        except (ImportError, ModuleNotFoundError):
            return self.fallback_solver()
        R, C = puz.rows, puz.cols
        # build variables
        all_dirs = list(Direction.all())
        prob = pulp.LpProblem(lp_title, pulp.LpStatusOptimal)
        X: Dict[CellDirection, pulp.LpVariable] = {}
        for p in puz.arrow_cells:
            for d in filter(lambda direction: self.is_direction_allowed(p, direction), all_dirs):
                X[(p,d)] = pulp.LpVariable(f"X_{p[0]}_{p[1]}_{d.value}", lowBound=0, upBound=1, cat="Binary")
        # exactness expectation - check that every arrow cell has exactly 1 direction
        for p in puz.arrow_cells:
            prob += pulp.lpSum(X[(p,d)] for d in all_dirs if (p,d) in X) == 1
        # single-digit caps per potential number cell
        for t in number_positions: # iterating over either a List[Pos] or Dict[Pos, int] should still yield Pos
            inbound = []
            r,c = t
            # TODO: abstract the directional scans into some common helper functions like the helpers in utils.py
            # left (E), right (W), up (S), down (N)
            for j in range(c-1, -1, -1):
                p = (r,j)
                if p in puz.arrow_cells and (p,Direction.E) in X:
                    inbound.append(X[(p,Direction.E)])
            for j in range(c+1, C):
                p = (r,j)
                if p in puz.arrow_cells and (p,Direction.W) in X:
                    inbound.append(X[(p,Direction.W)])
            for i in range(r-1, -1, -1):
                p = (i,c)
                if p in puz.arrow_cells and (p,Direction.S) in X:
                    inbound.append(X[(p,Direction.S)])
            for i in range(r+1, R):
                p = (i,c)
                if p in puz.arrow_cells and (p,Direction.N) in X:
                    inbound.append(X[(p,Direction.N)])
            prob += self.compare_lp_sum(pulp.lpSum(inbound), t)
            # prob += pulp.lpSum(inbound) == v
        # random objective to diversify
        prob += pulp.lpSum(self.rng.random() * X[k] for k in X.keys())
        status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
        if pulp.LpStatus[status] != "Optimal":
            return None
        sol: Dict[Pos, Direction] = {}
        for p in puz.arrow_cells:
            for d in all_dirs:
                if (p,d) in X and X[(p,d)].value() > 0.5:
                    sol[p] = d
                    break
        return sol if len(sol) == len(puz.arrow_cells) else None



class ILPNumberedPuzzleSolver(ILPSolverBase):
    """ ILP solver for numbered Arrow Puzzles """
    def fallback_solver(self) -> Optional[Dict[Pos, Direction]]:
        # fallback: Use the custom CSP solver
        print("[WARNING] ILP solver dependencies not found, falling back to CSP solver.")
        sol = ArrowCSP(self.puzzle).solve()
        return sol

    def compare_lp_sum(self, lp_sum, pos: Pos) -> 'pulp.pulp.LpConstraint':
        return lp_sum == self.puzzle.numbers[pos]

    def solve(self, puz: Puzzle, lp_title: str = "ArrowPuzzle") -> Optional[Dict[Pos, Direction]]:
        return self.solve_ilp(
            puz=puz,
            number_positions=puz.numbers.keys(),
            lp_title=lp_title
        )


class ILPArrowsOnlyPuzzleSolver(ILPSolverBase):
    """ ILP solver for arrows-only Arrow Puzzles """
    def __init__(self, max_digit: int, rng_seed: Optional[int] = None, puzzle: Optional[Puzzle] = None):
        super().__init__(rng_seed=rng_seed, puzzle=puzzle)
        self.max_digit = max_digit

    def fallback_solver(self) -> Optional[Dict[Pos, Direction]]:
        # fallback: simple random fill of allowed directions
        print("[WARNING] ILP solver dependencies not found, falling back to CSP solver.")
        sol: Dict[Pos, Direction] = {}
        for r in range(self.puzzle.rows):
            for c in range(self.puzzle.cols):
                p = (r,c)
                if p in self.puzzle.arrow_cells:
                    allowed_dirs = [d for d in Direction.all() if self.is_direction_allowed(p, d)]
                    if not allowed_dirs:
                        continue
                    sol[p] = self.rng.choice(allowed_dirs)
        return sol

    def compare_lp_sum(self, lp_sum, pos: Pos) -> 'pulp.pulp.LpConstraint':
        return lp_sum <= self.max_digit

    def solve(self, dims: Tuple[int, int], number_positions: List[Pos], lp_title: str = "ArrowsOnly") -> Optional[Dict[Pos, Direction]]:
        rows, cols = dims
        all_cells = set((r,c) for r in range(rows) for c in range(cols))
        arrow_cells = all_cells - set(number_positions)
        puz = Puzzle(rows=rows, cols=cols, numbers={}, fixed_arrows={}, arrow_cells=arrow_cells)
        return self.solve_ilp(
            puz = puz,
            number_positions = number_positions,
            lp_title = lp_title
        )

