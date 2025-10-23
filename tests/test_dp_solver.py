# tests/test_solvers_dp.py
# temporary solution for imports (provided I remember to remove it before packaging the project)
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
#####################################################################################################
import random
from typing import Tuple, List
from src.structs import Puzzle
from src.solver.solver_dp import DPPuzzleSolver
from src.generator import generate_initial_puzzle
from src.utils import validate_solution_against_puzzle, pretty_print



def test_dp_solver(shape: Tuple[int,int] = (7,7)) -> None:
    puzzle, reference = generate_initial_puzzle(*shape, rng=random.Random(2025))
    puz = Puzzle.from_grid(puzzle)
    solver = DPPuzzleSolver(puz)
    solved = solver.solve()
    if solved is None:
        print("DP/CSP could not solve the puzzle (UNSAT or bug).")
        return
    ok, msg = validate_solution_against_puzzle(puzzle, solved) if solved else (False, "UNSAT")
    print(f"DP({shape}):", ok, msg)
    if not ok:
        print("Puzzle:")
        # TODO: add a variant of pretty_print directly to the puzzle classes that accesses the arrow cell mapping directly
        pretty_print(puzzle, render_arrows=True)
        print("Reference:")
        pretty_print(reference, render_arrows=True)
        print("DP output:")
        pretty_print(solved, render_arrows=True)

if __name__ == "__main__":
    test_dp_solver((15,11))
    # test_dp_solver((7,7))
