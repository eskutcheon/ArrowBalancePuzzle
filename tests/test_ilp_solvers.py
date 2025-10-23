
# temporary solution for imports (provided I remember to remove it before packaging the project)
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
#####################################################################################################
import random
from typing import Tuple, List
from src.structs import Puzzle
from src.utils import default_number_layout, layout_to_grid, validate_filled_grid, count_visible_arrows
from src.generator import generate_initial_puzzle
from src.solver.solver_ilp import ILPNumberedPuzzleSolver, ILPArrowsOnlyPuzzleSolver
from src.solver.solver_csp import ArrowCSP



def small_numbered_example(shape: Tuple[int,int]) -> List[List[str]]:
    rows, cols = shape
    # build an arrows-only solution via the generator, then use its numbers
    puzzle, sol = generate_initial_puzzle(rows, cols, random.Random(1234), difficulty="hard")
    return puzzle  # puzzle has numbers + hidden arrows ('.') / fixed arrows

def test_ilp_numbered(shape: Tuple[int,int] = (5,5)):
    grid = small_numbered_example(shape)
    puz = Puzzle.from_grid(grid)
    solver = ILPNumberedPuzzleSolver(rng_seed=1, puzzle=puz)
    solmap = solver.solve(puz)
    assert solmap is not None
    # write directions back onto grid and validate
    out = [row[:] for row in grid]
    for (r,c), d in solmap.items():
        out[r][c] = d.value
    ok, msg = validate_filled_grid(out)
    print("ILP numbered:", ok, msg)

def test_ilp_arrows_only(shape: Tuple[int,int] = (5,5)):
    rows, cols = shape
    nums = default_number_layout(rows, cols)
    solver = ILPArrowsOnlyPuzzleSolver(max_digit=((rows+cols)//2)-1, rng_seed=42)
    # solver = ILPArrowsOnlyPuzzleSolver(max_digit=cols+1, rng_seed=42)
    solmap = solver.solve((rows, cols), list(nums))
    assert solmap is not None
    # build the full grid and validate numbers are single-digit
    grid = layout_to_grid(rows, cols, nums)
    for (r,c), d in solmap.items():
        grid[r][c] = d.value
    # add numbers in from arrow solutions the validate the final grid
    for (r,c) in nums:
        grid[r][c] = str(count_visible_arrows(grid, (r,c)))
    ok, msg = validate_filled_grid(grid)
    print("ILP arrows-only:", ok, msg)

if __name__ == "__main__":
    test_ilp_numbered((15,11))
    test_ilp_arrows_only((7,9))