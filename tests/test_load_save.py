
# temporary solution for imports (provided I remember to remove it before packaging the project)
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
from pathlib import Path
from typing import List, Optional, Tuple, Union
# importing classes used to define the puzzle structure
from src.solver.solver_csp import count_solutions, solve_grid
from src.utils import validate_solution_against_puzzle, pretty_print
from src.structs import PuzzleMetadata


# TODO: haven't actually tested this file, just moved stuff that used to be in main.py


def load_test_puzzles_by_shape(shape: Tuple[int, int], limit: Optional[int] = None) -> List[PuzzleMetadata]:
    """ Load all test puzzles of a given shape (rows x cols) from the 'tests' directory and return them as PuzzleMetadata instances. """
    loaded = []
    shape_str = f"{shape[0]}x{shape[1]}"
    # find all files matching the shape pattern
    for p in Path(r"tests/puzzles").glob(f"puzzle*_*.json"):
        if shape_str in p.name:
            loaded.append(load_single_test_puzzle(p))
    if limit is not None:
        loaded = loaded[:limit]
    return loaded

def load_single_test_puzzle(file_path: Union[str, Path]) -> PuzzleMetadata:
    """ Load a single test puzzle from a specified JSON file and return it as a PuzzleMetadata instance. """
    p = Path(file_path) if not isinstance(file_path, Path) else file_path
    if not p.exists():
        raise FileNotFoundError(f"File {file_path} does not exist.")
    with open(p, "r") as f:
        data = json.load(f)
        return PuzzleMetadata(file_source = p, **data)


def view_loaded_test_puzzle(puzzle_metadata: PuzzleMetadata, show_solved: bool = True, show_count: bool = False):
    """ Pretty-print a loaded puzzle and its solution from a PuzzleMetadata instance. """
    file_src = puzzle_metadata.file_source if puzzle_metadata.file_source else "N/A"
    print(f"Puzzle loaded from '{file_src}':")
    pretty_print(puzzle_metadata.puzzle.to_grid(), render_arrows=True) # might attach this to some config option later
    if show_solved:
        sol = puzzle_metadata.greedy_solution
        print("Solution:")
        if sol is None:
            sol = solve_grid(puzzle_metadata.puzzle.to_grid())
        pretty_print(sol, render_arrows=True)
    if show_count:
        count = puzzle_metadata.num_solutions
        if count is None:
            count = count_solutions(puzzle_metadata.puzzle.to_grid(), limit=10)
        print(f"Number of solutions (stops at 10): {count}")


# TODO: might want to change a lot of the data structures used in the whole project to replace lists of strings with numpy arrays
    #+ empty cells could be NaN and arrows could be represented with negative integers (using a mapping to directions)
    #+ this would make it easier to do vectorized operations and checks and should really speed certain things up

def test_loading():
    # mostly doing this to test loading from JSON and the validation logic:
    test_shape = (13, 11)
    test_puzzles = load_test_puzzles_by_shape(test_shape)
    for idx, puzzle_metadata in enumerate(test_puzzles):
        puzzle: List[List[str]] = puzzle_metadata.puzzle.to_grid()
        print(f"Validating puzzle with seed {puzzle_metadata.seed} and shape {puzzle_metadata.shape}...")
        pretty_print(puzzle, render_arrows=False)
        print("Solving test grid...")
        solution = solve_grid(puzzle)
        ok, msg = validate_solution_against_puzzle(puzzle, solution) if solution else (False, "UNSAT")
        print(f"Validation result: {ok}; Message: {msg}")
        pretty_print(solution)