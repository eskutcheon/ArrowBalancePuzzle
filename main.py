
import json
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Literal, Union
# importing classes used to define the puzzle structure
from solver import count_solutions, solve_grid
from utils import validate_filled_grid, validate_solution_against_puzzle, pretty_print
from generator import generate_initial_puzzle
from structs import PuzzleMetadata, Difficulty



def get_clue_rate(row, cols, clue_rate: Optional[float] = None, difficulty: Optional[Literal['easy', 'medium', 'hard']] = None) -> float:
    """ Determine the clue rate to use based on difficulty or a provided clue rate. If both given, the explicit clue_rate takes precedence. """
    if clue_rate is not None:
        assert 0 < clue_rate < 1, "clue_rate must be between 0 and 1"
        return clue_rate
    if difficulty is not None:
        assert difficulty in ('easy', 'medium', 'hard'), "Invalid difficulty level; must be one of 'easy', 'medium', or 'hard'"
        clue_rate = {'easy': 0.25, 'medium': 0.2, 'hard': 0.15}[difficulty]
        if row * cols <= 81:  # smaller grids should be more solvable so reduce clue rate slightly
            clue_rate = round(clue_rate - 0.05, 2)
        return clue_rate
    # default if neither is provided (default clue_rate for medium difficulty)
    return 0.2

# TODO: kind of want to remove the use of Pos in favor of Tuple[int, int] to improve transparency from here on out

# TODO: maybe add to generator.py
def generate_puzzle(
    rows: int,
    cols: int,
    seed: Optional[int] = None,
    clue_rate: Optional[float] = None,  # now optional
    max_tries: int = 500,
    difficulty: Optional[Literal['easy', 'medium', 'hard']] = None,
    unique_only: bool = False,
) -> Tuple[List[List[str]], List[List[str]]]:
    """ Generate a fresh puzzle that is guaranteed to have at least one solution
        Strategy:
            1. Fix a number layout (checkerboard, even/even indices) per your assumptions.
            2. Randomly assign directions to all arrow cells (this is a *solution*).
            3. Compute all numbers from the arrows. Reassign arrows when necessary to avoid contradictions.
            4. Hide a fraction of arrows (1 - clue_rate) to form the puzzle.
        Returns
            (puzzle_grid, solution_grid).
        Notes:
            - This guarantees solvability (the hidden arrows are consistent with the numbers), but does NOT
                enforce uniqueness. Use `count_solutions` if you want to filter to unique instances.
            - To keep single-digit numbers, we reject samples whose max count > 9.
    """
    assert rows % 2 == 1 and cols % 2 == 1, "rows/cols must be odd per the spec"
    if difficulty is not None:
        assert difficulty in ('easy', 'medium', 'hard'), "Invalid difficulty level; must be one of 'easy', 'medium', or 'hard'"
        clue_rate = get_clue_rate(rows, cols, clue_rate, difficulty)
        difficulty = getattr(Difficulty, difficulty.upper())
    rng = random.Random(seed)
    for _ in range(max_tries):
        try:
            puzzle, sol = generate_initial_puzzle(
                rows=rows,
                cols=cols,
                rng=rng,
                clue_rate=clue_rate,
                difficulty=difficulty,
            )
        # except ValueError:
        except Exception as e:
            print("Generation error:", e)
            print("Retrying...")
            continue
        #** updating to skip solving - might add back if I want to filter to unique solutions only later
        # quick sanity: solver should find at least one solution
        # solved = solve_grid(puzzle)
        # if solved is None:
        #     continue  # (should be rare) numerical symmetry + masking caused a contradiction - try again
        # final validation (paranoid but cheap)
        # ok1, msg1 = validate_solution_against_puzzle(puzzle, solved)
        # if not ok1:
        #     continue
        ok2, _ = validate_filled_grid(sol)
        if not ok2:
            continue
        if unique_only:
            # using a limit of 2 to avoid long computation times - quits after at most 2 are found
            num_solutions = count_solutions(puzzle, limit=2)
            if num_solutions != 1:
                continue
        return puzzle, sol
    raise RuntimeError("Failed to generate a valid puzzle within max_tries; try a different seed or smaller size.")


def save_generated_test_puzzles(to_generate: List[Dict[str, Any]], verbose: bool = False):
    """ Save generated test puzzles to JSON files in the 'tests/puzzles' directory. """
    # from tqdm import tqdm # removing this for now since it's literally the only nonstandard library used in the project, requiring a download
    for idx, recipe in enumerate(to_generate): #, desc="Generating puzzles", total=len(to_generate)):
        clue_rate = recipe.get('clue_rate', None)
        diff_rating = recipe.get('difficulty', None)
        if verbose:
            print(f"Generating puzzle with shape={recipe['shape']}, seed={recipe['seed']}, clue_rate={clue_rate}, difficulty={diff_rating}...")
        puzzle, solution = generate_puzzle(
            *recipe['shape'], seed=recipe['seed'], clue_rate=clue_rate,
            difficulty=diff_rating, unique_only=recipe.get('unique_only', False)
        )
        if verbose:
            print("Generated: ")
            pretty_print(puzzle, render_arrows=True)
            print("Solution: ")
            pretty_print(solution, render_arrows=True)
            print("Now counting solutions...")
        num_solutions = count_solutions(puzzle, limit=10)
        # TODO: might want the following attributes later: "difficulty", "clue_rate", generation method, and all known solutions (or their hashes)
        to_write = {
            "puzzle": puzzle,
            "greedy_solution": solution,
            "seed": recipe['seed'],
            "shape": recipe['shape'],
            "difficulty": recipe.get('difficulty', 'medium'),
            "num_solutions": num_solutions,
            "clue_rate": get_clue_rate(*recipe['shape'], clue_rate, diff_rating),
            "date_created": datetime.now().isoformat(timespec='seconds'),
        }
        shape_str = f"{recipe['shape'][0]}x{recipe['shape'][1]}"
        difficulty = recipe.get('difficulty', 'medium')
        num_with_shape = len([p for p in Path(r"tests/puzzles").glob(f"puzzle*_*_{difficulty}.json") if shape_str in p.name])
        file_path = Path(r"tests/puzzles", f"puzzle{num_with_shape + 1}_{shape_str}_{difficulty}.json")
        if verbose:
            print(f"Number of solutions found (stops at 10): {num_solutions}")
            print(f"Saving to {file_path}...")
        # print(f"writing file path '{file_path}'...")
        with open(file_path, "w") as fptr:
            json.dump(to_write, fptr, indent=2)

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


if __name__ == "__main__":
    # validate the solver's output on existing test grids
    # test_loading()
    test_shapes = [
        (5,5), (5,7), (5,9),
        (7,5), (7,7), (7,9),
        (9,7), (9,9), (9,11),
        (11,9), (11,11), # (11,13),
        #(13,11), (13,13), (13,15),
        #(15,13), (15,15), (15,17)
    ]
    test_seeds = [42] #, 123] #, 54321]
    test_difficulties = ['easy', 'medium', 'hard'] # using default clue rates according to difficulty
    # for diff in test_difficulties:
    for seed in test_seeds:
        for shape in test_shapes:
            diff = random.choice(test_difficulties)
            to_generate = [{"shape": shape, "seed": seed, "clue_rate": None, "difficulty": diff, "unique_only": False}]
            save_generated_test_puzzles(to_generate, verbose=True)

    # save_generated_test_puzzles([{"shape": (7,5), "seed": 54321, "clue_rate": None, "difficulty": "hard", "unique_only": False}], verbose=True)
    # puzzle_metadata = load_single_test_puzzle(r"tests/puzzles/puzzle4_7x5.json")
    # view_loaded_test_puzzle(puzzle_metadata, show_solved=True, show_count=False)