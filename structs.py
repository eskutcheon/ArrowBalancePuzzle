

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Union


SUPPORTED_DIRECTIONS = ("N", "E", "S", "W")

class Direction(Enum):
    """ Cardinal directions that an arrow can point in - may be extended to include diagonals later """
    N: str = "N" # north
    E: str = "E" # east
    S: str = "S" # south
    W: str = "W" # west

    @staticmethod
    def all() -> Tuple["Direction", ...]:
        return (Direction.N, Direction.E, Direction.S, Direction.W)


# TODO: a dataclass feels like overkill for just this much so might replace this with a namedtuple later or just keep Tuple[int,int]
@dataclass(frozen=True)
class Pos:
    """ grid position (row, col) """
    r: int
    c: int


class Difficulty(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass
class Puzzle:
    """ Immutable puzzle data (numbers and fixed arrows). Doesn't contain solver state; just base puzzle instance """
    rows: int
    cols: int
    numbers: Dict[Pos, int]            # numbered cells and their required inbound counts
    fixed_arrows: Dict[Pos, Direction] # pre-placed arrow directions
    arrow_cells: Set[Pos]              # cells that must contain an arrow

    @staticmethod
    def from_grid(grid: List[List[str]]) -> "Puzzle":
        """ Build a puzzle from an MxN list of tokens (digits '0'..'9', '.', 'N','E','S','W').
            Numbers are not limited to single digits but by the puzzle size with largest digit ((rows + cols) // 2) - 1
        """
        R, C = len(grid), len(grid[0])
        numbers: Dict[Pos, int] = {}
        fixed_arrows: Dict[Pos, Direction] = {}
        arrow_cells: Set[Pos] = set()
        for r in range(R):
            for c in range(C):
                tok = grid[r][c]
                p = Pos(r, c)
                if tok == ".":
                    arrow_cells.add(p)
                # TODO: update to check from directions returned by get_allowed_directions
                elif tok in SUPPORTED_DIRECTIONS:
                    fixed_arrows[p] = Direction(tok)
                    arrow_cells.add(p)
                elif tok.isdigit():
                    v = int(tok)
                    numbers[p] = v
                else:
                    raise ValueError(f"Unknown token {tok!r} at {(r,c)}")
        return Puzzle(rows=R, cols=C,
                      numbers=numbers,
                      fixed_arrows=fixed_arrows,
                      arrow_cells=arrow_cells)

    def to_grid(self) -> List[List[str]]:
        """ Convert the puzzle to a grid representation (MxN list of strings). """
        grid = [["." for _ in range(self.cols)] for _ in range(self.rows)]
        for pos, num in self.numbers.items():
            grid[pos.r][pos.c] = str(num)
        for pos, dir_ in self.fixed_arrows.items():
            grid[pos.r][pos.c] = dir_.value
        return grid


@dataclass
class PuzzleMetadata:
    """ Metadata for a puzzle instance, used for testing and generation purposes. """
    puzzle: Union[Puzzle, List[List[str]]]  # can be a Puzzle instance or a grid of strings
    shape: Optional[Tuple[int, int]] = None
    clue_rate: Optional[float] = None
    seed: int = 123
    num_solutions: Optional[int] = None  # if known, for filtering unique instances
    greedy_solution: Optional[List[List[str]]] = None  # solution found during generation for reference
    file_source: Optional[str] = None  # path to the file this puzzle was loaded from, if applicable

    def __post_init__(self):
        """ Validate the provided puzzle and shape.
            If a Puzzle instance is provided, shape is inferred from it.
            If a grid is provided, shape must be specified or inferred from the grid dimensions.
        """
        if not isinstance(self.puzzle, Puzzle):
            self.puzzle = Puzzle.from_grid(self.puzzle)
        if not self.shape:
            self.shape = (self.puzzle.rows, self.puzzle.cols)
        else:
            assert self.shape == (self.puzzle.rows, self.puzzle.cols), \
                "Provided 'shape' does not match the Puzzle instance dimensions."
        if self.clue_rate is not None:
            assert 0 <= self.clue_rate <= 1, "Clue rate must be between 0 and 1."
        else: # approximate clue rate based on the number of fixed arrows
            self.clue_rate = round(len(self.puzzle.fixed_arrows) / len(self.puzzle.arrow_cells), 2)

    def __dict__(self):
        """ Convert to a dictionary for easy serialization. """
        return {
            "puzzle": self.puzzle.to_grid() if isinstance(self.puzzle, Puzzle) else self.puzzle,
            "greedy_solution": self.greedy_solution,
            "seed": self.seed,
            "num_solutions": self.num_solutions,
            "clue_rate": self.clue_rate,
        }