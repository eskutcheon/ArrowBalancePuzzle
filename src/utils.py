# src/utils.py
from functools import lru_cache
from copy import deepcopy
from typing import Literal, List, Tuple, Optional, Callable, Set


#? NOTE: might eventually add support for diagonal directions ("NE", "SE", "SW", "NW") later
SUPPORTED_DIRECTIONS = ("N", "E", "S", "W")

def get_supported_digits(r:int, c: int) -> Tuple[str, ...]:
    max_digit = ((r + c) // 2) - 1 # assumes rows and cols are always odd
    return tuple(str(i) for i in range(max_digit + 1))


def get_clue_rate(row, cols, clue_rate: Optional[float] = None, difficulty: Optional[Literal['easy', 'medium', 'hard']] = None) -> float:
    """ Determine the clue rate to use based on difficulty or a provided clue rate. If both given, the explicit clue_rate takes precedence. """
    if clue_rate is not None:
        assert 0 <= clue_rate < 1, "clue_rate must be between 0 and 1"
        return clue_rate
    if difficulty is not None:
        assert difficulty in ('easy', 'medium', 'hard'), "Invalid difficulty level; must be one of 'easy', 'medium', or 'hard'"
        clue_rate = {'easy': 0.25, 'medium': 0.2, 'hard': 0.15}[difficulty]
        if row * cols <= 81:  # smaller grids (9x9) should be more solvable so reduce clue rate slightly
            clue_rate = round(clue_rate - 0.05, 2)
        return clue_rate
    # default if neither is provided (default clue_rate for medium difficulty)
    return 0.2

def _is_valid_token(tok: str, expected_tok: str) -> bool:
    """ Check if a token is a valid arrow direction or a digit. """
    return tok in SUPPORTED_DIRECTIONS and tok == expected_tok # or tok.isdigit()

def _loop_in_direction(
    grid: List[List[str]],
    # idx: int,
    step: int,
    center_idx: Tuple[int, int],
    is_row: bool,
    loop_condition: Callable[[int], bool],
    callback: Callable[[str], None]
) -> int:
    """ Generic helper to loop in one direction (row or column) until the edge of the grid.
        Calls callback(token) for each cell visited, and returns the number of steps taken.
    """
    assert step in [1, -1], "step increment must be either +1 or -1"
    r, c = center_idx
    idx = center_idx[int(is_row)] + step  # start one step away from the center
    while loop_condition(idx):
        # tok = grid[idx // len(grid[0])][idx % len(grid[0])]
        tok = grid[r][idx] if is_row else grid[idx][c]
        callback(tok) # keep validity check in the callback
        idx += step
    return idx


# TODO: generalize further and replace ArrowCSP._index_visibility with a similar helper function
def visible_arrows_from(grid: List[List[str]], t: Tuple[int, int]) -> List[str]:
    """ Return the list of arrow tokens ('N','E','S','W') that contribute to number at t
        by scanning in the four cardinal directions until the grid edge.
    """
    R, C = len(grid), len(grid[0])
    r, c = t #t.r, t.c
    seen: List[str] = []
    def append_if_valid(tok: str, target_tok: str) -> None:
        """ Append token to seen if it matches the target direction """
        nonlocal seen
        if _is_valid_token(tok, target_tok):
            seen.append(tok)
    # search to left: arrows at (r, j< c) pointing E
    _loop_in_direction(grid, -1, (r, c), True, lambda k: k >= 0, lambda tok: append_if_valid(tok, 'E'))
    # search to right: arrows at (r, j>c) pointing W
    _loop_in_direction(grid, 1, (r, c), True, lambda k: k < C, lambda tok: append_if_valid(tok, 'W'))
    # search above: arrows at (i<r, c) pointing S
    _loop_in_direction(grid, -1, (r, c), False, lambda k: k >= 0, lambda tok: append_if_valid(tok, 'S'))
    # search below: arrows at (i>r, c) pointing N
    _loop_in_direction(grid, 1, (r, c), False, lambda k: k < R, lambda tok: append_if_valid(tok, 'N'))
    # assert len(seen) > 0, f"No visible arrows found at {t} in grid of size {R}x{C}"
    return seen

def count_visible_arrows(grid: List[List[str]], t: Tuple[int, int]) -> int:
    """ Fast count of arrows that contribute to number at t (no list allocs) """
    R, C = len(grid), len(grid[0])
    r, c = t #t.r, t.c
    cnt = 0
    def increment_if_valid(tok: str, target_tok: str) -> None:
        nonlocal cnt
        if _is_valid_token(tok, target_tok):
            cnt += 1
    # left (E)
    _loop_in_direction(grid, -1, (r, c), True, lambda k: k >= 0, lambda tok: increment_if_valid(tok, 'E'))
    # right (W)
    _loop_in_direction(grid, 1, (r, c), True, lambda k: k < C, lambda tok: increment_if_valid(tok, 'W'))
    # up (S)
    _loop_in_direction(grid, -1, (r, c), False, lambda k: k >= 0, lambda tok: increment_if_valid(tok, 'S'))
    # down (N)
    _loop_in_direction(grid, 1, (r, c), False, lambda k: k < R, lambda tok: increment_if_valid(tok, 'N'))
    return cnt


def validate_filled_grid(grid: List[List[str]]) -> Tuple[bool, str]:
    """ Brute-force validator for a completed puzzle grid (numbers + all arrows chosen).
        Checks:
            - Every non-number cell is one of 'N','E','S','W'
            - Every number equals its total incoming visible arrows
        Returns (True, "OK") if valid, or (False, "reason") if invalid
    """
    R, C = len(grid), len(grid[0])
    digit_set = set(get_supported_digits(R, C))
    # validation: arrows must be valid tokens
    for r in range(R):
        for c in range(C):
            allowed_tokens = set(get_allowed_directions(r, c, R, C)).union(digit_set)
            tok = grid[r][c]
            if tok == ".":  # not allowed in a filled grid
                return False, f"Unfilled cell at {(r, c)}"
            if tok not in allowed_tokens:
                return False, f"Unknown token {tok!r} at {(r, c)}"
    # validation: each number must match visible arrow count
    for r in range(R):
        for c in range(C):
            tok = grid[r][c]
            if tok.isdigit():
                want = int(tok)
                seen = visible_arrows_from(grid, (r, c))
                got = len(seen)
                if got != want:
                    return False, f"Number {want} at {(r,c)} only sees {got} incoming arrows ({seen})"
    return True, "OK"


def validate_solution_against_puzzle(
    puzzle_grid: List[List[str]],
    filled_grid: List[List[str]]
) -> Tuple[bool, str]:
    """ Ensure a solution grid matches the puzzle structure:
        - preserves all numbers,
        - respects every pre-placed arrow in the puzzle,
        - fills every arrow cell with a direction token,
        - and satisfies all counts (uses validate_filled_grid).
    """
    R, C = len(puzzle_grid), len(puzzle_grid[0])
    # a) structure match
    if (R, C) != (len(filled_grid), len(filled_grid[0])):
        return False, "Shape mismatch between puzzle and solution"
    # b) numbers and fixed arrows preserved
    for r in range(R):
        for c in range(C):
            p = puzzle_grid[r][c]
            s = filled_grid[r][c]
            if p.isdigit():
                if s != p:
                    return False, f"Number changed at {(r,c)}"
            elif p in SUPPORTED_DIRECTIONS:
                if s != p:
                    return False, f"Fixed arrow changed at {(r,c)}"
    # c) now run full count validation
    return validate_filled_grid(filled_grid)


@lru_cache(maxsize=None)
def get_allowed_directions(r: int, c: int, r_max: int, c_max: int) -> Tuple[str, ...]: #List[str]:
    allowed = [] #set()
    if r > 0:
        allowed.append("N")
    if r < r_max - 1:
        allowed.append("S")
    if c > 0:
        allowed.append("W")
    if c < c_max - 1:
        allowed.append("E")
    return tuple(allowed)


# TODO: replace with pprint.pprint or some table formatting library later
def pretty_print(grid: List[List[str]], render_arrows: bool = True) -> None:
    pgrid = deepcopy(grid) if render_arrows else grid
    if render_arrows:
        # replace arrows with their unicode representations and "." placeholders with spaces
        arrow_map = {"N": "↑", "E": "→", "S": "↓", "W": "←", ".": " "}
        for r in range(len(pgrid)):
            for c in range(len(pgrid[0])):
                if pgrid[r][c] in arrow_map:
                    pgrid[r][c] = f"\033[36m{arrow_map[pgrid[r][c]]}\033[0m"  # using cyan unicode arrows
    for row in pgrid:
        print(" ".join(row))


def default_number_layout(rows: int, cols: int) -> Set[Tuple[int, int]]:
    """ return set of checkerboard positions (no two numbers at Manhattan distance 1) """
    # if the row and column indices have the same parity (both even or both odd), add a position to the final set
    return set((r, c) for r in range(rows) for c in range(cols) if (r % 2 == 0) == (c % 2 == 0))

def layout_to_grid(rows: int, cols: int, numbers: Set[Tuple[int, int]]) -> List[List[str]]:
    """ return grid with '.' at arrow cells and '0' placeholders at number cells, where digits are filled later """
    g = [["." for _ in range(cols)] for _ in range(rows)]
    for (r, c) in numbers:
        g[r][c] = "0"
    return g