
import random
from collections import defaultdict
from itertools import product, filterfalse
from typing import List, Dict, Set, Tuple, Optional
from src.structs import Difficulty
from src.utils import get_allowed_directions, count_visible_arrows


SUPPORTED_DIRECTIONS = ("N", "E", "S", "W")
_DEF_BASES = {
    Difficulty.EASY: 0.26,
    Difficulty.MEDIUM: 0.18,
    Difficulty.HARD: 0.11,
    }
_DEF_CLAMPS = {
    Difficulty.EASY: (0.15, 0.35),
    Difficulty.MEDIUM: (0.10, 0.28),
    Difficulty.HARD: (0.06, 0.20),
}


# #? NOTE: should be reusable for checking if (r,c) is a number cell instead of using str.isdigit()
#     #? which we may want to move away from in the future (especially if we move to numpy later)
# def _has_same_parity(r: int, c: int) -> bool:
#     """ Check if row and column indices have the same parity (both even or both odd) """
#     return (r % 2 == 0) == (c % 2 == 0)


def _default_number_layout(rows: int, cols: int) -> Set[Tuple[int, int]]:
    """ Place numbers on the checkerboard pattern at (even, even) and (odd, odd) cells so that every number is at
        Manhattan distance >= 2 from any other number.
    """
    # if the row and column indices have the same parity (both even or both odd), add a position to the final set
    return set(((r, c) for r in range(rows) for c in range(cols) if (r % 2 == 0) == (c % 2 == 0)))


def _layout_to_grid(rows: int, cols: int, numbers: Set[Tuple[int, int]]) -> List[List[str]]:
    """ Build an empty puzzle grid with '.' where arrows will go and '0' placeholders in number cells
        0s get replaced with actual counts later
    """
    g = [["." for _ in range(cols)] for _ in range(rows)]
    for (r, c) in numbers:
        g[r][c] = "0"
    return g


def _assign_random_arrows(grid: List[List[str]], rng: random.Random) -> None:
    """ Fill every '.' with a random direction token in-place. """
    R, C = len(grid), len(grid[0])
    # for r, c in product(range(R), range(C)):
    #     if grid[r][c] == ".":
    for r, c in filterfalse(lambda idx: grid[idx[0]][idx[1]] != ".", product(range(R), range(C))):
        # iterating over (r,c) where grid[r][c] == "."
        allowed = get_allowed_directions(r, c, R, C)
        #? NOTE: this should never happen given the grid size constraints, but it helps to fail fast if it does
        assert allowed, f"No allowed directions at {(r,c)}; check grid size."
        grid[r][c] = rng.choice(allowed)


def _mask_arrows(
    solution: List[List[str]],
    clue_rate: float,
    rng: random.Random
) -> List[List[str]]:
    """ Return a grid by hiding each arrow with probability (1-clue_rate) while numbers are kept. clue_rate in [0,1]. """
    R, C = len(solution), len(solution[0])
    puzzle = [[solution[r][c] for c in range(C)] for r in range(R)]
    for r, c in product(range(R), range(C)):
        # if current token is in arrow directions and random check fails, mask it
        if solution[r][c] in SUPPORTED_DIRECTIONS and rng.random() > clue_rate:
            puzzle[r][c] = "."
    return puzzle


def _inbound_dir_to(rt: int, ct: int, r: int, c: int) -> Optional[str]:
    """ Precompute inbound direction per relative position for speed, i.e. for a target at (rt,ct):
        - same row, j < ct => inbound dir 'E';      j > ct => inbound dir 'W'
        - same col, i < rt => inbound dir 'S';      i > rt => inbound dir 'N'
    """
    if r == rt:
        return 'E' if c < ct else ('W' if c > ct else None)
    if c == ct:
        return 'S' if r < rt else ('N' if r > rt else None)
    return None

def _count_contributors(
        numbers: Set[Tuple[int, int]],
        bounds: Tuple[int, int]
    ) -> Tuple[
            Dict[Tuple[int, int], List[Tuple[Tuple[int, int], str]]],
            Dict[Tuple[int, int], List[Tuple[Tuple[int, int], str]]]
        ]:
    """ precompute which arrows can contribute to which numbers and vice versa for faster repair """
    contributors_by_number = defaultdict(list)
    arrows_to_numbers = defaultdict(list)
    R, C = bounds
    for rt, ct in numbers:
        # collect *contributing* arrow coordinates to t
        for c in range(C):
            if c == ct or (rt, c) in numbers:
                continue
            inbound = _inbound_dir_to(rt, ct, rt, c)
            if inbound:
                pos = (rt, c)
                contributors_by_number[(rt, ct)].append((pos, inbound))
                # collect *contributing* arrow coordinates to t
                arrows_to_numbers[pos].append(((rt, ct), inbound))
        for r in range(R):
            if r == rt or (r, ct) in numbers:
                continue
            inbound = _inbound_dir_to(rt, ct, r, ct)
            if inbound:
                pos = (r, ct)
                contributors_by_number[(rt, ct)].append((pos, inbound))
                arrows_to_numbers[pos].append(((rt, ct), inbound))
    return contributors_by_number, arrows_to_numbers


def _choose_flip_direction(rng: random.Random, current: str, allowed: Set[str]) -> Optional[str]:
    """ Choose a new direction different from current from the allowed set. """
    # use explicit "allowed" set to avoid invalid directions based on the grid edges
    choices = [d for d in allowed if d != current]
    if not choices:
        return None
    return rng.choice(choices)


def _repair_overflows(
    sol_grid: List[List[str]],
    numbers: Set[Tuple[int, int]],
    max_digit: int,
    rng: random.Random,
    max_flips: int = 10000 # might want to dynamically increase this for larger grids
) -> bool:
    """ Iteratively flip arrows to eliminate counts > max_digit.
        Returns True on success; False if we exceeded max_flips (caller can resample).
    """
    R, C = len(sol_grid), len(sol_grid[0])
    # counts = _compute_all_counts(sol_grid, numbers)
    counts = {p: count_visible_arrows(sol_grid, p) for p in numbers}
    contrib_by_number, arrows_to_numbers = _count_contributors(numbers, (R, C))
    flips = 0
    # Build a quick lookup of number positions for membership tests
    while True: # main repairing loop
        # find any overflow
        over = [p for p, v in counts.items() if v > max_digit]
        if not over: # if no more overflows, exit successfully
            # write numbers back into the grid and return True
            for p in numbers:
                sol_grid[p[0]][p[1]] = str(counts[p])
            return True
        # pick the worst overflow to reduce fastest
        over.sort(key=lambda p: counts[p], reverse=True)
        t = over[0]
        rt, ct = t # unpacking Pos while keeping t for indexing counts
        need_reduce = counts[t] - max_digit
        contributors = [pos for (pos, d) in contrib_by_number[t] if sol_grid[pos[0]][pos[1]] == d]
        if not contributors:  # Shouldn’t happen, but defensive
            return False
        # Flip up to `need_reduce` contributors this round
        # for (ra, ca) in rng.sample(contributors, k=min(need_reduce, max(1, len(contributors)))):
        #& TESTING: Heuristically prefer flipping contributors with many victims first (greedy degree) instead of randomly
        contributors.sort(key=lambda rc: (abs(rc[0] - rt) + abs(rc[1] - ct)), reverse=True)
        k = min(need_reduce, max(1, len(contributors)))
        for (ra, ca) in contributors[:k]:
            curr = sol_grid[ra][ca]
            forbid = _inbound_dir_to(rt, ct, ra, ca)  # direction that would keep contributing to t
            allowed = set(get_allowed_directions(ra, ca, R, C))
            if forbid:
                allowed.discard(forbid)
            new_dir = _choose_flip_direction(rng, curr, allowed)
            if new_dir is None or new_dir == curr:
                continue
            sol_grid[ra][ca] = new_dir
            flips += 1
            if flips > max_flips:
                return False
            # update counts for all affected numbers
            for (target_pos, inbound_dir) in arrows_to_numbers[(ra, ca)]:
                if curr == inbound_dir:
                    counts[target_pos] -= 1
                if new_dir == inbound_dir:
                    counts[target_pos] += 1
            # terminate early if we already fixed this target’s overflow
            if counts[t] <= max_digit:
                break


#& new difficulty-related functions
def _scaled_clue_rate(rows: int, cols: int, diff: Difficulty) -> float:
    import math
    base = _DEF_BASES[diff]
    scale = math.sqrt(99.0 / float(rows * cols))
    lo, hi = _DEF_CLAMPS[diff]
    return max(lo, min(hi, base * scale))


# OPTIONAL: tiny nudge for easy puzzles – ensure at least one arrow stays fixed on each border line if any exist
# no change in layout - only prevent masking some border arrows when difficulty is EASY
def _apply_easy_border_nudge(puzzle: List[List[str]], solution: List[List[str]]) -> None:
    """ Ensure at least one arrow remains fixed on each border line of the puzzle using for-else logic (like switch-case-finally) """
    R, C = len(puzzle), len(puzzle[0])
    def _add_solution_to_edge(idx: int, is_row: bool) -> None:
        bound = C if is_row else R
        # TODO: each use of SUPPORTED_DIRECTIONS should probably be replaced with get_allowed_directions to be safer
        for i in range(bound):
            tok = puzzle[idx][i] if is_row else puzzle[i][idx]
            if tok in SUPPORTED_DIRECTIONS:
                break
        else: # no break => no fixed arrow on this edge so copy one from solution if possible
            for i in range(bound):
                tok = solution[idx][i] if is_row else solution[i][idx]
                if tok in SUPPORTED_DIRECTIONS:
                    a, b, = (idx, i) if is_row else (i, idx)
                    puzzle[a][b] = tok
                    break
    # top and bottom edges
    _add_solution_to_edge(0, is_row=True)
    _add_solution_to_edge(R-1, is_row=True)
    # left and right columns
    _add_solution_to_edge(0, is_row=False)
    _add_solution_to_edge(C-1, is_row=False)



def generate_initial_puzzle(
    rows: int,
    cols: int,
    rng: random.Random, # random number generator for reproducibility
    clue_rate: Optional[float] = None,
    difficulty: Optional[Difficulty] = None,
    max_resamples: int = 20,
) -> Tuple[List[List[str]], List[List[str]]]:
    """ Generate a fresh puzzle (with at least one solution)
        - If `clue_rate` is None and `difficulty` is set, derive `clue_rate` from difficulty.
        Returns (puzzle_grid, solution_grid).
    """
    max_digit = ((rows + cols) // 2) - 1 # assumes rows and cols are always odd
    numbers = _default_number_layout(rows, cols)
    # derive clue_rate if not provided
    if clue_rate is None:
        diff = difficulty or Difficulty.MEDIUM
        clue_rate = _scaled_clue_rate(rows, cols, diff)
    for _ in range(max_resamples):
        sol = _layout_to_grid(rows, cols, numbers)
        _assign_random_arrows(sol, rng)
        # try to repair any overflows; if repair fails, resample
        if not _repair_overflows(sol, numbers, max_digit=max_digit, rng=rng):
            continue
        # digits are now written into sol by _repair_overflows
        puzzle = _mask_arrows(sol, clue_rate=clue_rate, rng=rng)
        if difficulty == Difficulty.EASY:
            _apply_easy_border_nudge(puzzle, sol)
        return puzzle, sol
    raise RuntimeError("Generator: failed to repair within resample budget; try another seed/grid size.")
