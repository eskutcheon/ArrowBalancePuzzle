"""
SVG renderer for Arrow/Number puzzles.
Input: 2D list of strings (digits, "N/E/S/W", ".")
Output: SVG string or file with per-cell borders, optional checkerboard, and arrows rendered as Unicode.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union, Iterable
import html

ArrowGrid = List[List[str]]

ARROW_GLYPHS = {"N": "↑", "E": "→", "S": "↓", "W": "←"}

@dataclass
class RenderTheme:
    # Sizes
    cell_px: int = 44
    stroke_px: float = 1.5
    outer_stroke_px: float = 2.5
    # Colors
    fg: str = "#111111"
    line: str = "#000000"
    shade: str = "#f4f6f8"  # checkerboard for one parity
    bg: str = "#ffffff"
    # Typography
    font_family: str = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"
    font_weight_num: int = 700
    font_weight_arrow: int = 600
    font_size_num_pct: int = 48   # % of cell_px
    font_size_arrow_pct: int = 54 # % of cell_px
    # Options
    use_checkerboard: bool = True
    show_coords: bool = False  # optional row/col headers
    padding_px: int = 12       # outer padding around the grid

@dataclass
class RenderOpts:
    theme: RenderTheme = field(default_factory = RenderTheme)
    # If provided, render two grids side-by-side (puzzle | solution)
    solution: Optional[ArrowGrid] = None
    title: Optional[str] = None
    # blank '.' should remain blank in puzzle output (per requirements)
    # arrows render as ↑→↓←

def _is_digit(tok: str) -> bool:
    return len(tok) >= 1 and tok.isdigit()

def _token_to_text(tok: str) -> str:
    if tok in ARROW_GLYPHS:
        return ARROW_GLYPHS[tok]
    if tok == '.':
        return ''  # blank
    return tok  # digits or other printable

def _grid_size(grid: ArrowGrid) -> Tuple[int, int]:
    R = len(grid)
    C = len(grid[0]) if R else 0
    return R, C

def _escape(t: str) -> str:
    return html.escape(t, quote=True)

def _cell_classes(r: int, c: int) -> str:
    # For potential CSS-based styling later; not used heavily now.
    return f"cell r{r+1} c{c+1}"

def _text_y_center(font_px: float) -> float:
    # SVG text baseline adjustment heuristic so glyphs look optically centered.
    # Monospace digits/arrows sit a bit below mathematical center; nudge up slightly.
    return 0.34 * font_px  # relative baseline shift


# TODO: replace with decomposed methods and separate rendering to single grid (one of puzzle and solution, not both)
def render_svg(
    grid: ArrowGrid,
    opts: Optional[RenderOpts] = None,
) -> str:
    """
    Return an SVG string for the puzzle grid.
    - Every cell has a border on all four sides (drawn as individual rect strokes).
    - '.' cells render blank (no glyph).
    - 'N/E/S/W' render as Unicode arrows.
    - digits render as bold numbers.
    If opts.solution is provided, renders a side-by-side figure: puzzle | solution.
    """
    opts = opts or RenderOpts()
    th = opts.theme
    R, C = _grid_size(grid)
    assert R > 0 and C > 0, "Grid must be non-empty"
    # Layout
    cell = th.cell_px
    pad = th.padding_px
    gap = cell // 2  # gap between puzzle and solution when both shown
    # Determine total width/height
    panels = 2 if opts.solution is not None else 1
    width = pad*2 + panels*C*cell + (panels-1)*gap
    height = pad*2 + R*cell
    # Font sizes
    num_px = int(th.font_size_num_pct * cell / 100.0)
    arr_px = int(th.font_size_arrow_pct * cell / 100.0)

    def draw_one_grid(x0: int, y0: int, grid_local: ArrowGrid) -> str:
        parts: list[str] = []
        # Outer background
        parts.append(f'<rect x="{x0}" y="{y0}" width="{C*cell}" height="{R*cell}" fill="{th.bg}" stroke="{th.line}" stroke-width="{th.outer_stroke_px}"/>\n')
        # Cells
        for r in range(R):
            for c in range(C):
                x = x0 + c*cell
                y = y0 + r*cell
                # Checkerboard fill
                if th.use_checkerboard and ((r + c) % 2 == 1):
                    parts.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{th.shade}" stroke="{th.line}" stroke-width="{th.stroke_px}"/>\n')
                else:
                    parts.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="none" stroke="{th.line}" stroke-width="{th.stroke_px}"/>\n')
                tok = grid_local[r][c]
                if tok == '.':
                    continue  # blank
                text = _escape(_token_to_text(tok))
                if not text:
                    continue
                # Numbers vs arrows get slightly different sizing/weight
                is_num = _is_digit(tok)
                fp = num_px if is_num else arr_px
                fw = th.font_weight_num if is_num else th.font_weight_arrow
                # Center text
                cx = x + cell/2
                cy = y + cell/2 + _text_y_center(fp)
                parts.append(
                    f'<text x="{cx}" y="{cy}" text-anchor="middle" font-family="{_escape(th.font_family)}" '
                    f'font-weight="{fw}" font-size="{fp}px" fill="{th.fg}">{text}</text>\n'
                )
        return ''.join(parts)

    # Title
    title_str = ""
    if opts.title:
        title_str = f"<title>{_escape(opts.title)}</title>\n"

    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           title_str]
    # Draw puzzle (and solution if present)
    x_cursor = th.padding_px
    svg.append(draw_one_grid(x_cursor, th.padding_px, grid))
    if opts.solution is not None:
        x_cursor += C*cell + gap
        svg.append(draw_one_grid(x_cursor, th.padding_px, opts.solution))
    svg.append("</svg>")
    return ''.join(svg)


def save_svg(svg_str: str, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(svg_str)


def render_and_save(
    puzzle: ArrowGrid,
    out_svg_path: str,
    solution: Optional[ArrowGrid] = None,
    theme: Optional[RenderTheme] = None,
    title: Optional[str] = None,
) -> None:
    svg = render_svg(puzzle, RenderOpts(theme=theme or RenderTheme(), solution=solution, title=title))
    save_svg(svg, out_svg_path)

if __name__ == "__main__":
    # Tiny demo if run directly
    sample = [
        ["3","W","1",".","0"],
        [".","1",".","3","W"],
        ["4",".","0",".","0"],
        [".","1",".","3","W"],
        ["2","S","0",".","0"],
        [".","2",".","2","."],
        ["2",".","0",".","2"],
    ]
    sol = [
        ["3","W","1","S","0"],
        ["S","1","N","3","W"],
        ["4","W","0","N","0"],
        ["N","1","E","3","W"],
        ["2","S","0","S","0"],
        ["N","2","W","2","S"],
        ["2","W","0","E","2"],
    ]
    svg = render_svg(sample, RenderOpts(solution=sol, title="Sample 7x5"))
    os.makedirs("examples", exist_ok=True)
    save_svg(svg, "examples/sample_7x5.svg")