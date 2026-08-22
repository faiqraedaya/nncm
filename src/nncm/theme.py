"""Design tokens — the single source of truth for how NNCM looks.

Deliberately Qt-free: the desktop shell (:mod:`nncm.gui.style`) and the
matplotlib figures (:mod:`nncm.sampling`, :mod:`nncm.training`) both read their
colours from here, so a chart drawn inside the window and the same chart saved
as a PNG by the CLI cannot drift apart.

Colours are named for the *job* they do, never for the hue they happen to be.
Nothing outside this module writes a hex value.

The categorical fill set is fixed, ordered and finite. It was validated on the
white chart surface under the all-pairs rule (the pairlist a scatter needs):
worst pair dE 10.1 simulated deutan, 21.2 normal vision, every hue over 3:1 on
the surface. :func:`series_colors` therefore raises past the end of the set
rather than cycling — a cycled palette silently reuses a hue and lies about
identity. Callers fold the tail into :data:`SERIES_OTHER`, validated against
the same three (worst dE 18.4 simulated, 23.5 normal).
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Colour
# ---------------------------------------------------------------------------
WINDOW = "#F2F2F5"        # application ground, behind the cards
SURFACE = "#FFFFFF"       # cards, fields, tables, drawing surfaces
SURFACE_ALT = "#F5F5F7"   # hover fill, secondary fill, totals-row tint
TEXT = "#1D1D1F"          # primary label and value
SUBTLE = "#6E6E73"        # secondary label, captions, explanations
TERTIARY = "#8E8E93"      # placeholder, disabled, units, faint annotation
BORDER = "#E1E1E4"        # separators, control outlines, chart gridlines

ACCENT = "#0071E3"        # interaction only
ACCENT_HOVER = "#0062C8"
ACCENT_TEXT = "#FFFFFF"   # text on an accent fill
ACCENT_SOFT = "#E8F1FD"   # selection fill in lists and tables

DANGER = "#D70015"        # genuine failures only
SUCCESS = "#0F7A4F"       # the good half of a verdict
WARNING = "#9A6700"       # an advisory that is not yet a failure

# Categorical chart fills, in slot order. Three is the whole set.
SERIES = ("#2A78D6", "#D95F28", "#18996C")
SERIES_OTHER = "#48484A"  # the folded tail, never an identity of its own

# ---------------------------------------------------------------------------
# Type
# ---------------------------------------------------------------------------
FONTS_DIR = Path(__file__).resolve().parent / "gui" / "fonts"
"""Bundled faces. Both toolkits register from here — the Qt shell and the
matplotlib figures — so a chart drawn in the window is set in the same
typeface as the words beside it."""

FONT_STACK = ("Inter", "Segoe UI Variable Text", "Segoe UI", "sans-serif")
FONT_MONO_STACK = ("Cascadia Mono", "Consolas", "SF Mono", "monospace")

FONT_WORDMARK = 29        # about-box wordmark
FONT_METRIC = 20          # headline metric figure (19.5 in the guide; Qt pixel
                          # sizes are integers, and half a pixel is not worth a
                          # second sizing mechanism)
FONT_TITLE = 15           # pane header, header-card title
FONT_SECTION = 14         # section-card title in a dialog
FONT_BODY = 13            # the default for everything
FONT_CONTROL = 13         # button label, tab label, field header
FONT_UNIT = 12            # unit beside a metric, status bar
FONT_CAPTION = 11         # metric caption, column head, explanation

WEIGHT_REGULAR = 400
WEIGHT_MEDIUM = 500
WEIGHT_SEMIBOLD = 600

# ---------------------------------------------------------------------------
# Space and shape
# ---------------------------------------------------------------------------
SPACE = (2, 4, 6, 8, 10, 12, 14, 16, 18)  # pick from the scale, never between

RADIUS_CONTROL = 6        # buttons, fields, combo boxes, menus
RADIUS_CONTAINER = 10     # grouped cards, drawing surfaces, panels
RADIUS_INDICATOR = 5      # checkbox indicator, menu-bar item
RADIUS_SMALL = 4          # in-cell controls, menu items

ROW_HEIGHT = 30           # table row
INDENT_UNIT = 16          # one indent step
ICON_SIZE = 15            # icon on a button or tab
BORDER_WIDTH = 1          # the only border weight there is


def tokens() -> dict[str, Any]:
    """Every token as a flat mapping, for substitution into a stylesheet."""
    return {
        name: value
        for name, value in globals().items()
        if name.isupper() and isinstance(value, (str, int, float))
    }


def series_colors(count: int) -> list[str]:
    """The first ``count`` categorical fills.

    Raises past the end of the set on purpose: a caller with more categories
    than hues must decide what to do about it (fold the tail into "Other",
    facet, or drop colour identity) rather than have a hue quietly repeat.
    """
    if count > len(SERIES):
        raise ValueError(
            f"{count} categories exceeds the {len(SERIES)}-hue categorical set; "
            "fold the tail into an 'Other' category or drop colour identity"
        )
    return list(SERIES[:count])


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def matplotlib_rc() -> "dict[str, Any]":
    """rcParams that put a figure on the same surface as the rest of the app."""
    return {
        "figure.facecolor": SURFACE,
        "figure.edgecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "savefig.edgecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": BORDER,
        "axes.labelcolor": SUBTLE,
        "axes.titlecolor": TEXT,
        "axes.linewidth": 0.8,
        "axes.axisbelow": True,
        "axes.labelsize": FONT_CAPTION,
        "axes.titlesize": FONT_BODY,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.titlepad": 8,
        "grid.color": BORDER,
        "grid.linewidth": 0.8,
        "xtick.color": TERTIARY,
        "ytick.color": TERTIARY,
        "xtick.labelcolor": SUBTLE,
        "ytick.labelcolor": SUBTLE,
        "xtick.labelsize": FONT_CAPTION,
        "ytick.labelsize": FONT_CAPTION,
        "legend.frameon": False,
        "legend.fontsize": FONT_CAPTION,
        "legend.labelcolor": SUBTLE,
        "font.family": "sans-serif",
        "font.sans-serif": list(FONT_STACK),
        "font.size": FONT_CAPTION,
        "text.color": TEXT,
        "lines.solid_capstyle": "round",
    }


def _register_fonts() -> None:
    """Hand the bundled faces to matplotlib's own font manager.

    Registering with Qt does not reach matplotlib: it keeps a separate font
    list, and without this a chart would quietly fall back to a second
    typeface — the one thing the type rules do not allow.
    """
    if not FONTS_DIR.is_dir():
        return
    from matplotlib import font_manager

    known = {Path(f.fname).name for f in font_manager.fontManager.ttflist}
    for path in sorted(FONTS_DIR.iterdir()):
        if path.suffix.lower() in {".ttf", ".otf"} and path.name not in known:
            try:
                font_manager.fontManager.addfont(str(path))
            except Exception:
                # A face that will not load costs polish, not function: the
                # stack falls through to the system UI face.
                pass


def apply_matplotlib_style() -> None:
    """Push the token rcParams onto matplotlib. Cheap, and idempotent."""
    import matplotlib

    _register_fonts()
    matplotlib.rcParams.update(matplotlib_rc())


def style_axes(ax, grid: str = "both", zero_line: bool = False) -> None:
    """Strip an axes back to its data: hairline gridlines, no box, quiet ticks.

    ``grid`` is "both", "x", "y" or "none" — a chart that direct-labels every
    mark does not also need an axis repeating the same figures.
    """
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BORDER)
    ax.tick_params(length=3, width=0.8, colors=TERTIARY)
    ax.grid(False)
    if grid in ("x", "both"):
        ax.xaxis.grid(True, which="major", color=BORDER, linewidth=0.8)
    if grid in ("y", "both"):
        ax.yaxis.grid(True, which="major", color=BORDER, linewidth=0.8)
    if zero_line:
        # One step firmer than a gridline: it is the reference the marks are
        # read against, not another interval.
        ax.axhline(0.0, color=TERTIARY, linewidth=1.0, zorder=1)


def rounded_barh(ax, y, width, height, color, radius=4.0):
    """A horizontal bar rounded on the data end only, square on the baseline.

    matplotlib has no per-corner radius, so the bar is drawn as an explicit
    path. It lives here rather than in a chart module because every bar in the
    application is drawn this way.
    """
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path

    # The radius is a screen radius, so convert it through the current x span
    # rather than treating it as a data distance.
    lo, hi = ax.get_xlim()
    span = (hi - lo) or 1.0
    r = min(radius / max(ax.bbox.width, 1.0) * span, abs(width) / 2.0, height / 2.0)
    sign = 1.0 if width >= 0 else -1.0
    end = width - sign * r
    y0, y1 = y - height / 2.0, y + height / 2.0
    vertices = [
        (0.0, y0), (end, y0),
        (width, y0), (width, y0 + r),
        (width, y1 - r), (width, y1), (end, y1),
        (0.0, y1), (0.0, y0),
    ]
    codes = [
        Path.MOVETO, Path.LINETO,
        Path.CURVE3, Path.CURVE3,
        Path.LINETO, Path.CURVE3, Path.CURVE3,
        Path.LINETO, Path.CLOSEPOLY,
    ]
    patch = PathPatch(Path(vertices, codes), facecolor=color, edgecolor="none", zorder=2)
    ax.add_patch(patch)
    return patch


def figure_header(fig, title: str, caveats: "str | list[str]") -> float:
    """Title, then the caveats under it in the explanatory voice.

    The subtitle is where a chart discloses what it capped, excluded or
    normalised, so it is part of the chart rather than an optional extra.
    Returns the figure fraction the axes may start at — with room left for the
    first row's own titles, which sit above the axes box.

    A single caveat may be passed as a string: taking one and iterating it
    character by character is a mistake worth absorbing here rather than
    printing down the side of a figure.
    """
    if isinstance(caveats, str):
        caveats = [caveats]
    width_in, height_in = fig.get_size_inches()
    height = height_in * fig.dpi
    # Wrapped here rather than by matplotlib's own ``wrap``, which measures
    # against the canvas as drawn — and a figure that is laid out before it is
    # shown has no canvas width to measure against yet.
    columns = max(int((width_in - 0.3) / (FONT_CAPTION * 0.5 / 72.0)), 20)
    lines: list[str] = []
    for caveat in caveats:
        lines += textwrap.wrap(caveat, columns) or [""]

    line = (FONT_CAPTION + 4) / height          # one caveat line, in figure units
    top = 1.0 - 10.0 / height                   # a little air above the title
    fig.text(0.006, top, title, ha="left", va="top",
             fontsize=FONT_TITLE, fontweight="semibold", color=TEXT)
    body = top - (FONT_TITLE + 8) / height
    if lines:
        fig.text(0.006, body, "\n".join(lines), ha="left", va="top",
                 fontsize=FONT_CAPTION, color=SUBTLE, style="italic", linespacing=1.5)
    panel_title = (FONT_BODY + 14) / height
    return max(body - line * (len(lines) + 1.8) - panel_title, 0.5)
