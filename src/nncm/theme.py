"""Design tokens — the single source of truth for every colour and metric.

Deliberately Qt-free. The desktop shell (:mod:`nncm.gui.theme`) turns these
into a style sheet; the matplotlib figures (:mod:`nncm.sampling`,
:mod:`nncm.training`) read them directly. A chart drawn inside the window and
the same chart saved as a PNG by the CLI therefore cannot drift apart, and
``core/`` never has to import anything from ``gui/`` to draw one.

**Canvas and ink.** The canvas is white and the ink is black; every neutral in
the application is that ink at a stated alpha over that canvas. There are no
hex greys and no tinted neutrals, so a neutral cannot acquire a hue by
accident. Reach a rung through :func:`ink` (for a style sheet) or
:func:`ink_hex` (for matplotlib, which cannot parse ``rgba()``).

Colour survives in five places only: semantic state, interactive selection,
data series, encoded values, and the app icon. Everything else is ink.

The categorical fill set is fixed, ordered and finite. It was validated on the
white canvas under the all-pairs rule (the pairlist a scatter needs): worst
pair dE 10.1 simulated deutan, 21.2 normal vision, every hue over 3:1 on the
canvas. :func:`series_colors` raises past the end of the set rather than
cycling — a cycled palette silently reuses a hue and lies about identity.
Callers fold the tail into :data:`SERIES_OTHER`, validated against the same
three (worst dE 18.4 simulated, 23.5 normal).

Run ``python -m nncm.theme`` to print the ladder with its contrast ratios.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Canvas and ink
# ---------------------------------------------------------------------------
CANVAS = "#FFFFFF"        # the one ground colour
INK = "#000000"           # the one mark colour

# The ink ladder: alpha over the canvas, and nothing else.
INK_PRIMARY = 1.00        # values, headings, active labels, body text
INK_SECONDARY = 0.64      # supporting text, inactive nav items, legend text
INK_TERTIARY = 0.56       # units, captions, metadata, placeholder, empty states
INK_GLYPH = 0.44          # icons, dividers, focus rings — never text
INK_DISABLED = 0.38       # inactive controls only

# Surfaces are the same ink, lower down the ladder.
SURFACE_WASH = 0.03       # alternate rows, read-only fields, the nav rail
SURFACE_HOVER = 0.05      # mouse-over feedback
SURFACE_PRESSED = 0.09    # active state, selection highlight, current nav item
SURFACE_SELECTED = 0.09
SURFACE_BORDER = 0.10     # default control and panel borders
SURFACE_BORDER_STRONG = 0.18  # borders needing more emphasis

# ---------------------------------------------------------------------------
# Semantic colour
# ---------------------------------------------------------------------------
# Used sparingly, and never alone: every use pairs the colour with words.
ACTION = "#2563EB"        # primary action, checked toggle, running progress
ACTION_HOVER = "#1D4ED8"
ACTION_PRESSED = "#1E40AF"
ACTION_TEXT = "#FFFFFF"

SUCCESS = "#16A34A"
SUCCESS_BG = "#F0FDF4"
WARNING = "#D97706"
WARNING_BG = "#FFFBEB"
ERROR = "#DC2626"
ERROR_BG = "#FEF2F2"
ERROR_PRESSED = "#B91C1C"

# Categorical chart fills, in slot order. Three is the whole set.
SERIES = ("#2A78D6", "#EB6834", "#1BAF7A")
SERIES_OTHER = "#48484A"  # the folded tail, never an identity of its own


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------
def _channels(hex_colour: str) -> tuple[int, int, int]:
    value = hex_colour.lstrip("#")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def ink(alpha: float) -> str:
    """An ink rung as an ``rgba()`` string, for a Qt style sheet."""
    r, g, b = _channels(INK)
    return f"rgba({r}, {g}, {b}, {alpha})"


def ink_hex(alpha: float) -> str:
    """An ink rung flattened to an opaque hex colour.

    For the libraries that cannot read ``rgba()`` — matplotlib, ``QPainter``,
    anything taking a colour as a string.
    """
    return blend(INK, CANVAS, alpha)


def blend(foreground: str, background: str, alpha: float) -> str:
    fr, fg, fb = _channels(foreground)
    br, bg, bb = _channels(background)
    r = round(fr * alpha + br * (1 - alpha))
    g = round(fg * alpha + bg * (1 - alpha))
    b = round(fb * alpha + bb * (1 - alpha))
    return f"#{r:02X}{g:02X}{b:02X}"


def sequential_ink(index: int, count: int) -> str:
    """An ordered ramp in neutral ink, for reference contours.

    Lines a value is read *off* — a fan of references behind the real data —
    are ordered chrome, not identities. Giving them a hue would collide with
    the categorical series sharing the axes.
    """
    low, high = INK_GLYPH * 0.6, INK_SECONDARY
    if count <= 1:
        return ink_hex((low + high) / 2)
    return ink_hex(low + (high - low) * index / (count - 1))


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
# Type
# ---------------------------------------------------------------------------
def _fonts_dir() -> Path:
    """Where the bundled faces live, surviving a frozen bundle."""
    beside_package = Path(__file__).resolve().parent / "gui" / "fonts"
    if beside_package.is_dir():
        return beside_package
    root = getattr(sys, "_MEIPASS", None)  # PyInstaller unpack directory
    return Path(root) / "nncm" / "gui" / "fonts" if root else beside_package


FONTS_DIR = _fonts_dir()
"""Bundled faces. Both toolkits register from here — the Qt shell and the
matplotlib figures — so a chart drawn in the window is set in the same
typeface as the words beside it."""

FONT_FAMILY = "Inter"
FONT_VARIABLE_FAMILY = "Inter Variable"
"""What Qt calls ``InterVariable.ttf``. A variable font registers under its own
family name, not as extra styles of the static one, so it has to be asked for
by this name or Qt hands back the static faces."""

_FALLBACKS = ("Segoe UI Variable Text", "Segoe UI", "sans-serif")

FONT_STACK = (FONT_VARIABLE_FAMILY, FONT_FAMILY) + _FALLBACKS
"""For Qt. The variable face first: its named instances cover the whole weight
axis from one file, and — the reason it is here — Qt applies its kerning at
sub-pixel positions, which the static faces only match once hinting is relaxed
(see ``FONT_HINTING`` in :mod:`nncm.gui.theme`). The static family stays behind
it so a missing variable file costs nothing."""

MPL_FONT_STACK = (FONT_FAMILY,) + _FALLBACKS
"""For matplotlib, which must have the static faces.

matplotlib has no variable-axis support: it reads a variable font's default
instance and nothing else, so every weight it registers from ``InterVariable``
comes back as 400 and the italic file is indistinguishable from the upright.
Asking it for the static family is what keeps a bold axis title bold. Same
typeface either way, so the figure still matches the window."""

FONT_MONO_STACK = ("Cascadia Mono", "Consolas", "SF Mono", "monospace")

FONT_DISPLAY_FAMILY = "Newsreader"
FONT_DISPLAY_STACK = (FONT_DISPLAY_FAMILY, "Newsreader Medium", "Georgia", "serif")
"""For titles and headings only: page titles, card headings, the brand and chart
titles. Everything a user reads to work — labels, values, tables — stays in
Inter. The bundled face is a static Medium instance of the variable font at a
16 pt optical size, cut so that Qt and matplotlib get the same glyphs (neither
handles the optical-size axis reliably). Windows registers it under its legacy
name, ``Newsreader Medium``, hence both entries."""
MPL_DISPLAY_STACK = (FONT_DISPLAY_FAMILY, "serif")
"""For matplotlib, which logs a warning for every absent family in a stack; the
bundled face is always registered, so no platform names are needed."""
WEIGHT_DISPLAY = 500

# Sizes in px. Qt scales px with DPI; pt diverges across platforms.
FONT_TITLE = 22           # the name of the current page. One per screen.
FONT_HEADING = 16         # section heading inside a page; the brand in the rail
FONT_METRIC = 20          # a headline metric figure
FONT_BODY = 14            # body, values, inputs
FONT_LABEL = 13           # labels, buttons, nav items, tabs
FONT_CAPTION = 12         # captions, units, status bar, empty states

WEIGHT_REGULAR = 400
WEIGHT_MEDIUM = 500
WEIGHT_SEMIBOLD = 600


def pt(px: float, dpi: int = 100) -> float:
    """Convert a px type token to matplotlib points.

    Qt sizes text in px; matplotlib sizes it in points. Passing a px token
    straight into an rcParam renders the text about 39 % oversized at dpi 100,
    which is the single most common way an embedded figure stops matching the
    window around it.
    """
    return px * 72.0 / dpi


# ---------------------------------------------------------------------------
# Space and shape — an 8 px grid, with 4 px adjustments allowed
# ---------------------------------------------------------------------------
MARGIN_WINDOW = 20        # window margins
MARGIN_GROUP = 16         # inside a panel
SPACING_ROW = 8           # between rows
SPACING_GROUP = 20        # between groups
SPACING_SECTION = 32      # between sections

CONTROL_HEIGHT = 32
CONTROL_COMPACT = 26
RADIUS = 6                # controls
RADIUS_PANEL = 8          # panels
ICON_SIZE = 16
ROW_HEIGHT = 32           # table row
FIELD_MIN_W = 128         # a value field never narrower than its longest value

SPLITTER_VISUAL = 1       # the line the user sees
SPLITTER_GRAB = 8         # the area the mouse can catch

MIN_DIALOG_W = 360

DURATION_MS = 150
animate = True            # motion, behind one switch every call site checks


def tokens() -> dict[str, Any]:
    """Every scalar token as a flat mapping, for substitution into a sheet."""
    return {
        name: value
        for name, value in globals().items()
        if name.isupper() and isinstance(value, (str, int, float))
    }


# ---------------------------------------------------------------------------
# Contrast audit
# ---------------------------------------------------------------------------
def _luminance(hex_colour: str) -> float:
    """WCAG 2.x relative luminance from sRGB hex."""
    out = []
    for channel in _channels(hex_colour):
        v = channel / 255.0
        out.append(v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4)
    return 0.2126 * out[0] + 0.7152 * out[1] + 0.0722 * out[2]


def contrast(one: str, two: str) -> float:
    a, b = _luminance(one), _luminance(two)
    lighter, darker = max(a, b), min(a, b)
    return (lighter + 0.05) / (darker + 0.05)


LADDER = (
    ("Primary", INK_PRIMARY, 21.0),
    ("Secondary", INK_SECONDARY, 6.7),
    ("Tertiary", INK_TERTIARY, 5.0),
    ("Glyph", INK_GLYPH, 3.2),
    ("Disabled", INK_DISABLED, 2.7),
)
"""Each rung with the ratio it must still clear against the canvas."""


def check_ladder() -> list[dict[str, Any]]:
    """Print and return the ink ladder with its measured contrast ratios."""
    results = []
    for name, alpha, floor in LADDER:
        flat = ink_hex(alpha)
        ratio = contrast(flat, CANVAS)
        results.append(
            {"name": name, "alpha": alpha, "hex": flat, "ratio": ratio, "floor": floor}
        )
        # Compared on the printed figure: the floors are quoted to one decimal,
        # so a rung must not be failed by the digits below the one shown.
        mark = "ok " if round(ratio, 1) >= floor else "LOW"
        print(f"  {mark} {name:10s} a={alpha:.2f}  {flat}  {ratio:5.1f}:1  (min {floor})")
    return results


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def matplotlib_rc() -> dict[str, Any]:
    """rcParams that put a figure on the same canvas as the rest of the app.

    Chart chrome — axes, ticks, grid, titles — is ink. Only the data keeps a
    colour. Every size goes through :func:`pt`, because matplotlib measures
    type in points and the tokens are in pixels.
    """
    return {
        "figure.facecolor": CANVAS,
        "figure.edgecolor": CANVAS,
        "savefig.facecolor": CANVAS,
        "savefig.edgecolor": "none",
        "axes.facecolor": CANVAS,
        "axes.edgecolor": ink_hex(SURFACE_BORDER_STRONG),
        "axes.labelcolor": ink_hex(INK_SECONDARY),
        "axes.titlecolor": ink_hex(INK_PRIMARY),
        "axes.linewidth": 0.8,
        "axes.axisbelow": True,
        "axes.labelsize": pt(FONT_CAPTION),
        "axes.titlesize": pt(FONT_LABEL),
        "axes.titleweight": "medium",
        "axes.titlelocation": "left",
        "axes.titlepad": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": ink_hex(SURFACE_BORDER),
        "grid.linewidth": 0.8,
        "xtick.color": ink_hex(INK_GLYPH),
        "ytick.color": ink_hex(INK_GLYPH),
        "xtick.labelcolor": ink_hex(INK_SECONDARY),
        "ytick.labelcolor": ink_hex(INK_SECONDARY),
        "xtick.labelsize": pt(FONT_CAPTION),
        "ytick.labelsize": pt(FONT_CAPTION),
        "legend.frameon": False,
        "legend.fontsize": pt(FONT_CAPTION),
        "legend.labelcolor": ink_hex(INK_SECONDARY),
        "font.family": "sans-serif",
        "font.sans-serif": list(MPL_FONT_STACK),
        "font.size": pt(FONT_CAPTION),
        "text.color": ink_hex(INK_PRIMARY),
        "figure.titlesize": pt(FONT_HEADING),
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
    # Panel titles are headings, so they take the display face. Titles are
    # placed left (``axes.titlelocation``), which is its own text object, so
    # each location is restyled rather than only ``ax.title``.
    for loc in ("left", "center", "right"):
        text = ax.get_title(loc=loc)
        if text:
            ax.set_title(text, loc=loc, fontfamily=list(MPL_DISPLAY_STACK))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(ink_hex(SURFACE_BORDER_STRONG))
    ax.tick_params(length=3, width=0.8, colors=ink_hex(INK_GLYPH))
    ax.grid(False)
    if grid in ("x", "both"):
        ax.xaxis.grid(True, which="major", color=ink_hex(SURFACE_BORDER), linewidth=0.8)
    if grid in ("y", "both"):
        ax.yaxis.grid(True, which="major", color=ink_hex(SURFACE_BORDER), linewidth=0.8)
    if zero_line:
        # One step firmer than a gridline: it is the reference the marks are
        # read against, not another interval.
        ax.axhline(0.0, color=ink_hex(INK_GLYPH), linewidth=1.0, zorder=1)


def rounded_barh(ax, y, width, height, color, radius=4.0):
    """A horizontal bar rounded on the data end only, square on the baseline.

    matplotlib has no per-corner radius, so the bar is drawn as an explicit
    path. It lives here rather than in a chart module because every bar in the
    application is drawn this way.
    """
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path as MplPath

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
        MplPath.MOVETO, MplPath.LINETO,
        MplPath.CURVE3, MplPath.CURVE3,
        MplPath.LINETO, MplPath.CURVE3, MplPath.CURVE3,
        MplPath.LINETO, MplPath.CLOSEPOLY,
    ]
    patch = PathPatch(MplPath(vertices, codes), facecolor=color, edgecolor="none", zorder=2)
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
    title_px, caption_px = FONT_HEADING, FONT_CAPTION
    # Wrapped here rather than by matplotlib's own ``wrap``, which measures
    # against the canvas as drawn — and a figure that is laid out before it is
    # shown has no canvas width to measure against yet.
    columns = max(int((width_in - 0.3) / (caption_px * 0.5 / 72.0)), 20)
    lines: list[str] = []
    for caveat in caveats:
        lines += textwrap.wrap(caveat, columns) or [""]

    line = (caption_px + 4) / height          # one caveat line, in figure units
    top = 1.0 - 10.0 / height                 # a little air above the title
    fig.text(0.006, top, title, ha="left", va="top", fontfamily=list(MPL_DISPLAY_STACK),
             fontsize=pt(title_px), fontweight="medium", color=ink_hex(INK_PRIMARY))
    body = top - (title_px + 8) / height
    if lines:
        fig.text(0.006, body, "\n".join(lines), ha="left", va="top",
                 fontsize=pt(caption_px), color=ink_hex(INK_SECONDARY),
                 style="italic", linespacing=1.5)
    panel_title = (FONT_LABEL + 14) / height
    return max(body - line * (len(lines) + 1.8) - panel_title, 0.5)


if __name__ == "__main__":  # pragma: no cover - a hand-run audit
    print("Ink ladder — contrast against the canvas:")
    check_ladder()
    print("\nSemantic colours on the canvas:")
    for name, value in (
        ("Action", ACTION), ("Success", SUCCESS), ("Warning", WARNING), ("Error", ERROR)
    ):
        print(f"      {name:10s} {value}  {contrast(value, CANVAS):5.1f}:1")
    print(f"      {'Action text':10s} {ACTION_TEXT} on {ACTION}  "
          f"{contrast(ACTION_TEXT, ACTION):5.1f}:1")
    print("\nData series slots on the canvas:")
    for index, value in enumerate(SERIES):
        print(f"      slot {index + 1:<5d} {value}  {contrast(value, CANVAS):5.1f}:1")
    print(f"      {'Other':10s} {SERIES_OTHER}  {contrast(SERIES_OTHER, CANVAS):5.1f}:1")
