"""The icon set: one family, one stroke weight, one size.

The glyphs are authored here as SVG path data in the Lucide idiom — 24-unit
grid, round caps and joins, geometric construction, no fills — rather than
pulled from a package, because five destinations and a panel toggle do not
justify a dependency.

**Stroke width.** Lucide draws 2 px on its 24 grid. Rendered down to 16 px
that lands at 1.33 px, which reads thin beside 13 px type. ``2.25`` on the 24
grid is exactly 1.5 px at 16 px, which is the weight the rest of the interface
is drawn at.

**Colour is never baked in.** The tint is injected per state from the ink
ladder, so a change to a rung in :mod:`nncm.theme` reaches the icons without
anyone touching artwork. A resting neutral icon takes the glyph rung, never
primary ink — at primary it competes with the text beside it.

Every icon in this application sits beside its own text label, so no icon is
ever the sole carrier of meaning.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from PySide6.QtCore import QRectF, QStandardPaths, Qt
from PySide6.QtGui import QIcon, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer

from .. import theme as T

_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
    'stroke="{colour}" stroke-width="2.25" stroke-linecap="round" '
    'stroke-linejoin="round">{body}</svg>'
)

# The five destinations, plus the rail toggle. Each drawn on the 24 grid.
_GLYPHS: dict[str, str] = {
    # Project — a folder holding the configuration
    "folder": '<path d="M3 7a2 2 0 0 1 2-2h4l2 2.5h8a2 2 0 0 1 2 2V17a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2Z"/>',
    # Sample — points scattered over a design space
    "scatter": (
        '<path d="M4 4v15a1 1 0 0 0 1 1h15"/>'
        '<circle cx="9" cy="15" r="1.3"/><circle cx="13" cy="9" r="1.3"/>'
        '<circle cx="17.5" cy="12.5" r="1.3"/><circle cx="8" cy="8.5" r="1.3"/>'
    ),
    # Phast — a workbook going out and coming back
    "exchange": (
        '<path d="M4 8h11"/><path d="M12 5l3 3-3 3"/>'
        '<path d="M20 16H9"/><path d="M12 13l-3 3 3 3"/>'
    ),
    # Train — a network of units
    "network": (
        '<circle cx="5" cy="12" r="2"/><circle cx="19" cy="6.5" r="2"/>'
        '<circle cx="19" cy="17.5" r="2"/>'
        '<path d="M7 11 17 7"/><path d="M7 13l10 4"/>'
    ),
    # Predict — a value read off a rising trace
    "trend": (
        '<path d="M4 4v15a1 1 0 0 0 1 1h15"/>'
        '<path d="M7.5 16l3.5-4.5 3 2.5L20 8"/>'
    ),
    # The navigation rail toggle
    "panel-left": (
        '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M9.5 4v16"/>'
    ),
    # The log panel toggle
    "terminal": '<path d="M5 8l3.5 3.5L5 15"/><path d="M12 16h7"/>',
    # A style sheet can only reference an image by URL, so these two are also
    # written to disk by `glyph_url` below.
    "chevron-down": '<path d="M6 9.5l6 6 6-6"/>',
    "check": '<path d="M5 12.5l4.5 4.5L19 7"/>',
}

# Qt state to ink rung. A checked item is at primary because it is the current
# destination; everything at rest is glyph, so no icon outshouts a label.
_STATES = (
    (QIcon.Normal, QIcon.Off, T.INK_GLYPH),
    (QIcon.Active, QIcon.Off, T.INK_SECONDARY),   # hover
    (QIcon.Normal, QIcon.On, T.INK_PRIMARY),      # checked
    (QIcon.Active, QIcon.On, T.INK_PRIMARY),
    (QIcon.Selected, QIcon.On, T.INK_PRIMARY),
    (QIcon.Disabled, QIcon.Off, T.INK_DISABLED),
    (QIcon.Disabled, QIcon.On, T.INK_DISABLED),
)


def _render(body: str, colour: str, size: int, ratio: int = 2) -> QPixmap:
    """One glyph at one tint, drawn at 2x for a crisp edge on a HiDPI screen."""
    pixmap = QPixmap(size * ratio, size * ratio)
    pixmap.fill(Qt.transparent)
    pixmap.setDevicePixelRatio(ratio)
    renderer = QSvgRenderer(_SVG.format(colour=colour, body=body).encode("utf-8"))
    renderer.setAspectRatioMode(Qt.KeepAspectRatio)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing, True)
    # The target rect is not optional. Without it the SVG lays out against the
    # pixmap's *device* rect and the glyph overflows its box by the device
    # pixel ratio — it looks like the icon has been cropped to a corner.
    renderer.render(painter, QRectF(0, 0, size, size))
    painter.end()
    return pixmap


@lru_cache(maxsize=64)
def icon(name: str, size: int = T.ICON_SIZE) -> QIcon:
    """The named glyph, tinted for every state it can be drawn in."""
    try:
        body = _GLYPHS[name]
    except KeyError:
        raise KeyError(
            f"No '{name}' in the icon set. Add it in the set's idiom "
            f"(24 grid, 2.25 stroke, round caps, no fill), or use a text "
            f"label — never reach for a second family. "
            f"Available: {', '.join(sorted(_GLYPHS))}"
        ) from None
    built = QIcon()
    for mode, state, alpha in _STATES:
        built.addPixmap(_render(body, T.ink_hex(alpha), size), mode, state)
    return built


def names() -> tuple[str, ...]:
    """Every glyph in the set, for a test that wants to render them all."""
    return tuple(sorted(_GLYPHS))


@lru_cache(maxsize=16)
def glyph_url(name: str, colour: str | None = None, size: int = T.ICON_SIZE) -> str | None:
    """A cached PNG of one glyph, as a URL a style sheet can reference.

    A combo box arrow and a check mark have to be images: Qt's own
    border-triangle trick renders as a smear under Fusion, and a check mark
    has no CSS equivalent at all. Cached on disk because a repaint must never
    redraw an immutable asset, and returned with forward slashes because Qt
    does not read a backslash in a URL.

    ``colour`` defaults to the glyph rung. It is passed explicitly only where
    the glyph sits on a filled control — a check mark on the action fill takes
    the action's own text colour, not a rung of the ink ladder.

    Returns None if the cache cannot be written, in which case the caller
    leaves the rule out and Qt draws its own. A control is never allowed to
    fail over an ornament.
    """
    colour = colour or T.ink_hex(T.INK_GLYPH)
    root = QStandardPaths.writableLocation(QStandardPaths.CacheLocation)
    if not root:
        return None
    directory = Path(root) / "glyphs"
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    target = directory / f"{name}-{colour.lstrip('#')}-{size}.png"
    if not target.exists():
        try:
            if not _render(_GLYPHS[name], colour, size).save(str(target), "PNG"):
                return None
        except (OSError, KeyError):
            return None
    return target.as_posix()
