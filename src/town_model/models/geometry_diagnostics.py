"""Semantic axes for stored historical box/town diagnostics.

This metadata is intentionally separate from the geometric algorithms that
produce the values and from generic classifier/image point transport.
"""

GEOMETRY_DIAGNOSTIC_AXES: dict[str, tuple[str, ...]] = {
    "box_low": ("class", "coordinate"),
    "box_high": ("class", "coordinate"),
    "frame_low": ("class", "house", "coordinate"),
    "frame_high": ("class", "house", "coordinate"),
    "window_low": ("class", "house", "window", "coordinate"),
    "window_high": ("class", "house", "window", "coordinate"),
    "frame_margin": ("point", "class", "house"),
    "frame_contains": ("point", "class", "house"),
    "window_margin": ("point", "class", "house", "window"),
    "window_contains": ("point", "class", "house", "window"),
    "house_margin": ("point", "class", "house"),
    "house_contains": ("point", "class", "house"),
}
