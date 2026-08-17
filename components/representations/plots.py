"""Dependency-free SVG diagnostics for two real Module 2 smoke records."""

from __future__ import annotations

import html
import os
import tempfile
from pathlib import Path

import numpy as np
import numpy.typing as npt


def write_representation_diagnostic_svg(
    path: Path,
    *,
    logical_id: str,
    condition: str,
    mel: npt.NDArray[np.float32],
    recurrence: npt.NDArray[np.uint8],
    graph_nodes: npt.NDArray[np.float32],
    graph_edges: npt.NDArray[np.int16],
) -> None:
    """Write a visual-only Mel/RP/NVG panel; numerical arrays remain authoritative."""

    width, height = 1200, 460
    panel = 340
    top = 75
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        "<style>text{font-family:Arial,sans-serif;fill:#172033}"
        ".title{font-size:18px;font-weight:700}.label{font-size:14px;font-weight:600}"
        ".note{font-size:11px;fill:#526071}</style>",
        f'<text x="35" y="30" class="title">{html.escape(logical_id)} — Module 2 diagnostic</text>',
        f'<text x="35" y="51" class="note">{html.escape(condition)}; visual validation only, '
        "not a class comparison</text>",
    ]
    elements.extend(_heatmap(mel, 35, top, panel, "Log-Mel (power dB)"))
    elements.extend(_heatmap(recurrence.astype(np.float32), 430, top, panel, "Recurrence plot"))
    elements.extend(_graph(graph_nodes, graph_edges, 825, top, panel, "Natural visibility graph"))
    elements.append("</svg>")
    _atomic_write(path, "\n".join(elements) + "\n")


def write_crop_audit_diagnostic_svg(
    path: Path,
    *,
    logical_id: str,
    condition: str,
    duration_seconds: float,
    crop_start_seconds: float,
    crop_end_seconds: float,
    mel: npt.NDArray[np.float32],
    recurrence: npt.NDArray[np.uint8],
    graph_nodes: npt.NDArray[np.float32],
    graph_edges: npt.NDArray[np.int16],
) -> None:
    """Write the full-duration/crop relationship plus all three representations."""

    width, height = 1200, 555
    panel = 340
    top = 145
    timeline_left, timeline_width = 35.0, 1130.0
    crop_x = timeline_left + timeline_width * crop_start_seconds / duration_seconds
    crop_width = timeline_width * (crop_end_seconds - crop_start_seconds) / duration_seconds
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        "<style>text{font-family:Arial,sans-serif;fill:#172033}"
        ".title{font-size:18px;font-weight:700}.label{font-size:14px;font-weight:600}"
        ".note{font-size:11px;fill:#526071}</style>",
        f'<text x="35" y="28" class="title">{html.escape(logical_id)} — real crop audit</text>',
        f'<text x="35" y="49" class="note">{html.escape(condition)}; full recording '
        f'{duration_seconds:.6f} s; visual validation only</text>',
        f'<rect x="{timeline_left:.2f}" y="69" width="{timeline_width:.2f}" height="18" '
        'fill="#dbe3ed" stroke="#728197"/>',
        f'<rect x="{crop_x:.2f}" y="66" width="{max(crop_width, 2.0):.2f}" height="24" '
        'fill="#e05d44" opacity="0.9"/>',
        f'<text x="35" y="106" class="note">centered Mel crop: {crop_start_seconds:.6f}-'
        f'{crop_end_seconds:.6f} s (8.000000 s); RP and NVG summarize the complete '
        "recording</text>",
    ]
    elements.extend(_heatmap(mel, 35, top, panel, "Log-Mel (centered 8 s)"))
    elements.extend(
        _heatmap(recurrence.astype(np.float32), 430, top, panel, "Recurrence (complete signal)")
    )
    elements.extend(
        _graph(graph_nodes, graph_edges, 825, top, panel, "NVG (complete signal)")
    )
    elements.append("</svg>")
    _atomic_write(path, "\n".join(elements) + "\n")


def _heatmap(values: npt.NDArray[np.float32], x: int, y: int, size: int, label: str) -> list[str]:
    display_size = 56
    indices_y = np.linspace(0, values.shape[0] - 1, display_size, dtype=np.int64)
    indices_x = np.linspace(0, values.shape[1] - 1, display_size, dtype=np.int64)
    display = values[np.ix_(indices_y, indices_x)]
    low, high = float(np.min(display)), float(np.max(display))
    normalized = np.zeros_like(display, dtype=np.float64)
    if high > low:
        normalized = (display - low) / (high - low)
    cell = size / display_size
    result = [f'<text x="{x}" y="{y - 12}" class="label">{html.escape(label)}</text>']
    for row in range(display_size):
        for column in range(display_size):
            value = float(normalized[display_size - 1 - row, column])
            red = round(255 * np.clip(1.8 * value - 0.3, 0, 1))
            green = round(255 * np.clip(1.7 * value, 0, 1))
            blue = round(255 * np.clip(1.2 - 1.5 * value, 0, 1))
            result.append(
                f'<rect x="{x + column * cell:.2f}" y="{y + row * cell:.2f}" '
                f'width="{cell + 0.1:.2f}" height="{cell + 0.1:.2f}" '
                f'fill="rgb({red},{green},{blue})"/>'
            )
    result.append(
        f'<text x="{x}" y="{y + size + 18}" class="note">range {low:.4g} to {high:.4g}</text>'
    )
    return result


def _graph(
    nodes: npt.NDArray[np.float32],
    edges: npt.NDArray[np.int16],
    x: int,
    y: int,
    size: int,
    label: str,
) -> list[str]:
    amplitude = nodes[:, 1].astype(np.float64)
    low, high = float(np.min(amplitude)), float(np.max(amplitude))
    normalized = np.full(amplitude.shape, 0.5, dtype=np.float64)
    if high > low:
        normalized = (amplitude - low) / (high - low)
    xs = x + nodes[:, 0].astype(np.float64) * size
    ys = y + (1.0 - normalized) * size
    result = [f'<text x="{x}" y="{y - 12}" class="label">{html.escape(label)}</text>']
    for left, right in edges.T:
        source, target = int(left), int(right)
        if source >= target:
            continue
        result.append(
            f'<line x1="{xs[source]:.2f}" y1="{ys[source]:.2f}" '
            f'x2="{xs[target]:.2f}" y2="{ys[target]:.2f}" '
            'stroke="#a9b5c6" stroke-width="0.45" opacity="0.5"/>'
        )
    for node_x, node_y in zip(xs, ys, strict=True):
        result.append(f'<circle cx="{node_x:.2f}" cy="{node_y:.2f}" r="1.6" fill="#1558a6"/>')
    result.append(
        f'<text x="{x}" y="{y + size + 18}" class="note">'
        f"{nodes.shape[0]} nodes; {edges.shape[1] // 2} undirected edges</text>"
    )
    return result


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
