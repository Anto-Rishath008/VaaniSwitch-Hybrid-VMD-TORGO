"""Dependency-free representative SVG plots for Module 1 smoke reports."""

from __future__ import annotations

import html
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


def write_vmd_diagnostic_svg(
    path: Path,
    *,
    logical_id: str,
    waveform: FloatArray,
    modes: FloatArray,
    reconstruction: FloatArray,
    relative_energies: Sequence[float],
    reconstruction_label: str,
) -> None:
    """Atomically write a compact waveform, modes, reconstruction, and energy plot."""

    if modes.ndim != 2 or modes.shape[1] != waveform.size:
        raise ValueError("modes must have shape (K, waveform_length)")
    if reconstruction.shape != waveform.shape:
        raise ValueError("reconstruction must match waveform shape")

    width = 1_200
    left = 70
    right = width - 30
    panel_height = 120
    gap = 32
    panels = 2 + modes.shape[0]
    energy_height = 180
    height = 55 + panels * (panel_height + gap) + energy_height
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#1f2937}'
        '.label{font-size:15px;font-weight:600}.small{font-size:12px}</style>',
        f'<text x="{left}" y="28" class="label">{html.escape(logical_id)} '
        f'— {html.escape(reconstruction_label)} diagnostic</text>',
    ]

    y = 55
    elements.extend(
        _wave_panel(
            waveform,
            left=left,
            right=right,
            top=y,
            height=panel_height,
            label="Standardized input",
            color="#2563eb",
        )
    )
    y += panel_height + gap
    elements.extend(
        _overlay_panel(
            waveform,
            reconstruction,
            left=left,
            right=right,
            top=y,
            height=panel_height,
            reconstruction_label=reconstruction_label,
        )
    )
    y += panel_height + gap
    for index, mode in enumerate(modes, start=1):
        elements.extend(
            _wave_panel(
                mode,
                left=left,
                right=right,
                top=y,
                height=panel_height,
                label=f"Mode {index}",
                color="#7c3aed",
            )
        )
        y += panel_height + gap

    elements.append(f'<text x="{left}" y="{y}" class="label">Relative mode energy</text>')
    chart_top = y + 14
    chart_bottom = chart_top + 130
    count = max(len(relative_energies), 1)
    bar_space = (right - left) / count
    max_energy = max(max(relative_energies, default=0.0), 1e-12)
    for index, energy in enumerate(relative_energies, start=1):
        bar_height = 110 * max(float(energy), 0.0) / max_energy
        x = left + (index - 1) * bar_space + bar_space * 0.2
        bar_width = bar_space * 0.6
        elements.append(
            f'<rect x="{x:.2f}" y="{chart_bottom - bar_height:.2f}" '
            f'width="{bar_width:.2f}" height="{bar_height:.2f}" fill="#059669"/>'
        )
        elements.append(
            f'<text x="{x + bar_width / 2:.2f}" y="{chart_bottom + 17}" '
            f'text-anchor="middle" class="small">M{index}: {float(energy):.3f}</text>'
        )
    elements.append(
        f'<line x1="{left}" y1="{chart_bottom}" x2="{right}" y2="{chart_bottom}" '
        'stroke="#9ca3af" stroke-width="1"/>'
    )
    elements.append("</svg>")
    _atomic_write_text(path, "\n".join(elements) + "\n")


def _wave_panel(
    signal: FloatArray,
    *,
    left: int,
    right: int,
    top: int,
    height: int,
    label: str,
    color: str,
) -> list[str]:
    middle = top + height / 2
    return [
        f'<text x="{left}" y="{top - 7}" class="label">{html.escape(label)}</text>',
        f'<line x1="{left}" y1="{middle:.2f}" x2="{right}" y2="{middle:.2f}" '
        'stroke="#d1d5db" stroke-width="1"/>',
        f'<polyline points="{_polyline(signal, left, right, top, height)}" '
        f'fill="none" stroke="{color}" stroke-width="1"/>',
    ]


def _overlay_panel(
    original: FloatArray,
    reconstruction: FloatArray,
    *,
    left: int,
    right: int,
    top: int,
    height: int,
    reconstruction_label: str,
) -> list[str]:
    scale = max(float(np.max(np.abs(original), initial=0.0)), 1e-12)
    return [
        f'<text x="{left}" y="{top - 7}" class="label">Input vs '
        f'{html.escape(reconstruction_label)}</text>',
        f'<polyline points="{_polyline(original, left, right, top, height, scale=scale)}" '
        'fill="none" stroke="#2563eb" stroke-width="1.4"/>',
        f'<polyline points="{_polyline(reconstruction, left, right, top, height, scale=scale)}" '
        'fill="none" stroke="#dc2626" stroke-width="0.9" opacity="0.8"/>',
        f'<text x="{right - 205}" y="{top + 15}" class="small">'
        'blue=input; red=reconstruction</text>',
    ]


def _polyline(
    signal: FloatArray,
    left: int,
    right: int,
    top: int,
    height: int,
    *,
    max_points: int = 1_000,
    scale: float | None = None,
) -> str:
    if signal.size == 0:
        return ""
    count = min(signal.size, max_points)
    indices = np.linspace(0, signal.size - 1, count, dtype=np.int64)
    values = signal[indices]
    peak = max(
        float(np.max(np.abs(values), initial=0.0)) if scale is None else scale,
        1e-12,
    )
    normalized = np.clip(values / peak, -1.0, 1.0)
    xs = np.linspace(left, right, count)
    ys = top + height / 2 - normalized * (height * 0.45)
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in zip(xs, ys, strict=True))


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
