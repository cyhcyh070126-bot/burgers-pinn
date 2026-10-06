"""Render a standards-compliant animation of the exact Burgers shock.

Plot classification: scientific teaching figure with a one-dimensional curve
and a space-time position diagram.  It contains no scalar field map, so a
colorbar would have no physical meaning.  This visualizes the analytical
entropy solution, not a PINN prediction.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.animation import FuncAnimation, PillowWriter
import fitz
import numpy as np
from PIL import Image


OUTPUT_DIR = Path(__file__).resolve().parent / "results"
PROFILE_GIF = OUTPUT_DIR / "burgers_exact_solution_profile.gif"
TRAJECTORY_GIF = OUTPUT_DIR / "burgers_shock_trajectory.gif"
CANDIDATE_GIF = OUTPUT_DIR / "burgers_shock_motion_candidate.gif"
CANDIDATE_PDF = OUTPUT_DIR / "burgers_shock_motion_reference_candidate.pdf"
CANDIDATE_PREVIEW = OUTPUT_DIR / "burgers_shock_motion_reference_candidate.png"
CANDIDATE_FRAME_PREVIEWS = (
    OUTPUT_DIR / "burgers_shock_motion_t000_candidate.png",
    OUTPUT_DIR / "burgers_shock_motion_t050_candidate.png",
    OUTPUT_DIR / "burgers_shock_motion_t100_candidate.png",
)

SHOCK_SPEED = 0.5
T_FINAL = 1.0
REFERENCE_TIME = 0.5
N_FRAMES = 61
FPS = 10
GIF_DPI = 120

FIGURE_SIZE = (13.333, 7.5)
SPLIT_FIGURE_SIZE = (8.0, 7.5)
MAIN_TITLE_SIZE = 24
INFORMATION_SIZE = 30
PANEL_TITLE_SIZE = 20
AXIS_LABEL_SIZE = 17
TICK_LABEL_SIZE = 14
LEGEND_SIZE = 14
ANNOTATION_SIZE = 15

SOLUTION_COLOR = "#2F6BBD"
SHOCK_COLOR = "#D1495B"
TIME_COLOR = "#F28E2B"
REFERENCE_COLOR = "#2A9D8F"


def configure_typography() -> dict[str, str]:
    """Require all four real Times New Roman faces and configure STIX math."""
    resolved: dict[str, str] = {}
    for label, style, weight in (
        ("regular", "normal", "normal"),
        ("bold", "normal", "bold"),
        ("italic", "italic", "normal"),
        ("bold_italic", "italic", "bold"),
    ):
        path = font_manager.findfont(
            font_manager.FontProperties(
                family="Times New Roman", style=style, weight=weight
            ),
            fallback_to_default=False,
        )
        resolved[label] = str(Path(path).resolve())

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    return resolved


def entropy_solution(x: np.ndarray, t: float) -> np.ndarray:
    """Return the entropy solution for the Riemann data u_L=1 and u_R=0."""
    return np.where(x < SHOCK_SPEED * t, 1.0, 0.0)


def style_axis(axis: plt.Axes) -> None:
    axis.tick_params(
        axis="both",
        which="major",
        direction="out",
        length=4.0,
        width=1.0,
        labelsize=TICK_LABEL_SIZE,
        pad=4.0,
    )
    axis.grid(True, color="#b0b0b0", linewidth=0.7, alpha=0.25)
    for spine in axis.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.25)


def build_figure():
    x = np.linspace(-1.0, 1.0, 1201)
    times = np.linspace(0.0, T_FINAL, N_FRAMES)
    trajectory_t = np.linspace(0.0, T_FINAL, 401)

    figure, (profile_axis, trajectory_axis) = plt.subplots(
        1, 2, figsize=FIGURE_SIZE, facecolor="white"
    )
    figure.suptitle(
        "Inviscid Burgers Equation",
        fontsize=MAIN_TITLE_SIZE,
        fontweight="bold",
        y=0.982,
    )
    figure.text(
        0.5,
        0.875,
        r"$\frac{\partial u}{\partial t}+u\frac{\partial u}{\partial x}=0$",
        ha="center",
        va="center",
        fontsize=INFORMATION_SIZE,
    )

    profile_axis.set_title(
        "Exact Solution Profile", fontsize=PANEL_TITLE_SIZE, pad=34
    )
    profile_axis.set_xlim(-1.0, 1.0)
    profile_axis.set_ylim(-0.1, 1.1)
    profile_axis.set_xticks(np.linspace(-1.0, 1.0, 5))
    profile_axis.set_yticks(np.linspace(0.0, 1.0, 5))
    profile_axis.set_xlabel(
        r"$x$", fontsize=AXIS_LABEL_SIZE, labelpad=8
    )
    profile_axis.set_ylabel(
        r"$u$", fontsize=AXIS_LABEL_SIZE, labelpad=8
    )
    style_axis(profile_axis)

    profile_line, = profile_axis.plot(
        x,
        entropy_solution(x, 0.0),
        color=SOLUTION_COLOR,
        linewidth=4.0,
        drawstyle="steps-post",
        label="Exact entropy solution",
        zorder=3,
    )
    shock_marker = profile_axis.axvline(
        0.0,
        color=SHOCK_COLOR,
        linestyle="--",
        linewidth=3.0,
        label="Shock position",
        zorder=4,
    )
    time_text = profile_axis.text(
        0.5,
        1.018,
        "",
        transform=profile_axis.transAxes,
        fontsize=ANNOTATION_SIZE,
        ha="center",
        va="bottom",
    )
    profile_axis.text(
        -0.80,
        0.91,
        r"$u_L=1$",
        color=SOLUTION_COLOR,
        fontsize=ANNOTATION_SIZE,
    )
    profile_axis.text(
        0.73,
        0.06,
        r"$u_R=0$",
        color=SOLUTION_COLOR,
        fontsize=ANNOTATION_SIZE,
    )
    profile_axis.legend(
        loc="center left",
        fontsize=LEGEND_SIZE,
        frameon=True,
        facecolor="white",
        edgecolor="#AAB4C0",
    )

    trajectory_axis.set_title(
        "Shock Trajectory", fontsize=PANEL_TITLE_SIZE, pad=14
    )
    trajectory_axis.set_xlim(-1.0, 1.0)
    trajectory_axis.set_ylim(0.0, T_FINAL)
    trajectory_axis.set_xticks(np.linspace(-1.0, 1.0, 5))
    trajectory_axis.set_yticks(np.linspace(0.0, T_FINAL, 5))
    trajectory_axis.set_xlabel(
        r"$x$", fontsize=AXIS_LABEL_SIZE, labelpad=8
    )
    trajectory_axis.set_ylabel(
        r"$t$", fontsize=AXIS_LABEL_SIZE, labelpad=8
    )
    style_axis(trajectory_axis)
    trajectory_axis.plot(
        SHOCK_SPEED * trajectory_t,
        trajectory_t,
        color=REFERENCE_COLOR,
        linestyle="--",
        linewidth=2.8,
        label=r"Exact path: $x_s(t)=0.5t$",
        zorder=2,
    )
    path_line, = trajectory_axis.plot(
        [], [], color=SHOCK_COLOR, linewidth=5.0, zorder=3
    )
    current_point, = trajectory_axis.plot(
        [], [], "o", color=SHOCK_COLOR, markersize=10.0, zorder=4
    )
    current_time = trajectory_axis.axhline(
        0.0, color=TIME_COLOR, linewidth=2.2, zorder=2
    )
    trajectory_axis.legend(
        loc="lower right",
        fontsize=LEGEND_SIZE,
        frameon=True,
        facecolor="white",
        edgecolor="#AAB4C0",
    )

    figure.subplots_adjust(
        left=0.075, right=0.975, bottom=0.115, top=0.700, wspace=0.24
    )

    def update(frame: int):
        t = float(times[frame])
        shock_x = SHOCK_SPEED * t
        profile_line.set_ydata(entropy_solution(x, t))
        shock_marker.set_xdata([shock_x, shock_x])
        time_text.set_text(rf"$t={t:0.2f},\quad x_s(t)={shock_x:0.2f}$")

        visible_t = trajectory_t[trajectory_t <= t]
        path_line.set_data(SHOCK_SPEED * visible_t, visible_t)
        current_point.set_data([shock_x], [t])
        current_time.set_ydata([t, t])
        return (
            profile_line,
            shock_marker,
            time_text,
            path_line,
            current_point,
            current_time,
        )

    return figure, times, update


def validate_pdf(path: Path) -> dict[str, object]:
    forbidden = ("dejavu", "liberation", "nimbus")
    with fitz.open(path) as document:
        if len(document) != 1:
            raise RuntimeError(f"Expected one PDF page, found {len(document)}")
        page = document[0]
        font_names = sorted({row[3] for row in page.get_fonts(full=True)})
        lowered = [name.lower() for name in font_names]
        if not any("timesnewroman" in name.replace("-", "").lower() for name in font_names):
            raise RuntimeError(f"Times New Roman is not embedded: {font_names}")
        if any(any(bad in name for bad in forbidden) for name in lowered):
            raise RuntimeError(f"Forbidden fallback font found: {font_names}")

        blocks = page.get_text("blocks")
        if not blocks:
            raise RuntimeError("No PDF text blocks were found")
        margins = [
            min(
                block[0],
                block[1],
                page.rect.width - block[2],
                page.rect.height - block[3],
            )
            for block in blocks
            if str(block[4]).strip()
        ]
        minimum_margin = float(min(margins))
        if minimum_margin < 12.0:
            raise RuntimeError(
                f"PDF text margin is {minimum_margin:.2f} pt; at least 12 pt is required"
            )
        return {
            "page_count": len(document),
            "fonts": font_names,
            "minimum_text_margin_pt": minimum_margin,
            "page_size_pt": [float(page.rect.width), float(page.rect.height)],
        }


def validate_gif(path: Path) -> dict[str, object]:
    with Image.open(path) as image:
        frame_count = int(getattr(image, "n_frames", 1))
        if frame_count != N_FRAMES:
            raise RuntimeError(f"Expected {N_FRAMES} GIF frames, found {frame_count}")
        if image.info.get("loop") != 0:
            raise RuntimeError("GIF must loop continuously")
        return {
            "frame_count": frame_count,
            "size_px": list(image.size),
            "loop": image.info.get("loop"),
            "duration_ms": image.info.get("duration"),
        }


def save_pdf_preview(pdf_path: Path, preview_path: Path) -> None:
    """Rasterize the saved PDF for visual review without changing its layout."""
    with fitz.open(pdf_path) as document:
        pixmap = document[0].get_pixmap(matrix=fitz.Matrix(150 / 72, 150 / 72), alpha=False)
        pixmap.save(preview_path)


def save_frame_previews(figure, update) -> list[str]:
    """Save the first, middle, and last frames for explicit visual review."""
    preview_indices = (0, (N_FRAMES - 1) // 2, N_FRAMES - 1)
    saved: list[str] = []
    for frame_index, destination in zip(
        preview_indices, CANDIDATE_FRAME_PREVIEWS, strict=True
    ):
        update(frame_index)
        figure.savefig(
            destination,
            format="png",
            dpi=GIF_DPI,
            facecolor="white",
            bbox_inches=None,
        )
        saved.append(str(destination))
    return saved


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    resolved_fonts = configure_typography()
    gif_reports: dict[str, object] = {}
    for output, visible_axis_index in (
        (PROFILE_GIF, 0),
        (TRAJECTORY_GIF, 1),
    ):
        figure, _, update = build_figure()
        figure.set_size_inches(*SPLIT_FIGURE_SIZE, forward=True)
        for index, axis in enumerate(figure.axes):
            axis.set_visible(index == visible_axis_index)
        figure.axes[visible_axis_index].set_position([0.145, 0.115, 0.82, 0.585])

        animation = FuncAnimation(
            figure,
            update,
            frames=N_FRAMES,
            interval=1000 / FPS,
            blit=False,
            repeat=True,
        )
        animation.save(output, writer=PillowWriter(fps=FPS), dpi=GIF_DPI)
        plt.close(figure)
        gif_reports[output.name] = validate_gif(output)

    report = {
        "completed": True,
        "classification": "scientific teaching curve and position figure",
        "quantity": "analytical entropy solution; not a PINN prediction",
        "equation": "partial u / partial t + u partial u / partial x = 0",
        "shock_path": "x_s(t) = 0.5 t",
        "colorbar": "not applicable to line and trajectory plots",
        "resolved_fonts": resolved_fonts,
        "gifs": gif_reports,
    }
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
