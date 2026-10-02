"""Original Times New Roman/STIX renderers retained without layout changes.

These functions are copied from the archived baseline. They require genuine
Times New Roman fonts; no silent font substitution is performed.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any
import matplotlib
matplotlib.use("Agg")
# Import Matplotlib before PyMuPDF, matching the source import contract.
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.animation import FuncAnimation, PillowWriter
import fitz
import numpy as np
from torch import Tensor
from PIL import Image
from .core import TrainingConfig
ANIMATION_FRAME_COUNT = 61
ANIMATION_FPS = 10
ANIMATION_DPI = 120


def configure_typography() -> dict[str, str]:
    """Require real Times New Roman faces and STIX mathtext before plotting."""
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


def style_history_axis(axis: plt.Axes) -> None:
    axis.tick_params(
        axis="both",
        which="major",
        direction="in",
        length=5.0,
        width=1.0,
        labelsize=13,
        pad=5.0,
    )
    axis.minorticks_off()
    axis.grid(True, alpha=0.25, linewidth=0.7)
    for spine in axis.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color("black")


def validate_pdf(path: Path) -> dict[str, Any]:
    """Check saved-page identity, real fonts, and the 12 pt text safety margin."""
    forbidden = ("dejavu", "liberation", "nimbus")
    with fitz.open(path) as document:
        if len(document) != 1:
            raise RuntimeError(f"Expected exactly one PDF page in {path.name}")
        page = document[0]
        font_names = sorted({entry[3] for entry in page.get_fonts(full=True)})
        compact_names = [
            name.replace("-", "").replace(" ", "").lower() for name in font_names
        ]
        if not any("timesnewroman" in name for name in compact_names):
            raise RuntimeError(
                f"Times New Roman was not embedded in {path.name}: {font_names}"
            )
        if any(any(bad in name.lower() for bad in forbidden) for name in font_names):
            raise RuntimeError(f"Fallback font found in {path.name}: {font_names}")
        text_blocks = [
            block for block in page.get_text("blocks") if str(block[4]).strip()
        ]
        if not text_blocks:
            raise RuntimeError(f"No visible text blocks found in {path.name}")
        minimum_margin = min(
            min(
                block[0],
                block[1],
                page.rect.width - block[2],
                page.rect.height - block[3],
            )
            for block in text_blocks
        )
        if minimum_margin < 12.0:
            raise RuntimeError(
                f"PDF text margin in {path.name} is {minimum_margin:.2f} pt; at least 12 pt is required"
            )
        # Exercise the post-save rasterization path used for human candidate review.
        page.get_pixmap(matrix=fitz.Matrix(144 / 72, 144 / 72), alpha=False)
        return {
            "page_count": len(document),
            "fonts": font_names,
            "minimum_text_margin_pt": float(minimum_margin),
        }


def save_training_history_pdf(
    history: list[dict[str, float]], output: Path, method_label: str | None = None
) -> dict[str, Any]:
    records = {key: np.asarray([row[key] for row in history]) for key in history[0]}
    figure, axes = plt.subplots(2, 2, figsize=(16.5, 11.0), facecolor="white")
    figure.suptitle(
        "Training History"
        if method_label is None
        else f"{method_label} Training History",
        fontsize=18,
        fontweight="bold",
        y=0.965,
    )
    panel_specs = (
        ("(a)", "Total Loss", [("total_loss", "Total loss", "#6A4C93")], "Loss"),
        (
            "(b)",
            "MSE Loss Components",
            [
                ("initial_condition_loss", r"$L_{\mathrm{IC}}$ (IC MSE)", "#4267AC"),
                ("boundary_condition_loss", r"$L_{\mathrm{BC}}$ (BC MSE)", "#FF924C"),
                ("pde_residual_loss", r"$L_{\mathrm{PDE}}$ (residual MSE)", "#FF595E"),
            ],
            "MSE loss",
        ),
        (
            "(c)",
            r"Relative $L_2$ Error",
            [("relative_l2_fixed_grid", r"Relative $L_2$ Error", "#FFCA3A")],
            r"Relative $L_2$ Error",
        ),
        (
            "(d)",
            "Learning Rate",
            [("learning_rate", "Learning rate", "#8AC926")],
            "Learning rate",
        ),
    )
    for axis, (panel_label, title, curves, y_label) in zip(
        axes.flat, panel_specs, strict=True
    ):
        for key, label, color in curves:
            linestyle = "--" if key == "learning_rate" else "-"
            finite = np.isfinite(records[key])
            axis.plot(
                records["epoch"][finite],
                records[key][finite],
                label=label,
                color=color,
                linewidth=2.0,
                linestyle=linestyle,
            )
        axis.set_yscale("log")
        axis.set_title(title, fontsize=17, pad=12)
        axis.set_xlabel("Epoch", fontsize=15, labelpad=7)
        axis.set_ylabel(y_label, fontsize=15, labelpad=7)
        axis.legend(fontsize=10, frameon=True, facecolor="white", edgecolor="#666666")
        axis.text(
            -0.14,
            1.04,
            panel_label,
            transform=axis.transAxes,
            fontsize=15,
            fontweight="bold",
        )
        style_history_axis(axis)
    figure.subplots_adjust(
        left=0.09, right=0.985, bottom=0.09, top=0.89, wspace=0.25, hspace=0.32
    )
    figure.savefig(output, format="pdf", facecolor="white", bbox_inches=None)
    plt.close(figure)
    return validate_pdf(output)


def save_solution_comparison_pdf(
    arrays: dict[str, np.ndarray],
    output: Path,
    method_label: str = "Vanilla PINN",
    equation_label: str = r"Inviscid Burgers equation: $u_t+u u_x=0$; analytical solution is used for evaluation only",
) -> dict[str, Any]:
    requested_times = np.asarray([0.0, 0.25, 0.50, 0.75, 1.0])
    time_indices = [
        int(np.argmin(np.abs(arrays["t"] - value))) for value in requested_times
    ]
    figure, axes = plt.subplots(1, 5, figsize=(23.0, 5.8), facecolor="white")
    figure.suptitle(
        f"{method_label} and Analytical Entropy Solution",
        fontsize=18,
        fontweight="bold",
        y=0.950,
    )
    figure.text(
        0.5,
        0.905,
        equation_label,
        ha="center",
        va="center",
        fontsize=14,
    )
    for axis, time_index, requested_time in zip(
        axes, time_indices, requested_times, strict=True
    ):
        axis.plot(
            arrays["x"],
            arrays["truth"][time_index],
            color="#4267AC",
            linewidth=2.5,
            drawstyle="steps-post",
            label="Analytical entropy solution",
        )
        axis.plot(
            arrays["x"],
            arrays["prediction"][time_index],
            color="#FF595E",
            linewidth=2.0,
            label=method_label,
        )
        axis.set_title(rf"$t={requested_time:0.2f}$", fontsize=16, pad=10)
        axis.set_xlim(-1.0, 1.0)
        axis.set_ylim(-0.15, 1.15)
        axis.set_xticks(np.linspace(-1.0, 1.0, 5))
        axis.set_yticks(np.linspace(0.0, 1.0, 5))
        axis.set_xlabel(r"Position, $x$ (dimensionless)", fontsize=11, labelpad=6)
        if axis is axes[0]:
            axis.set_ylabel(r"State, $u(x,t)$ (dimensionless)", fontsize=11, labelpad=6)
            axis.legend(
                loc="lower left",
                fontsize=10,
                frameon=True,
                facecolor="white",
                edgecolor="#666666",
            )
        axis.tick_params(
            axis="both",
            which="major",
            direction="out",
            length=4.0,
            width=1.0,
            labelsize=10,
            pad=4.0,
        )
        axis.grid(True, alpha=0.25, linewidth=0.7)
        for spine in axis.spines.values():
            spine.set_color("black")
            spine.set_linewidth(1.2)
    figure.subplots_adjust(left=0.055, right=0.975, bottom=0.19, top=0.76, wspace=0.28)
    figure.savefig(output, format="pdf", facecolor="white", bbox_inches=None)
    plt.close(figure)
    return validate_pdf(output)


def _representative_subset(values: np.ndarray, count: int) -> np.ndarray:
    """Select evenly indexed members of an existing fixed pool for display only."""
    if values.size <= count:
        return values
    return values[np.linspace(0, values.size - 1, count, dtype=int)]


def save_training_point_layout_pdf(
    fixed_points: dict[str, Tensor],
    config: TrainingConfig,
    output: Path,
    pde_display_count: int = 2400,
    equation_label: str = r"Inviscid Burgers PINN: $u_t+u u_x=0$ on $[-1,1]\times[0,1]$",
) -> dict[str, Any]:
    """Render a readable subset of the actual fixed training-coordinate pools."""

    def cpu_values(key: str) -> np.ndarray:
        return fixed_points[key].detach().cpu().numpy().reshape(-1)

    x_ic_left = _representative_subset(cpu_values("x_ic_left"), 80)
    x_ic_right = _representative_subset(cpu_values("x_ic_right"), 80)
    t_bc_left = _representative_subset(cpu_values("t_bc_left"), 80)
    t_bc_right = _representative_subset(cpu_values("t_bc_right"), 80)
    x_pde = _representative_subset(cpu_values("x_pde"), pde_display_count)
    t_pde = _representative_subset(cpu_values("t_pde"), pde_display_count)

    # A 16:9 composition keeps the figure legible when placed directly on a
    # presentation slide.  IC and BC markers are grouped by geometric role;
    # their prescribed values belong in the accompanying problem statement.
    figure, axis = plt.subplots(figsize=(13.333, 7.5), facecolor="white")
    figure.suptitle(
        "PINN Training Points in Space–Time", fontsize=20, fontweight="bold", y=0.965
    )
    figure.text(
        0.5,
        0.865,
        equation_label,
        ha="center",
        va="center",
        fontsize=14.5,
    )
    axis.scatter(
        x_pde,
        t_pde,
        s=14,
        marker="o",
        color="#6C8EBF",
        alpha=0.52,
        linewidths=0.0,
        label="PDE collocation points",
        zorder=1,
    )
    axis.scatter(
        np.concatenate((x_ic_left, x_ic_right)),
        np.zeros(x_ic_left.size + x_ic_right.size),
        s=34,
        marker="s",
        color="#E07A1F",
        edgecolors="white",
        linewidths=0.45,
        label=r"Initial-condition points ($t=0$)",
        zorder=4,
    )
    axis.scatter(
        np.concatenate(
            (
                np.full_like(t_bc_left, config.x_min),
                np.full_like(t_bc_right, config.x_max),
            )
        ),
        np.concatenate((t_bc_left, t_bc_right)),
        s=36,
        marker="^",
        color="#2A9D8F",
        edgecolors="white",
        linewidths=0.45,
        label=r"Boundary-condition points ($x=\pm1$)",
        zorder=4,
    )
    shock_times = np.linspace(config.t_min, config.t_max, 301)
    axis.plot(
        config.shock_speed * shock_times,
        shock_times,
        color="#D1495B",
        linestyle="--",
        linewidth=2.8,
        label=r"Shock path: $x_s(t)=0.5t$ (reference only)",
        zorder=3,
    )
    axis.set_xlim(config.x_min - 0.07, config.x_max + 0.07)
    axis.set_ylim(config.t_min - 0.04, config.t_max + 0.07)
    axis.set_xticks(np.linspace(config.x_min, config.x_max, 5))
    axis.set_yticks(np.linspace(config.t_min, config.t_max, 5))
    axis.set_xlabel(r"$x$", fontsize=15, labelpad=8)
    axis.set_ylabel(r"$t$", fontsize=15, labelpad=8)
    axis.set_aspect("equal", adjustable="box")
    style_history_axis(axis)
    axis.set_facecolor("#FBFCFE")
    axis.tick_params(axis="both", labelsize=12.5)
    axis.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.025),
        ncol=2,
        fontsize=11.5,
        frameon=True,
        facecolor="white",
        edgecolor="#B7C0CC",
        columnspacing=1.8,
        handletextpad=0.55,
    )
    figure.text(
        0.5,
        0.055,
        f"Displayed: 160 IC + 160 BC + {x_pde.size:,} PDE points   |   "
        "Fixed pools: 4,096 IC + 4,096 BC + 16,384 PDE points   |   "
        "Shuffled, not resampled, each epoch",
        ha="center",
        va="center",
        fontsize=10.5,
        color="#344054",
    )
    figure.subplots_adjust(left=0.085, right=0.975, bottom=0.145, top=0.70)
    figure.savefig(output, format="pdf", facecolor="white", bbox_inches=None)
    plt.close(figure)
    return validate_pdf(output)


def save_solution_animation_gif(
    arrays: dict[str, np.ndarray],
    config: TrainingConfig,
    output: Path,
    method_label: str = "Standard PINN",
    equation_label: str = r"$\frac{\partial u}{\partial t}+u\frac{\partial u}{\partial x}=0$",
    main_title: str = "Inviscid Burgers Equation",
) -> dict[str, Any]:
    """Render the final-model shock evolution without changing numerical results.

    The prediction and entropy solution arrays are evaluated before this CPU-only
    Matplotlib step.  The GIF is a candidate artifact for one trained model, not
    an animation of optimizer iterations.
    """
    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing animation candidate: {output}"
        )
    frame_times = np.linspace(config.t_min, config.t_max, ANIMATION_FRAME_COUNT)
    time_indices = [
        int(np.argmin(np.abs(arrays["t"] - value))) for value in frame_times
    ]
    figure, profile_axis = plt.subplots(
        1, 1, figsize=(8.0, 7.5), facecolor="white"
    )
    figure.suptitle(
        main_title, fontsize=24, fontweight="bold", y=0.982
    )
    figure.text(
        0.5,
        0.875,
        equation_label,
        ha="center",
        va="center",
        fontsize=26 if r"\partial^2" in equation_label else 30,
    )

    profile_axis.set_title(f"{method_label} Prediction", fontsize=20, pad=34)
    profile_axis.set_xlim(config.x_min, config.x_max)
    profile_axis.set_ylim(-0.1, 1.1)
    profile_axis.set_xticks(np.linspace(config.x_min, config.x_max, 5))
    profile_axis.set_yticks(np.linspace(0.0, 1.0, 5))
    profile_axis.set_xlabel(r"$x$", fontsize=17, labelpad=8)
    profile_axis.set_ylabel(r"$u$", fontsize=17, labelpad=8)
    style_history_axis(profile_axis)
    profile_axis.tick_params(axis="both", which="major", labelsize=14)
    (exact_line,) = profile_axis.plot(
        arrays["x"],
        arrays["truth"][time_indices[0]],
        color="#2F6BBD",
        linewidth=4.0,
        drawstyle="steps-post",
        label="Exact solution",
        zorder=2,
    )
    (prediction_line,) = profile_axis.plot(
        arrays["x"],
        arrays["prediction"][time_indices[0]],
        color="#D1495B",
        linewidth=3.5,
        label=f"{method_label} prediction",
        zorder=3,
    )
    profile_axis.legend(
        loc="center left",
        fontsize=14,
        frameon=True,
        facecolor="white",
        edgecolor="#AAB4C0",
    )
    time_text = profile_axis.text(
        0.5,
        1.018,
        "",
        transform=profile_axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=15,
    )
    profile_axis.set_position([0.145, 0.115, 0.82, 0.585])

    def update(frame: int) -> tuple[Any, ...]:
        time_index = time_indices[frame]
        time_value = frame_times[frame]
        exact_line.set_ydata(arrays["truth"][time_index])
        prediction_line.set_ydata(arrays["prediction"][time_index])
        time_text.set_text(rf"$t={time_value:0.2f}$")
        return exact_line, prediction_line, time_text

    animation = FuncAnimation(
        figure,
        update,
        frames=ANIMATION_FRAME_COUNT,
        interval=1000 / ANIMATION_FPS,
        blit=False,
        repeat=True,
    )
    animation.save(output, writer=PillowWriter(fps=ANIMATION_FPS), dpi=ANIMATION_DPI)
    plt.close(figure)
    with Image.open(output) as image:
        frame_count = int(getattr(image, "n_frames", 1))
        if frame_count != ANIMATION_FRAME_COUNT:
            raise RuntimeError(
                f"Expected {ANIMATION_FRAME_COUNT} GIF frames, found {frame_count}"
            )
        if image.info.get("loop") != 0:
            raise RuntimeError("GIF must loop continuously")
        return {
            "frame_count": frame_count,
            "size_px": list(image.size),
            "loop": image.info.get("loop"),
            "duration_ms": image.info.get("duration"),
            "quantity": "final-model prediction and analytical entropy solution over time",
        }
