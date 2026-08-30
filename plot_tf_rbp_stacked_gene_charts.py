from __future__ import annotations

import argparse
import hashlib
import os
import tempfile
from collections import defaultdict
from pathlib import Path

cache_root = Path(tempfile.gettempdir())
os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "fontconfig-cache"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from PIL import Image


NON_SEX_LABEL = "Non-sex-specifically spliced genes"
SEX_LABEL = "Sex-specifically spliced genes"

NAVY = "#000080"
ROYAL_BLUE = "#4169E1"
LIGHT_BLUE = "#ADD8E6"
GRID_GREY = "#E0E0E0"

NON_SEX_STYLE = {
    "facecolor": LIGHT_BLUE,
    "edgecolor": "black",
    "hatch": "",
    "label": NON_SEX_LABEL,
}
SEX_STYLE = {
    "facecolor": NAVY,
    "edgecolor": "black",
    "hatch": "///",
    "label": SEX_LABEL,
}
plt.rcParams.update(
    {
        "font.family": "Times New Roman",
        "font.size": 8,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 10,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "black",
        "axes.linewidth": 1.0,
        "savefig.facecolor": "white",
        "hatch.linewidth": 0.7,
    }
)


def require_columns(df: pd.DataFrame, required: list[str], filename: str) -> None:
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"{filename} is missing required columns: {missing}")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def average_image_hash(path: Path, size: int = 16) -> str:
    with Image.open(path) as image:
        resized = (
            image.convert("L")
            .resize((size, size), Image.Resampling.LANCZOS)
        )
        pixel_values = np.asarray(resized, dtype=float).ravel()
    mean_value = sum(pixel_values) / len(pixel_values)
    bits = "".join("1" if value >= mean_value else "0" for value in pixel_values)
    return f"{int(bits, 2):0{size * size // 4}x}"


def analyze_png_duplicates(graph_dir: Path, report_path: Path) -> list[Path]:
    png_paths = sorted(graph_dir.glob("*.png"))
    exact_groups: dict[str, list[Path]] = defaultdict(list)
    visual_groups: dict[str, list[Path]] = defaultdict(list)

    for path in png_paths:
        exact_groups[file_sha256(path)].append(path)
        visual_groups[average_image_hash(path)].append(path)

    rows: list[dict[str, object]] = []
    for path in png_paths:
        exact_digest = file_sha256(path)
        visual_digest = average_image_hash(path)
        rows.append(
            {
                "filename": path.name,
                "sha256": exact_digest,
                "average_hash": visual_digest,
                "exact_duplicate_group_size": len(exact_groups[exact_digest]),
                "visual_hash_group_size": len(visual_groups[visual_digest]),
                "selected_for_recreation": (
                    path == sorted(exact_groups[exact_digest])[0]
                    and path == sorted(visual_groups[visual_digest])[0]
                ),
            }
        )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(report_path, index=False)
    return [path for path in png_paths if any(row["filename"] == path.name for row in rows)]


def load_strict_sex_specific_events(age_sex_path: Path) -> set[str]:
    age_sex = pd.read_csv(age_sex_path)
    require_columns(
        age_sex,
        ["converted_event_id", "strict_significant"],
        age_sex_path.name,
    )
    strict = age_sex["strict_significant"].astype(str).str.strip().str.lower().eq("yes")
    return set(age_sex.loc[strict, "converted_event_id"].dropna().astype(str))


def build_regulator_gene_summary(
    events_path: Path,
    sex_specific_events: set[str],
    regulator_type: str,
) -> pd.DataFrame:
    events = pd.read_csv(events_path)
    require_columns(events, ["dataset_name", "gene_id", "event_id"], events_path.name)

    rows: list[dict[str, object]] = []
    for regulator, group in events.groupby("dataset_name", dropna=False):
        clean_group = group.dropna(subset=["gene_id"]).copy()
        clean_group["gene_id"] = clean_group["gene_id"].astype(str)
        clean_group["event_id"] = clean_group["event_id"].astype(str)

        sex_genes = set(
            clean_group.loc[
                clean_group["event_id"].isin(sex_specific_events),
                "gene_id",
            ]
        )
        all_genes = set(clean_group["gene_id"])
        non_sex_genes = all_genes - sex_genes

        rows.append(
            {
                "regulator_type": regulator_type,
                "regulator": str(regulator),
                "total_spliced_genes": len(all_genes),
                "non_sex_specific_spliced_genes": len(non_sex_genes),
                "sex_specific_spliced_genes": len(sex_genes),
                "percent_non_sex_specific_spliced_genes": (
                    100 * len(non_sex_genes) / len(all_genes) if all_genes else 0.0
                ),
                "percent_sex_specific_spliced_genes": (
                    100 * len(sex_genes) / len(all_genes) if all_genes else 0.0
                ),
                "unique_splicing_events": clean_group["event_id"].nunique(),
                "sex_specific_splicing_events": clean_group.loc[
                    clean_group["event_id"].isin(sex_specific_events),
                    "event_id",
                ].nunique(),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(["total_spliced_genes", "regulator"], ascending=[False, True])
        .reset_index(drop=True)
    )


def build_regulator_class_summary(
    tf_events_path: Path,
    rbp_events_path: Path,
    sex_specific_events: set[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for regulator_type, path in [("TF", tf_events_path), ("RBP", rbp_events_path)]:
        events = pd.read_csv(path)
        require_columns(events, ["gene_id", "event_id"], path.name)
        events = events.dropna(subset=["gene_id"]).copy()
        events["gene_id"] = events["gene_id"].astype(str)
        events["event_id"] = events["event_id"].astype(str)

        sex_genes = set(
            events.loc[events["event_id"].isin(sex_specific_events), "gene_id"]
        )
        all_genes = set(events["gene_id"])
        rows.append(
            {
                "regulator_type": regulator_type,
                "regulator": regulator_type,
                "total_spliced_genes": len(all_genes),
                "non_sex_specific_spliced_genes": len(all_genes - sex_genes),
                "sex_specific_spliced_genes": len(sex_genes),
                "percent_non_sex_specific_spliced_genes": (
                    100 * len(all_genes - sex_genes) / len(all_genes)
                    if all_genes
                    else 0.0
                ),
                "percent_sex_specific_spliced_genes": (
                    100 * len(sex_genes) / len(all_genes) if all_genes else 0.0
                ),
            }
        )
    return pd.DataFrame(rows)


def hex_luminance(hex_color: str) -> float:
    """Return perceived luminance on a 0-1 scale."""
    hex_color = hex_color.lstrip("#")
    red, green, blue = (int(hex_color[index : index + 2], 16) for index in (0, 2, 4))
    return (0.299 * red + 0.587 * green + 0.114 * blue) / 255


def label_color(facecolor: str) -> str:
    return "white" if hex_luminance(facecolor) < 0.45 else "black"


def text_stroke(color: str) -> list[path_effects.AbstractPathEffect]:
    stroke_color = "black" if color == "white" else "white"
    return [path_effects.withStroke(linewidth=1.0, foreground=stroke_color)]


def format_ieee_axis(
    ax: plt.Axes,
    ylabel: str,
    x_rotation: int,
) -> None:
    ax.set_ylabel(ylabel)
    ax.tick_params(
        axis="x",
        rotation=x_rotation,
        labelsize=8,
        width=1.0,
        colors="black",
    )
    ax.tick_params(axis="y", labelsize=8, width=1.0, colors="black")
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)


def apply_bar_styles(
    bars,
    labels: list[str],
    base_style: dict[str, str],
) -> list[dict[str, str]]:
    styles = [base_style for _ in labels]
    for bar, style in zip(bars, styles):
        bar.set_facecolor(style["facecolor"])
        bar.set_edgecolor(style["edgecolor"])
        bar.set_linewidth(1.0)
        bar.set_hatch(style["hatch"])
    return styles


def add_value_labels(
    ax: plt.Axes,
    bars,
    values: np.ndarray,
    styles: list[dict[str, str]],
    percentage: bool,
    stacked_bottoms: np.ndarray | None = None,
    rotate_inside: bool = False,
) -> None:
    ymax = ax.get_ylim()[1]
    inside_threshold = 0.055 * ymax
    lane_step = 0.055 * ymax if stacked_bottoms is None else 0.038 * ymax
    callouts: list[dict[str, object]] = []

    for index, (bar, value, style) in enumerate(zip(bars, values, styles)):
        if value <= 0:
            continue

        label = f"{value:.1f}%" if percentage else f"{int(value)}"
        x_center = bar.get_x() + bar.get_width() / 2
        bottom = 0.0 if stacked_bottoms is None else float(stacked_bottoms[index])
        y_center = bottom + value / 2

        if value >= inside_threshold:
            ax.text(
                x_center,
                y_center,
                label,
                ha="center",
                va="center",
                fontsize=7.5,
                color=label_color(style["facecolor"]),
                rotation=90 if rotate_inside else 0,
                clip_on=True,
                zorder=5,
                path_effects=text_stroke(label_color(style["facecolor"])),
            )
            continue

        callouts.append(
            {
                "index": index,
                "label": label,
                "x": x_center,
                "y": bottom + value,
                "width": bar.get_width(),
            }
        )

    if not callouts:
        return

    total_lanes = 5 if stacked_bottoms is None and len(values) > 12 else 4
    total_lanes = 2 if len(values) <= 12 else total_lanes
    required_ymax = max(
        float(callout["y"]) + lane_step * (1.0 + order % total_lanes)
        for order, callout in enumerate(callouts)
    )
    if required_ymax > ymax * 0.92:
        ymax = required_ymax + lane_step
        ax.set_ylim(ax.get_ylim()[0], ymax)
    center_x = float(np.mean([callout["x"] for callout in callouts]))
    max_callout_y = ymax

    for order, callout in enumerate(callouts):
        x_center = float(callout["x"])
        segment_top = float(callout["y"])
        lane = order % total_lanes
        if stacked_bottoms is None:
            direction = 0
        elif len(values) <= 12:
            direction = -1 if x_center <= center_x else 1
        else:
            direction = 1

        x_offset = direction * float(callout["width"]) * (0.45 + 0.18 * lane)
        y_text = segment_top + lane_step * (1.0 + lane)
        max_callout_y = max(max_callout_y, y_text)
        ha = "center" if direction == 0 else ("right" if direction < 0 else "left")
        if len(values) <= 12 and stacked_bottoms is not None:
            ha = "center"
            x_offset = 0.0

        ax.annotate(
            str(callout["label"]),
            xy=(x_center, segment_top),
            xytext=(x_center + x_offset, y_text),
            textcoords="data",
            ha=ha,
            va="bottom",
            fontsize=7,
            color="black",
            path_effects=text_stroke("black"),
            arrowprops={
                "arrowstyle": "-",
                "color": "black",
                "linewidth": 0.5,
                "shrinkA": 1,
                "shrinkB": 1,
            },
            clip_on=False,
            zorder=6,
        )

    if max_callout_y > ymax * 0.96:
        ax.set_ylim(ax.get_ylim()[0], max_callout_y + lane_step)


def add_legend(ax: plt.Axes, anchor_y: float = -0.14) -> None:
    handles = [
        Patch(**NON_SEX_STYLE),
        Patch(**SEX_STYLE),
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, anchor_y),
        ncol=len(handles),
        frameon=False,
        borderaxespad=0.0,
        columnspacing=1.2,
        handletextpad=0.5,
        labelspacing=0.2,
    )


def draw_grouped_count_panel(
    ax: plt.Axes,
    summary: pd.DataFrame,
    x_rotation: int | None = None,
) -> None:
    labels = summary["regulator"].tolist()
    non_sex = summary["non_sex_specific_spliced_genes"].to_numpy(dtype=float)
    sex = summary["sex_specific_spliced_genes"].to_numpy(dtype=float)

    if x_rotation is None:
        x_rotation = 60 if len(labels) > 12 else 45

    x = np.arange(len(labels))
    grouped_width = 0.44

    count_ymax = max(float(np.nanmax(np.r_[non_sex, sex])) * 1.22, 1.0)
    ax.set_ylim(0, count_ymax)
    non_count_bars = ax.bar(
        x - grouped_width / 2,
        non_sex,
        width=grouped_width,
        label=NON_SEX_LABEL,
        zorder=3,
    )
    sex_count_bars = ax.bar(
        x + grouped_width / 2,
        sex,
        width=grouped_width,
        label=SEX_LABEL,
        zorder=3,
    )
    non_count_styles = apply_bar_styles(
        non_count_bars,
        labels,
        NON_SEX_STYLE,
    )
    sex_count_styles = apply_bar_styles(sex_count_bars, labels, SEX_STYLE)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, ha="right" if x_rotation else "center")
    format_ieee_axis(ax, "Number of spliced genes", x_rotation)
    add_value_labels(ax, non_count_bars, non_sex, non_count_styles, False)
    add_value_labels(ax, sex_count_bars, sex, sex_count_styles, False)


def draw_percentage_stacked_panel(
    ax: plt.Axes,
    summary: pd.DataFrame,
    x_rotation: int | None = None,
) -> None:
    labels = summary["regulator"].tolist()
    non_sex = summary["non_sex_specific_spliced_genes"].to_numpy(dtype=float)
    sex = summary["sex_specific_spliced_genes"].to_numpy(dtype=float)
    totals = summary["total_spliced_genes"].replace(0, np.nan).to_numpy(dtype=float)
    non_sex_pct = np.nan_to_num(non_sex / totals * 100)
    sex_pct = np.nan_to_num(sex / totals * 100)
    sex_pct = np.where((non_sex_pct + sex_pct) > 0, 100.0 - non_sex_pct, 0.0)

    if x_rotation is None:
        x_rotation = 60 if len(labels) > 12 else 45

    x = np.arange(len(labels))
    stacked_width = 0.64

    ax.set_ylim(0, 106)
    ax.set_xlim(-0.7, len(labels) - 0.3 + (0.8 if len(labels) > 16 else 0.0))
    non_pct_bars = ax.bar(
        x,
        non_sex_pct,
        width=stacked_width,
        label=NON_SEX_LABEL,
        zorder=3,
    )
    sex_pct_bars = ax.bar(
        x,
        sex_pct,
        bottom=non_sex_pct,
        width=stacked_width,
        label=SEX_LABEL,
        zorder=3,
    )
    non_pct_styles = apply_bar_styles(non_pct_bars, labels, NON_SEX_STYLE)
    sex_pct_styles = apply_bar_styles(sex_pct_bars, labels, SEX_STYLE)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, ha="right" if x_rotation else "center")
    ax.set_yticks(np.arange(0, 101, 20))
    format_ieee_axis(ax, "% of spliced genes", x_rotation)
    add_value_labels(
        ax,
        non_pct_bars,
        non_sex_pct,
        non_pct_styles,
        True,
        np.zeros_like(non_sex_pct),
        rotate_inside=len(labels) > 16,
    )
    add_value_labels(
        ax,
        sex_pct_bars,
        sex_pct,
        sex_pct_styles,
        True,
        non_sex_pct,
        rotate_inside=len(labels) > 16,
    )


def plot_grouped_count_chart(
    summary: pd.DataFrame,
    output_path: Path,
    x_rotation: int | None = None,
) -> None:
    labels = summary["regulator"].tolist()
    if x_rotation is None:
        x_rotation = 60 if len(labels) > 12 else 45

    fig_width = max(7, len(labels) * 0.56)
    fig, ax = plt.subplots(figsize=(fig_width, 5.6))
    draw_grouped_count_panel(ax, summary, x_rotation=x_rotation)
    add_legend(ax)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_percentage_stacked_chart(
    summary: pd.DataFrame,
    output_path: Path,
    x_rotation: int | None = None,
) -> None:
    labels = summary["regulator"].tolist()
    if x_rotation is None:
        x_rotation = 60 if len(labels) > 12 else 45

    fig_width = max(7, len(labels) * 0.42)
    fig, ax = plt.subplots(figsize=(fig_width, 5.6))
    draw_percentage_stacked_panel(ax, summary, x_rotation=x_rotation)
    add_legend(ax)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_charts(data_dir: Path, graph_dir: Path, table_dir: Path) -> None:
    tf_path = data_dir / "01_all_tf_splicing_events.csv"
    rbp_path = data_dir / "02_all_rbp_splicing_events.csv"
    age_sex_path = data_dir / "03_all_age_sex_splicing_events.csv"

    sex_specific_events = load_strict_sex_specific_events(age_sex_path)
    tf_summary = build_regulator_gene_summary(tf_path, sex_specific_events, "TF")
    rbp_summary = build_regulator_gene_summary(rbp_path, sex_specific_events, "RBP")
    combined_summary = build_regulator_class_summary(
        tf_path,
        rbp_path,
        sex_specific_events,
    )

    table_dir.mkdir(parents=True, exist_ok=True)
    tf_summary.to_csv(table_dir / "graph_a_b_tf_spliced_gene_summary.csv", index=False)
    rbp_summary.to_csv(table_dir / "graph_c_d_rbp_spliced_gene_summary.csv", index=False)
    combined_summary.to_csv(
        table_dir / "graph_e_tf_rbp_class_spliced_gene_summary.csv",
        index=False,
    )
    analyze_png_duplicates(graph_dir, table_dir / "png_duplicate_report.csv")

    plot_grouped_count_chart(
        tf_summary,
        graph_dir / "graph_tf_absolute_grouped_spliced_genes.png",
        x_rotation=45,
    )
    plot_percentage_stacked_chart(
        tf_summary,
        graph_dir / "graph_tf_100_percent_stacked_spliced_genes.png",
        x_rotation=45,
    )
    plot_grouped_count_chart(
        rbp_summary,
        graph_dir / "graph_rbp_absolute_grouped_spliced_genes.png",
        x_rotation=60,
    )
    plot_percentage_stacked_chart(
        rbp_summary,
        graph_dir / "graph_rbp_100_percent_stacked_spliced_genes.png",
        x_rotation=60,
    )
    plot_grouped_count_chart(
        combined_summary,
        graph_dir / "graph_tf_rbp_class_absolute_grouped_spliced_genes.png",
        x_rotation=0,
    )
    plot_percentage_stacked_chart(
        combined_summary,
        graph_dir / "graph_tf_rbp_class_100_percent_stacked_spliced_genes.png",
        x_rotation=0,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate publication-ready TF/RBP stacked spliced-gene charts."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--graph-dir", type=Path, default=Path("graphs"))
    parser.add_argument(
        "--table-dir",
        type=Path,
        default=Path("data/tf_rbp_output_analysis"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_charts(
        args.data_dir.expanduser().resolve(),
        args.graph_dir.expanduser().resolve(),
        args.table_dir.expanduser().resolve(),
    )


if __name__ == "__main__":
    main()
