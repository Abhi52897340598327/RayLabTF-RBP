from __future__ import annotations

import argparse
import hashlib
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib

# Allows the script to run without a graphical desktop.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm


AGE_ORDER = ["3d", "6d", "9d"]
AGE_PATTERNS = [
    "3d",
    "6d",
    "9d",
    "3d + 6d",
    "3d + 9d",
    "6d + 9d",
    "3d + 6d + 9d",
]

PATTERN_ORDER = {
    "3d + 6d + 9d": 0,
    "3d + 6d": 1,
    "3d + 9d": 2,
    "6d + 9d": 3,
    "3d": 4,
    "6d": 5,
    "9d": 6,
}


# ---------------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------------


def file_sha256(path: Path) -> str:
    """Return a SHA-256 digest for exact-content duplicate detection."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def unique_files_by_content(files: Iterable[Path]) -> tuple[list[Path], pd.DataFrame]:
    """
    Keep one representative from each exact-content duplicate group.

    The shortest/cleanest filename is retained. A duplicate report is also
    returned so the removed copies are documented.
    """
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(files):
        groups[file_sha256(path)].append(path)

    selected: list[Path] = []
    report_rows: list[dict[str, object]] = []

    for digest, group in groups.items():
        chosen = sorted(group, key=lambda p: (len(p.name), p.name))[0]
        selected.append(chosen)

        for path in sorted(group):
            report_rows.append(
                {
                    "sha256": digest,
                    "filename": path.name,
                    "selected": path == chosen,
                    "selected_filename": chosen.name,
                    "duplicate_group_size": len(group),
                }
            )

    report = pd.DataFrame(report_rows)
    return sorted(selected), report


def clean_dataset_name(path: Path, dataset_type: str) -> str:
    """Create a readable dataset name from the filename."""
    prefix = f"significant_splicing_events_{dataset_type} - "
    name = path.stem
    if name.startswith(prefix):
        name = name[len(prefix) :]

    # Remove one or more trailing copy suffixes: (1), (2)(1), etc.
    name = re.sub(r"(?:\s*\(\d+\))+$", "", name).strip()

    canonical = {
        "fkh": "FKH",
        "mettl3": "METTL3",
    }
    return canonical.get(name, name)


def require_columns(df: pd.DataFrame, required: Iterable[str], filename: str) -> None:
    """Raise a clear error when a required input column is missing."""
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"{filename} is missing required columns: {missing}")


def save_figure(fig: plt.Figure, output_path: Path, dpi: int = 300) -> None:
    """Save and close a Matplotlib figure."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Load TF and RBP event sets
# ---------------------------------------------------------------------------


def load_event_sets(
    paths: list[Path],
    dataset_type: str,
) -> tuple[dict[str, set[str]], pd.DataFrame]:
    """
    Load each input file as a set of unique exact event_id strings.

    Using a set ensures that repeated rows or the same event appearing in
    multiple experiment labels within one file do not inflate the overlap.
    """
    event_sets: dict[str, set[str]] = {}
    metadata_rows: list[dict[str, object]] = []

    for path in paths:
        df = pd.read_csv(path)
        require_columns(df, ["event_id"], path.name)

        name = clean_dataset_name(path, dataset_type)
        if name in event_sets:
            raise ValueError(
                f"Two non-identical files resolved to the same dataset name: {name}. "
                "Rename one file or change clean_dataset_name()."
            )

        event_ids = set(df["event_id"].dropna().astype(str))
        event_sets[name] = event_ids

        experiments: list[str] = []
        if "experiment" in df.columns:
            experiments = sorted(df["experiment"].dropna().astype(str).unique())

        metadata_rows.append(
            {
                "dataset_type": dataset_type,
                "dataset_name": name,
                "filename": path.name,
                "input_rows": len(df),
                "unique_event_ids": len(event_ids),
                "experiment_count": len(experiments),
                "experiments": " | ".join(experiments),
            }
        )

    return event_sets, pd.DataFrame(metadata_rows)


def load_event_sets_from_table(
    table_path: Path,
    dataset_type: str,
) -> tuple[dict[str, set[str]], pd.DataFrame]:
    """Load precombined TF/RBP event tables produced from the raw source files."""
    df = pd.read_csv(table_path)
    require_columns(df, ["dataset_name", "event_id"], table_path.name)

    event_sets: dict[str, set[str]] = {}
    metadata_rows: list[dict[str, object]] = []

    for dataset_name, group in df.groupby("dataset_name", dropna=False):
        name = str(dataset_name)
        event_ids = set(group["event_id"].dropna().astype(str))
        event_sets[name] = event_ids

        experiments: list[str] = []
        if "experiment" in group.columns:
            experiments = sorted(group["experiment"].dropna().astype(str).unique())

        source_file = ""
        if "source_file" in group.columns and not group["source_file"].dropna().empty:
            source_file = str(group["source_file"].dropna().iloc[0])

        metadata_rows.append(
            {
                "dataset_type": dataset_type,
                "dataset_name": name,
                "filename": source_file or table_path.name,
                "input_rows": len(group),
                "unique_event_ids": len(event_ids),
                "experiment_count": len(experiments),
                "experiments": " | ".join(experiments),
            }
        )

    return event_sets, pd.DataFrame(metadata_rows)


def load_full_event_table(data_dir: Path, dataset_type: str) -> pd.DataFrame:
    """Load the full row-level TF or RBP event table."""
    combined_name = f"01_all_{dataset_type}_splicing_events.csv"
    if dataset_type == "rbp":
        combined_name = "02_all_rbp_splicing_events.csv"
    elif dataset_type == "tf":
        combined_name = "01_all_tf_splicing_events.csv"

    combined_path = data_dir / combined_name
    if combined_path.exists():
        df = pd.read_csv(combined_path)
        require_columns(df, ["dataset_name", "event_id"], combined_path.name)
        return df

    raw_candidates = sorted(
        data_dir.glob(f"significant_splicing_events_{dataset_type} - *.csv")
    )
    if not raw_candidates:
        raise FileNotFoundError(
            f"No {dataset_type.upper()} event table found in {data_dir}."
        )

    raw_candidates, _ = unique_files_by_content(raw_candidates)
    frames: list[pd.DataFrame] = []

    for path in raw_candidates:
        df = pd.read_csv(path)
        require_columns(df, ["event_id"], path.name)
        df = df.copy()
        df["dataset_name"] = clean_dataset_name(path, dataset_type)
        df["source_file"] = path.name
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def summarize_event_rows(df: pd.DataFrame) -> dict[str, object]:
    """Compress repeated row-level records for the same exact event."""
    summary: dict[str, object] = {}

    for column in ["gene_id", "gene_name", "event_type", "event_id"]:
        if column in df.columns:
            values = df[column].dropna().astype(str)
            summary[column] = values.iloc[0] if not values.empty else ""

    if "experiment" in df.columns:
        experiments = sorted(df["experiment"].dropna().astype(str).unique())
        summary["experiments"] = " | ".join(experiments)
        summary["experiment_count"] = len(experiments)

    if "dPSI" in df.columns:
        summary["dPSI_values"] = " | ".join(
            df["dPSI"].dropna().astype(str).tolist()
        )

    if "p_value" in df.columns:
        summary["p_values"] = " | ".join(
            df["p_value"].dropna().astype(str).tolist()
        )

    if "source_file" in df.columns:
        source_files = sorted(df["source_file"].dropna().astype(str).unique())
        summary["source_files"] = " | ".join(source_files)

    if "record_id" in df.columns:
        summary["row_count"] = int(df["record_id"].notna().sum())
    else:
        summary["row_count"] = int(len(df))

    return summary


def build_jaccard_site_table(
    tf_table: pd.DataFrame,
    rbp_table: pd.DataFrame,
    overlap_statistics: pd.DataFrame,
) -> pd.DataFrame:
    """Expand Jaccard matrix cells into site-level rows for the shared events."""
    records: list[dict[str, object]] = []

    tf_groups = {
        str(dataset_name): group.copy()
        for dataset_name, group in tf_table.groupby("dataset_name", dropna=False)
    }
    rbp_groups = {
        str(dataset_name): group.copy()
        for dataset_name, group in rbp_table.groupby("dataset_name", dropna=False)
    }

    for _, pair in overlap_statistics.iterrows():
        shared_count = int(pair["Shared_exact_events"])
        if shared_count <= 0:
            continue

        tf_name = str(pair["TF"])
        rbp_name = str(pair["Regulator"])
        tf_group = tf_groups.get(tf_name)
        rbp_group = rbp_groups.get(rbp_name)
        if tf_group is None or rbp_group is None:
            continue

        shared_ids = sorted(
            set(tf_group["event_id"].dropna().astype(str))
            & set(rbp_group["event_id"].dropna().astype(str))
        )

        tf_lookup = {
            str(event_id): summarize_event_rows(group)
            for event_id, group in tf_group.groupby("event_id", dropna=False)
        }
        rbp_lookup = {
            str(event_id): summarize_event_rows(group)
            for event_id, group in rbp_group.groupby("event_id", dropna=False)
        }

        for event_id in shared_ids:
            tf_summary = tf_lookup.get(event_id, {})
            rbp_summary = rbp_lookup.get(event_id, {})

            record = {
                "TF": tf_name,
                "RBP": rbp_name,
                "event_id": event_id,
                "shared_exact_events_in_pair": shared_count,
                "jaccard_similarity": float(pair["Jaccard_similarity"]),
                "tf_event_count": int(pair["TF_events"]),
                "rbp_event_count": int(pair["Regulator_events"]),
                "percent_of_tf_events": float(pair["Percent_of_TF_events"]),
                "percent_of_rbp_events": float(pair["Percent_of_regulator_events"]),
            }

            for prefix, summary in [("tf", tf_summary), ("rbp", rbp_summary)]:
                record[f"{prefix}_gene_id"] = summary.get("gene_id", "")
                record[f"{prefix}_gene_name"] = summary.get("gene_name", "")
                record[f"{prefix}_event_type"] = summary.get("event_type", "")
                record[f"{prefix}_experiments"] = summary.get("experiments", "")
                record[f"{prefix}_experiment_count"] = summary.get("experiment_count", 0)
                record[f"{prefix}_dPSI_values"] = summary.get("dPSI_values", "")
                record[f"{prefix}_p_values"] = summary.get("p_values", "")
                record[f"{prefix}_source_files"] = summary.get("source_files", "")
                record[f"{prefix}_row_count"] = summary.get("row_count", 0)

            record["gene_id"] = tf_summary.get("gene_id", rbp_summary.get("gene_id", ""))
            record["gene_name"] = tf_summary.get(
                "gene_name", rbp_summary.get("gene_name", "")
            )
            record["event_type"] = tf_summary.get(
                "event_type", rbp_summary.get("event_type", "")
            )

            records.append(record)

    result = pd.DataFrame(records)
    if not result.empty:
        result = result.sort_values(
            ["jaccard_similarity", "shared_exact_events_in_pair", "TF", "RBP", "event_id"],
            ascending=[False, False, True, True, True],
        ).reset_index(drop=True)
    return result


# ---------------------------------------------------------------------------
# TF × RBP overlap matrix
# ---------------------------------------------------------------------------


def calculate_tf_rbp_overlap(
    tf_sets: dict[str, set[str]],
    rbp_sets: dict[str, set[str]],
) -> pd.DataFrame:
    """Calculate raw exact-event overlap and normalized overlap statistics."""
    records: list[dict[str, object]] = []

    for tf_name, tf_events in tf_sets.items():
        for rbp_name, rbp_events in rbp_sets.items():
            shared = len(tf_events & rbp_events)
            union = len(tf_events | rbp_events)

            records.append(
                {
                    "TF": tf_name,
                    "Regulator": rbp_name,
                    "TF_events": len(tf_events),
                    "Regulator_events": len(rbp_events),
                    "Shared_exact_events": shared,
                    "Jaccard_similarity": shared / union if union else np.nan,
                    "Percent_of_TF_events": (
                        100.0 * shared / len(tf_events) if tf_events else np.nan
                    ),
                    "Percent_of_regulator_events": (
                        100.0 * shared / len(rbp_events) if rbp_events else np.nan
                    ),
                }
            )

    return pd.DataFrame(records)


def plot_tf_rbp_heatmap(
    statistics: pd.DataFrame,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create the large TF × RBP matrix.

    Cell text = exact shared-event count.
    Cell color = Jaccard similarity.
    """
    count_matrix = statistics.pivot(
        index="TF",
        columns="Regulator",
        values="Shared_exact_events",
    )
    jaccard_matrix = statistics.pivot(
        index="TF",
        columns="Regulator",
        values="Jaccard_similarity",
    )

    # Order rows/columns by average normalized overlap.
    row_order = jaccard_matrix.mean(axis=1).sort_values(ascending=False).index
    col_order = jaccard_matrix.mean(axis=0).sort_values(ascending=False).index

    ordered_jaccard = jaccard_matrix.loc[row_order, col_order]
    ordered_counts = count_matrix.loc[row_order, col_order]

    fig, ax = plt.subplots(figsize=(22, 8.5))
    image = ax.imshow(ordered_jaccard.values, aspect="auto", alpha=0.72)

    ax.set_xticks(np.arange(len(col_order)))
    ax.set_xticklabels(col_order, rotation=55, ha="right", fontsize=9)
    ax.set_yticks(np.arange(len(row_order)))
    ax.set_yticklabels(row_order, fontsize=10)

    ax.set_xlabel("RNA-binding protein / RNA-associated regulator")
    ax.set_ylabel("Transcription factor")
    ax.set_title(
        "Exact alternative-splicing event overlap between transcription factors "
        "and RNA-associated regulators\n"
        "Cell color = Jaccard similarity; cell label = number of shared exact events"
    )

    for row_index in range(len(row_order)):
        for col_index in range(len(col_order)):
            value = int(ordered_counts.iloc[row_index, col_index])
            ax.text(
                col_index,
                row_index,
                str(value),
                ha="center",
                va="center",
                fontsize=6.5,
            )

    colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    colorbar.set_label("Jaccard similarity")
    fig.tight_layout()

    save_figure(fig, output_dir / "tf_rbp_exact_event_overlap_heatmap.png")

    # Save a vector version as well.
    fig_svg, ax_svg = plt.subplots(figsize=(22, 8.5))
    image_svg = ax_svg.imshow(ordered_jaccard.values, aspect="auto", alpha=0.72)
    ax_svg.set_xticks(np.arange(len(col_order)))
    ax_svg.set_xticklabels(col_order, rotation=55, ha="right", fontsize=9)
    ax_svg.set_yticks(np.arange(len(row_order)))
    ax_svg.set_yticklabels(row_order, fontsize=10)
    ax_svg.set_xlabel("RNA-binding protein / RNA-associated regulator")
    ax_svg.set_ylabel("Transcription factor")
    ax_svg.set_title(
        "Exact alternative-splicing event overlap between transcription factors "
        "and RNA-associated regulators\n"
        "Cell color = Jaccard similarity; cell label = number of shared exact events"
    )
    for row_index in range(len(row_order)):
        for col_index in range(len(col_order)):
            ax_svg.text(
                col_index,
                row_index,
                str(int(ordered_counts.iloc[row_index, col_index])),
                ha="center",
                va="center",
                fontsize=6.5,
            )
    colorbar_svg = fig_svg.colorbar(image_svg, ax=ax_svg, fraction=0.025, pad=0.02)
    colorbar_svg.set_label("Jaccard similarity")
    fig_svg.tight_layout()
    fig_svg.savefig(
        output_dir / "tf_rbp_exact_event_overlap_heatmap.svg",
        bbox_inches="tight",
    )
    plt.close(fig_svg)

    return ordered_counts, ordered_jaccard


# ---------------------------------------------------------------------------
# Convert rMATS-style age files to the same event_id convention
# ---------------------------------------------------------------------------


def rmats_to_event_id(row: pd.Series) -> str | None:
    """
    Convert a 3d/6d/9d rMATS-style row to the event_id format used in the
    TF and RBP CSV files.

    The age files store exon starts as zero-based coordinates. The TF/RBP
    event IDs use one-based starts, so +1 is applied where required.

    Supported event types:
        SE, RI, A3SS, A5SS
    """
    gene = str(row["GeneID"])
    event_type = str(row["event_type"])
    chrom = str(row["chr"])
    if chrom.startswith("chr"):
        chrom = chrom[3:]
    strand = str(row["strand"])

    exon_start = int(row["exonStart_0base"]) + 1
    exon_end = int(row["exonEnd"])
    upstream_start = int(row["upstreamES"]) + 1
    upstream_end = int(row["upstreamEE"])
    downstream_start = int(row["downstreamES"]) + 1
    downstream_end = int(row["downstreamEE"])

    if event_type == "SE":
        # Junction into the skipped exon : junction out of the skipped exon
        return (
            f"{gene};SE:{chrom}:"
            f"{upstream_end}-{exon_start}:"
            f"{exon_end}-{downstream_start}:{strand}"
        )

    if event_type == "RI":
        return (
            f"{gene};RI:{chrom}:"
            f"{upstream_start}:"
            f"{upstream_end}-{downstream_start}:"
            f"{downstream_end}:{strand}"
        )

    if event_type == "A3SS":
        if strand == "+":
            return (
                f"{gene};A3:{chrom}:"
                f"{downstream_end}-{exon_start}:"
                f"{downstream_end}-{upstream_start}:{strand}"
            )
        return (
            f"{gene};A3:{chrom}:"
            f"{exon_end}-{downstream_start}:"
            f"{upstream_end}-{downstream_start}:{strand}"
        )

    if event_type == "A5SS":
        if strand == "+":
            return (
                f"{gene};A5:{chrom}:"
                f"{exon_end}-{downstream_start}:"
                f"{upstream_end}-{downstream_start}:{strand}"
            )
        return (
            f"{gene};A5:{chrom}:"
            f"{downstream_end}-{exon_start}:"
            f"{downstream_end}-{upstream_start}:{strand}"
        )

    return None


def load_age_files(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Load and standardize the 3d, 6d, and 9d files."""
    combined_path = data_dir / "03_all_age_sex_splicing_events.csv"
    if combined_path.exists():
        df = pd.read_csv(combined_path)
        require_columns(
            df,
            ["age", "converted_event_id", "strict_significant", "dPSI", "fdr", "geneSymbol"],
            combined_path.name,
        )

        frames: dict[str, pd.DataFrame] = {}
        for age in AGE_ORDER:
            age_frame = df[df["age"].astype(str) == age].copy()
            age_frame["event_id"] = age_frame["converted_event_id"].astype(str)
            age_frame["strict"] = (
                age_frame["strict_significant"].astype(str).str.strip().str.lower().eq("yes")
            )
            age_frame["FDR"] = pd.to_numeric(age_frame["fdr"], errors="coerce")
            age_frame["IncLevelDifference"] = pd.to_numeric(
                age_frame["dPSI"], errors="coerce"
            )
            frames[age] = age_frame

        return frames

    required = [
        "GeneID",
        "geneSymbol",
        "event_type",
        "chr",
        "strand",
        "exonStart_0base",
        "exonEnd",
        "upstreamES",
        "upstreamEE",
        "downstreamES",
        "downstreamEE",
        "Significant",
        "FDR",
        "IncLevelDifference",
    ]

    frames: dict[str, pd.DataFrame] = {}

    for age in AGE_ORDER:
        path = data_dir / f"Ray Lab Data - {age}_significant.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing required age file: {path}")

        df = pd.read_csv(path)
        require_columns(df, required, path.name)

        df = df.copy()
        df["event_id"] = df.apply(rmats_to_event_id, axis=1)
        df["strict"] = (
            df["Significant"].astype(str).str.strip().str.lower().eq("yes")
        )
        frames[age] = df

    return frames


# ---------------------------------------------------------------------------
# Candidate TF–RBP events and age/sex matching
# ---------------------------------------------------------------------------


def build_candidate_age_table(
    tf_sets: dict[str, set[str]],
    rbp_sets: dict[str, set[str]],
    age_frames: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Identify exact events found in at least one TF and at least one RBP, then
    match those candidate events to the age/sex datasets.
    """
    all_tf_events = set().union(*tf_sets.values())
    all_rbp_events = set().union(*rbp_sets.values())
    candidate_events = all_tf_events & all_rbp_events

    event_to_tfs = {
        event: sorted([name for name, events in tf_sets.items() if event in events])
        for event in candidate_events
    }
    event_to_rbps = {
        event: sorted([name for name, events in rbp_sets.items() if event in events])
        for event in candidate_events
    }

    records: list[dict[str, object]] = []

    for event in sorted(candidate_events):
        record: dict[str, object] = {
            "event_id": event,
            "TFs": ", ".join(event_to_tfs[event]),
            "RBPs": ", ".join(event_to_rbps[event]),
            "n_TFs": len(event_to_tfs[event]),
            "n_RBPs": len(event_to_rbps[event]),
        }

        gene_symbol: str | None = None
        strict_ages: list[str] = []

        for age, frame in age_frames.items():
            matched = frame[frame["event_id"] == event].copy()

            if matched.empty:
                record[f"{age}_dPSI"] = np.nan
                record[f"{age}_FDR"] = np.nan
                record[f"{age}_strict"] = False
                continue

            # If duplicate matches exist, prioritize a strict row, then the
            # lowest FDR.
            matched = matched.sort_values(
                ["strict", "FDR"],
                ascending=[False, True],
                na_position="last",
            )
            row = matched.iloc[0]

            if gene_symbol is None and pd.notna(row.get("geneSymbol")):
                gene_symbol = str(row["geneSymbol"])

            is_strict = bool(row["strict"])
            record[f"{age}_strict"] = is_strict
            record[f"{age}_dPSI"] = float(row["IncLevelDifference"])
            record[f"{age}_FDR"] = float(row["FDR"])

            if is_strict:
                strict_ages.append(age)

        # The age plots include only candidate events significant at >=1 age.
        if not strict_ages:
            continue

        gene_id = event.split(";", maxsplit=1)[0]
        event_type = event.split(";", maxsplit=1)[1].split(":", maxsplit=1)[0]

        record["gene_id"] = gene_id
        record["geneSymbol"] = gene_symbol or gene_id
        record["event_type"] = event_type
        record["strict_age_pattern"] = " + ".join(strict_ages)
        record["n_strict_ages"] = len(strict_ages)
        record["max_abs_dPSI"] = max(
            abs(float(record[f"{age}_dPSI"]))
            for age in strict_ages
            if pd.notna(record[f"{age}_dPSI"])
        )
        records.append(record)

    result = pd.DataFrame(records)
    if result.empty:
        raise RuntimeError(
            "No exact TF–RBP candidate events matched a strict age/sex event."
        )

    result["pattern_order"] = result["strict_age_pattern"].map(PATTERN_ORDER)
    result = result.sort_values(
        ["pattern_order", "max_abs_dPSI", "n_TFs", "n_RBPs"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)

    # Create readable unique row labels for the heatmap.
    duplicate_counter: dict[str, int] = defaultdict(int)
    labels: list[str] = []

    for _, row in result.iterrows():
        base = f"{row['geneSymbol']} ({row['event_type']})"
        duplicate_counter[base] += 1
        suffix = (
            f" #{duplicate_counter[base]}" if duplicate_counter[base] > 1 else ""
        )
        labels.append(
            f"{base}{suffix}  |  {int(row['n_TFs'])} TF, "
            f"{int(row['n_RBPs'])} RBP"
        )

    result["plot_label"] = labels
    return result


def plot_age_dpsi_heatmap(result: pd.DataFrame, output_dir: Path) -> None:
    """Plot dPSI for strict age/sex-associated candidate events."""
    matrix = np.full((len(result), len(AGE_ORDER)), np.nan)

    for row_index, row in result.iterrows():
        for col_index, age in enumerate(AGE_ORDER):
            if bool(row[f"{age}_strict"]):
                matrix[row_index, col_index] = float(row[f"{age}_dPSI"])

    max_abs = float(np.nanmax(np.abs(matrix)))
    if max_abs == 0:
        max_abs = 1.0

    fig_height = max(14.0, len(result) * 0.34)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    masked = np.ma.masked_invalid(matrix)
    norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)
    image = ax.imshow(masked, aspect="auto", norm=norm)

    ax.set_xticks(np.arange(len(AGE_ORDER)))
    ax.set_xticklabels(["3-day", "6-day", "9-day"], fontsize=11)
    ax.set_yticks(np.arange(len(result)))
    ax.set_yticklabels(result["plot_label"], fontsize=7.5)

    ax.set_xlabel("Age at male-versus-female comparison")
    ax.set_ylabel("Exact TF–RBP candidate splicing event")
    ax.set_title(
        "Age and sex association of exact TF–RBP candidate splicing events\n"
        "Positive dPSI = higher inclusion in males; "
        "negative dPSI = higher inclusion in females"
    )

    for row_index in range(len(result)):
        for col_index in range(len(AGE_ORDER)):
            if not np.isnan(matrix[row_index, col_index]):
                ax.text(
                    col_index,
                    row_index,
                    f"{matrix[row_index, col_index]:+.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                )

    # Separate the multi-age and age-specific blocks visually.
    previous_pattern: str | None = None
    for row_index, pattern in enumerate(result["strict_age_pattern"]):
        if previous_pattern is not None and pattern != previous_pattern:
            ax.axhline(row_index - 0.5, linewidth=1)
        previous_pattern = pattern

    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.03)
    colorbar.set_label("Male − female inclusion difference (dPSI)")
    fig.tight_layout()

    save_figure(fig, output_dir / "tf_rbp_candidate_age_sex_dpsi_heatmap.png")

    # Vector copy.
    fig_svg, ax_svg = plt.subplots(figsize=(10, fig_height))
    image_svg = ax_svg.imshow(masked, aspect="auto", norm=norm)
    ax_svg.set_xticks(np.arange(len(AGE_ORDER)))
    ax_svg.set_xticklabels(["3-day", "6-day", "9-day"], fontsize=11)
    ax_svg.set_yticks(np.arange(len(result)))
    ax_svg.set_yticklabels(result["plot_label"], fontsize=7.5)
    ax_svg.set_xlabel("Age at male-versus-female comparison")
    ax_svg.set_ylabel("Exact TF–RBP candidate splicing event")
    ax_svg.set_title(
        "Age and sex association of exact TF–RBP candidate splicing events\n"
        "Positive dPSI = higher inclusion in males; "
        "negative dPSI = higher inclusion in females"
    )
    for row_index in range(len(result)):
        for col_index in range(len(AGE_ORDER)):
            if not np.isnan(matrix[row_index, col_index]):
                ax_svg.text(
                    col_index,
                    row_index,
                    f"{matrix[row_index, col_index]:+.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                )
    previous_pattern = None
    for row_index, pattern in enumerate(result["strict_age_pattern"]):
        if previous_pattern is not None and pattern != previous_pattern:
            ax_svg.axhline(row_index - 0.5, linewidth=1)
        previous_pattern = pattern
    colorbar_svg = fig_svg.colorbar(image_svg, ax=ax_svg, fraction=0.035, pad=0.03)
    colorbar_svg.set_label("Male − female inclusion difference (dPSI)")
    fig_svg.tight_layout()
    fig_svg.savefig(
        output_dir / "tf_rbp_candidate_age_sex_dpsi_heatmap.svg",
        bbox_inches="tight",
    )
    plt.close(fig_svg)


def plot_age_persistence_summary(result: pd.DataFrame, output_dir: Path) -> pd.Series:
    """Plot total candidate-event counts for every age-specificity pattern."""
    counts = (
        result["strict_age_pattern"]
        .value_counts()
        .reindex(AGE_PATTERNS, fill_value=0)
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(counts.index, counts.values)
    ax.set_xlabel("Ages with sex-associated splicing")
    ax.set_ylabel("Number of exact TF–RBP candidate events")
    ax.set_title("Age specificity and persistence of TF–RBP candidate events")
    ax.tick_params(axis="x", rotation=30)

    for bar, value in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            str(int(value)),
            ha="center",
            va="bottom",
        )

    fig.tight_layout()
    save_figure(fig, output_dir / "tf_rbp_candidate_age_persistence_summary.png")
    return counts


# ---------------------------------------------------------------------------
# Male-higher, female-higher, and combined age graphs
# ---------------------------------------------------------------------------


def classify_sex_direction_by_pattern(result: pd.DataFrame) -> pd.DataFrame:
    """
    Classify each event pattern as:
      - Male-higher: positive dPSI at every significant age
      - Female-higher: negative dPSI at every significant age
      - Mixed direction: the sign changes across significant ages
    """
    summary_rows: list[dict[str, object]] = []

    for pattern in AGE_PATTERNS:
        subset = result[result["strict_age_pattern"] == pattern]
        male_count = 0
        female_count = 0
        mixed_count = 0

        for _, row in subset.iterrows():
            ages = pattern.split(" + ")
            signs: list[float] = []

            for age in ages:
                value = row[f"{age}_dPSI"]
                if pd.notna(value):
                    sign = float(np.sign(value))
                    if sign != 0:
                        signs.append(sign)

            if signs and all(sign > 0 for sign in signs):
                male_count += 1
            elif signs and all(sign < 0 for sign in signs):
                female_count += 1
            elif signs:
                mixed_count += 1

        summary_rows.append(
            {
                "Age pattern": pattern,
                "Male-higher": male_count,
                "Female-higher": female_count,
                "Mixed direction": mixed_count,
                "Total": len(subset),
            }
        )

    return pd.DataFrame(summary_rows)


def plot_single_direction_bar(
    summary: pd.DataFrame,
    value_column: str,
    title: str,
    filename: str,
    output_dir: Path,
) -> None:
    """Create one male-higher or female-higher bar graph."""
    values = summary[value_column].to_numpy()
    labels = summary["Age pattern"].tolist()

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, values)
    ax.set_title(title)
    ax.set_xlabel("Ages with sex-associated splicing")
    ax.set_ylabel("Number of candidate events")
    ax.tick_params(axis="x", rotation=30)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            str(int(value)),
            ha="center",
            va="bottom",
        )

    fig.tight_layout()
    save_figure(fig, output_dir / filename)


def plot_combined_sex_direction(summary: pd.DataFrame, output_dir: Path) -> None:
    """Create the combined stacked male/female/mixed graph."""
    labels = summary["Age pattern"].tolist()
    male = summary["Male-higher"].to_numpy()
    female = summary["Female-higher"].to_numpy()
    mixed = summary["Mixed direction"].to_numpy()
    totals = summary["Total"].to_numpy()

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(labels, male, label="Higher inclusion in males")
    ax.bar(labels, female, bottom=male, label="Higher inclusion in females")
    ax.bar(
        labels,
        mixed,
        bottom=male + female,
        label="Direction changes across ages",
    )

    ax.set_title("Sex direction and age persistence of TF–RBP candidate events")
    ax.set_xlabel("Ages with sex-associated splicing")
    ax.set_ylabel("Number of candidate events")
    ax.tick_params(axis="x", rotation=30)
    ax.legend()

    for index, total in enumerate(totals):
        ax.text(index, total, str(int(total)), ha="center", va="bottom")

    fig.tight_layout()
    save_figure(
        fig,
        output_dir / "tf_rbp_candidate_age_persistence_male_female_combined.png",
    )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_analysis(data_dir: Path, output_dir: Path) -> None:
    """Run the complete analysis and save every table and figure."""
    output_dir.mkdir(parents=True, exist_ok=True)

    tf_candidates = sorted(data_dir.glob("significant_splicing_events_tf - *.csv"))
    rbp_candidates = sorted(data_dir.glob("significant_splicing_events_rbp - *.csv"))
    tf_table_path = data_dir / "01_all_tf_splicing_events.csv"
    rbp_table_path = data_dir / "02_all_rbp_splicing_events.csv"
    tf_table_df = load_full_event_table(data_dir, "tf")
    rbp_table_df = load_full_event_table(data_dir, "rbp")

    if tf_candidates and rbp_candidates:
        # 1. Remove exact duplicate files by content.
        tf_files, tf_duplicate_report = unique_files_by_content(tf_candidates)
        rbp_files, rbp_duplicate_report = unique_files_by_content(rbp_candidates)

        duplicate_report = pd.concat(
            [
                tf_duplicate_report.assign(dataset_type="tf"),
                rbp_duplicate_report.assign(dataset_type="rbp"),
            ],
            ignore_index=True,
        )
        duplicate_report.to_csv(output_dir / "input_duplicate_report.csv", index=False)

        # 2. Load unique exact event_id sets.
        tf_sets, tf_metadata = load_event_sets(tf_files, "tf")
        rbp_sets, rbp_metadata = load_event_sets(rbp_files, "rbp")
        tf_table_df = pd.concat([pd.read_csv(path) for path in tf_files], ignore_index=True)
        rbp_table_df = pd.concat([pd.read_csv(path) for path in rbp_files], ignore_index=True)
    elif tf_table_path.exists() and rbp_table_path.exists():
        tf_sets, tf_metadata = load_event_sets_from_table(tf_table_path, "tf")
        rbp_sets, rbp_metadata = load_event_sets_from_table(rbp_table_path, "rbp")
        tf_table_df = pd.read_csv(tf_table_path)
        rbp_table_df = pd.read_csv(rbp_table_path)
    else:
        raise FileNotFoundError(
            "Could not find either the raw significant_splicing_events_*.csv files "
            "or the combined 01_all_tf_splicing_events.csv / 02_all_rbp_splicing_events.csv tables."
        )

    dataset_metadata = pd.concat([tf_metadata, rbp_metadata], ignore_index=True)
    dataset_metadata.to_csv(output_dir / "dataset_metadata.csv", index=False)

    multi_experiment = dataset_metadata[dataset_metadata["experiment_count"] > 1]
    multi_experiment.to_csv(
        output_dir / "multi_experiment_files_warning.csv",
        index=False,
    )

    # 3. TF × RBP exact-event overlap matrix.
    overlap_statistics = calculate_tf_rbp_overlap(tf_sets, rbp_sets)
    overlap_statistics = overlap_statistics.sort_values(
        ["Jaccard_similarity", "Shared_exact_events"],
        ascending=False,
    )
    overlap_statistics.to_csv(
        output_dir / "tf_rbp_overlap_statistics.csv",
        index=False,
    )
    overlap_statistics.head(15).to_csv(
        output_dir / "top_tf_regulator_exact_event_overlaps.csv",
        index=False,
    )

    count_matrix, jaccard_matrix = plot_tf_rbp_heatmap(
        overlap_statistics,
        output_dir,
    )
    count_matrix.to_csv(output_dir / "tf_rbp_shared_event_count_matrix.csv")
    jaccard_matrix.to_csv(output_dir / "tf_rbp_jaccard_matrix.csv")

    important_sites = build_jaccard_site_table(
        tf_table_df,
        rbp_table_df,
        overlap_statistics,
    )
    important_sites.to_csv(
        output_dir / "tf_rbp_jaccard_important_sites.csv",
        index=False,
    )

    # 4. Convert and load the 3d/6d/9d age/sex files.
    age_frames = load_age_files(data_dir)

    # 5. Candidate convergence events and age/sex matching.
    candidate_age_table = build_candidate_age_table(
        tf_sets,
        rbp_sets,
        age_frames,
    )
    candidate_age_table.drop(columns=["pattern_order"]).to_csv(
        output_dir / "tf_rbp_candidate_age_sex_associations.csv",
        index=False,
    )

    # 6. Age/sex dPSI heatmap and total persistence graph.
    plot_age_dpsi_heatmap(candidate_age_table, output_dir)
    persistence_counts = plot_age_persistence_summary(
        candidate_age_table,
        output_dir,
    )

    # 7. Male, female, and combined graphs.
    sex_summary = classify_sex_direction_by_pattern(candidate_age_table)
    sex_summary.to_csv(
        output_dir / "tf_rbp_candidate_age_persistence_by_sex.csv",
        index=False,
    )

    plot_single_direction_bar(
        sex_summary,
        value_column="Male-higher",
        title="Male-higher TF–RBP candidate splicing events by age pattern",
        filename="tf_rbp_candidate_age_persistence_male_higher.png",
        output_dir=output_dir,
    )
    plot_single_direction_bar(
        sex_summary,
        value_column="Female-higher",
        title="Female-higher TF–RBP candidate splicing events by age pattern",
        filename="tf_rbp_candidate_age_persistence_female_higher.png",
        output_dir=output_dir,
    )
    plot_combined_sex_direction(sex_summary, output_dir)

    # Console summary.
    print("Analysis complete.")
    print(f"TF datasets after exact duplicate removal: {len(tf_sets)}")
    print(f"RBP/regulator datasets after exact duplicate removal: {len(rbp_sets)}")
    print(
        "Candidate TF–RBP events strict at >=1 age: "
        f"{len(candidate_age_table)}"
    )
    print("\nAge-pattern counts:")
    print(persistence_counts.to_string())
    print("\nMale/female direction counts:")
    print(sex_summary.to_string(index=False))
    print(f"\nOutputs saved to: {output_dir.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate TF–RBP exact-event overlap and age/sex splicing figures."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path.cwd(),
        help="Directory containing all input CSV files (default: current directory).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for figures and output CSVs. "
            "Default: <data-dir>/tf_rbp_analysis_outputs"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else data_dir / "tf_rbp_analysis_outputs"
    )
    run_analysis(data_dir, output_dir)


if __name__ == "__main__":
    main()