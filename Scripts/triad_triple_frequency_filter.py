#!/usr/bin/env python3
"""
Triad Frequency Connection Filter
---------------------------------

Purpose
~~~~~~~
Given connectivity CSV files (with columns: Method, Frequency, Region_From, Region_To, W_Stat, P_value),
retain only those rows where a given (Region_From, Region_To) connection exists at a triad of
frequencies +/-1 Hz around a center frequency (e.g., 9/10/11, 19/20/21, 24/25/26, 99/100/101).

By default, this script searches for CSVs in ../Results relative to the script's location and writes
filtered files next to each input as "<stem>_triad_frequencies.csv". You may also provide an explicit
--input path to process one file.

Usage
~~~~~
# Process all CSVs in ../Results:
python triad_triple_frequency_filter.py

# Process a single file:
python triad_triple_frequency_filter.py --input ../Results/myfile.csv

# Specify an alternative output directory:
python triad_triple_frequency_filter.py --output-dir ../Results/Filtered

Notes
~~~~~
- "Frequency" values are strings like '10Hz' (case/space-insensitive handling included).
- Triad centers are detected automatically by checking whether f-1 and f+1 exist in the dataset's
  frequency set; you can override with --centers 10 20 25 100 if desired.
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Iterable, List, Set, Tuple
import pandas as pd


REQUIRED_COLUMNS = ["Method", "Frequency", "Region_From", "Region_To"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter connections that exist at ±1 Hz around center frequencies (triads)."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Optional path to a single CSV. If omitted, all CSVs in ../Results are processed.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory. Default: same directory as the input file(s).",
    )
    parser.add_argument(
        "--centers",
        type=int,
        nargs="*",
        default=None,
        help="Optional list of center frequencies (e.g., 10 20 25 100). If omitted, centers are inferred.",
    )
    return parser.parse_args()


def infer_centers(all_freq_labels: Iterable[str]) -> List[int]:
    """
    Infer centers from the set of available frequency labels like {'9Hz','10Hz','11Hz',...}.
    A center c is included if labels for c-1, c, and c+1 all exist.
    """
    # Normalize: keep only labels ending with 'Hz' and pull integer part safely
    norm = set(str(f).strip() for f in all_freq_labels if isinstance(f, str))
    numbers: Set[int] = set()
    for lab in norm:
        if lab.lower().endswith("hz"):
            try:
                numbers.add(int(lab[:-2]))
            except ValueError:
                pass

    centers: List[int] = []
    for c in sorted(numbers):
        if c - 1 in numbers and c + 1 in numbers:
            centers.append(c)
    return centers


def validate_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input is missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )


def filter_triad_rows(df: pd.DataFrame, centers: List[int]) -> pd.DataFrame:
    # Normalize key fields
    df = df.copy()
    df["Frequency"] = df["Frequency"].astype(str).str.strip()
    df["Region_From"] = df["Region_From"].astype(str).str.strip()
    df["Region_To"] = df["Region_To"].astype(str).str.strip()

    # Build (from,to) -> set of freq labels
    pair_to_freqs = (
        df.groupby(["Region_From", "Region_To"])["Frequency"]
        .apply(lambda s: set(s.tolist()))
        .to_dict()
    )

    def has_triad(freq_set: Set[str], center: int) -> bool:
        triad = {f"{center-1}Hz", f"{center}Hz", f"{center+1}Hz"}
        return triad.issubset(freq_set)

    # Determine eligible pairs
    eligible_pairs: Set[Tuple[str, str]] = set()
    for pair, fset in pair_to_freqs.items():
        for c in centers:
            if has_triad(fset, c):
                eligible_pairs.add(pair)
                break  # pair is eligible if it matches ANY center

    if not eligible_pairs:
        return df.iloc[0:0].copy()  # empty with same columns

    # Keep only rows that belong to eligible pairs AND whose frequency is in any triad window
    triad_freq_labels = set()
    for c in centers:
        triad_freq_labels.update({f"{c-1}Hz", f"{c}Hz", f"{c+1}Hz"})

    mask = (
        df.apply(lambda r: (r["Region_From"], r["Region_To"]) in eligible_pairs, axis=1)
        & df["Frequency"].isin(triad_freq_labels)
    )
    return df.loc[mask].copy()


def process_file(in_file: Path, out_dir: Path | None, centers: List[int] | None) -> Path:
    df = pd.read_csv(in_file)
    validate_columns(df)

    # Determine centers if not provided
    if centers is None:
        centers = infer_centers(df["Frequency"].unique())
        if not centers:
            raise ValueError(
                "Could not infer any triad centers from Frequency column. "
                "Provide them explicitly with --centers."
            )

    filtered = filter_triad_rows(df, centers)

    # Resolve output path
    if out_dir is None:
        out_dir = in_file.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{in_file.stem}_triad_frequencies.csv"
    filtered.to_csv(out_path, index=False)
    return out_path


def find_default_inputs(script_path: Path) -> List[Path]:
    # Default Results directory is sibling of Scripts directory
    # e.g., <project>/Scripts/this_script.py and <project>/Results/*.csv
    scripts_dir = script_path.parent
    results_dir = scripts_dir.parent / "Results"
    return sorted(results_dir.glob("*.csv"))


def main():
    args = parse_args()
    script_path = Path(__file__).resolve()

    inputs: List[Path]
    if args.input is not None:
        inputs = [args.input]
    else:
        inputs = find_default_inputs(script_path)

    if not inputs:
        raise SystemExit("No input CSVs found. Provide --input or place CSVs in ../Results.")

    # Process each file
    for in_file in inputs:
        try:
            out_path = process_file(in_file, args.output_dir, args.centers)
            print(f"[OK] {in_file} -> {out_path}")
        except Exception as e:
            print(f"[ERROR] {in_file}: {e}")


if __name__ == "__main__":
    main()
