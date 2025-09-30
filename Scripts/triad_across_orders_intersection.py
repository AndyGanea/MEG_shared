#!/usr/bin/env python3
"""
Intersect Triad Connections Across Orders
-----------------------------------------

Goal
~~~~
Given three CSV files that already contain triad-filtered connections for different model orders
(e.g., Order5, Order10, Order20), write a new CSV containing only those connections that
**exist in all three files**.

- A "connection" is defined as the triple (Region_From, Region_To, Frequency).
- Only rows where this triple appears in **all three** input CSVs are written.
- The output keeps the same columns as the input. If values (e.g., W_Stat, P_value) differ
  across orders for the same connection, the values from the **Order10** file are written.

File Discovery
~~~~~~~~~~~~~~
By default, this script scans ../Results for any CSV file whose name contains "Order10" and
ends with "_triad_frequencies.csv". For each such file, it will look for the corresponding
Order5 and Order20 files by replacing the "Order10" token in the filename. If all three files
exist, they are processed together.

Output Naming
~~~~~~~~~~~~~
For each detected trio, a single output CSV is written in the same directory as the inputs,
with the filename formed by replacing "Order10" with **"OrderCombined"** in the Order10 filename.

Examples
~~~~~~~~
# Process all discovered trios (../Results relative to this script):
python triad_across_orders_intersection.py

# Process a specific trio by template (use {order} placeholder):
python triad_across_orders_intersection.py \
  --template "../Results/cuealign_Order{order}_movement_wilcoxon_summary_gPDC_20250919_172046_triad_frequencies.csv"

# Process with custom orders:
python triad_across_orders_intersection.py \
  --template "../Results/cuealign_Order{order}_movement_wilcoxon_summary_gPDC_20250919_172046_triad_frequencies.csv" \
  --orders 5 10 20
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple, Set
import pandas as pd
import sys


REQUIRED_COLUMNS = ["Method", "Frequency", "Region_From", "Region_To"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Intersect triad connections across multiple Order files.")
    p.add_argument(
        "--template",
        type=str,
        default=None,
        help=(
            "Optional path template with {order} placeholder. Example: "
            "'../Results/cuealign_Order{order}_movement_wilcoxon_summary_gPDC_20250919_172046_triad_frequencies.csv'."
        ),
    )
    p.add_argument(
        "--orders",
        type=int,
        nargs="+",
        default=[5, 10, 20],
        help="Orders to intersect across (default: 5 10 20).",
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="If no template is given, scan this directory (default: ../Results relative to script).",
    )
    return p.parse_args()


def validate_columns(df: pd.DataFrame, src: Path) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{src} is missing required columns: {missing}")


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    validate_columns(df, path)
    # Normalize key fields
    df["Frequency"] = df["Frequency"].astype(str).str.strip()
    df["Region_From"] = df["Region_From"].astype(str).str.strip()
    df["Region_To"] = df["Region_To"].astype(str).str.strip()
    return df


def connection_keys(df: pd.DataFrame) -> Set[Tuple[str, str, str]]:
    return set(df[["Region_From", "Region_To", "Frequency"]].itertuples(index=False, name=None))


def compute_intersection(dfs: Sequence[pd.DataFrame]) -> Set[Tuple[str, str, str]]:
    key_sets = [connection_keys(d) for d in dfs]
    if not key_sets:
        return set()
    inter = key_sets[0]
    for ks in key_sets[1:]:
        inter = inter & ks
    return inter


def select_rows_by_keys(df: pd.DataFrame, keys: Set[Tuple[str, str, str]]) -> pd.DataFrame:
    mask = df.apply(lambda r: (r["Region_From"], r["Region_To"], r["Frequency"]) in keys, axis=1)
    return df.loc[mask].copy()


def infer_trios(results_dir: Path, orders: Sequence[int]) -> List[Tuple[Path, List[Path]]]:
    """
    Discover groups by finding 'Order10' files and swapping the token for other orders.
    Returns list of (order10_path, [paths for orders]) where all exist.
    """
    trios = []
    for p in sorted(results_dir.glob("*.csv")):
        name = p.name
        if "Order10" not in name or not name.endswith("_triad_frequencies.csv"):
            continue
        paths = []
        ok = True
        for o in orders:
            alt_name = name.replace("Order10", f"Order{o}")
            alt_path = p.parent / alt_name
            if not alt_path.exists():
                ok = False
                break
            paths.append(alt_path)
        if ok:
            trios.append((p.parent / name.replace("Order10", "Order10"), paths))
    return trios


def build_output_path(order10_path: Path) -> Path:
    # Replace "Order10" with "OrderCombined" and keep the rest intact
    out_name = order10_path.name.replace("Order10", "OrderCombined")
    return order10_path.parent / out_name


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent

    if args.template:
        # Template mode
        trio_paths = []
        for o in args.orders:
            trio_paths.append(Path(args.template.format(order=o)))
        for p in trio_paths:
            if not p.exists():
                sys.exit(f"Missing expected file: {p}")
        # Use Order10 as the representative file for values
        try:
            idx10 = args.orders.index(10)
        except ValueError:
            sys.exit("Order 10 must be included in --orders to serve as the representative dataset.")
        order10_path = trio_paths[idx10]
        out_path = build_output_path(order10_path)

        dfs = [load_csv(p) for p in trio_paths]
        inter_keys = compute_intersection(dfs)
        if not inter_keys:
            print("No common connections across all provided orders.")
            # Write an empty CSV with headers to be explicit
            empty = dfs[idx10].iloc[0:0].copy()
            empty.to_csv(out_path, index=False)
            print(f"[OK] Wrote empty intersection to {out_path}")
            return

        # Keep rows from Order10 only for the intersected keys
        out_df = select_rows_by_keys(dfs[idx10], inter_keys)
        out_df.to_csv(out_path, index=False)
        print(f"[OK] {len(out_df)} rows -> {out_path}")
        return

    # Discovery mode
    results_dir = args.results_dir or (script_dir.parent / "Results")
    if not results_dir.exists():
        sys.exit(f"Results directory not found: {results_dir}")

    trios = infer_trios(results_dir, args.orders)
    if not trios:
        sys.exit("No valid Order10-based trios found. Provide --template or verify filenames.")

    for order10_path, all_paths in trios:
        try:
            # Determine index of Order10 within the orders list
            try:
                idx10 = args.orders.index(10)
            except ValueError:
                print("[SKIP] Orders do not include 10; cannot choose representative dataset.")
                continue

            dfs = [load_csv(p) for p in all_paths]
            inter_keys = compute_intersection(dfs)

            out_path = build_output_path(order10_path)
            if not inter_keys:
                # Write empty with headers from Order10 to be explicit
                empty = dfs[idx10].iloc[0:0].copy()
                empty.to_csv(out_path, index=False)
                print(f"[OK] {order10_path.name}: no common connections; wrote empty -> {out_path.name}")
                continue

            out_df = select_rows_by_keys(dfs[idx10], inter_keys)
            out_df.to_csv(out_path, index=False)
            print(f"[OK] {order10_path.name}: {len(out_df)} rows -> {out_path.name}")
        except Exception as e:
            print(f"[ERROR] {order10_path.name}: {e}")


if __name__ == "__main__":
    main()
