#!/usr/bin/env python3
# (Script contents generated in previous cell; re-creating due to kernel reset)
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

def choose_from_list(options: List[Path], prompt: str) -> Path:
    print(f"\n{prompt}")
    for i, p in enumerate(options, 1):
        print(f"[{i}] {p.name}")
    while True:
        sel = input("Enter number: ").strip()
        if sel.isdigit() and 1 <= int(sel) <= len(options):
            return options[int(sel)-1]
        print("Invalid selection; try again.")

def choose_many(labels: List[str], prompt: str) -> List[str]:
    print(f"\n{prompt}")
    for i, s in enumerate(labels, 1):
        print(f"[{i}] {s}")
    print("Enter numbers separated by commas, or 'a' for all.")
    while True:
        sel = input("Selection: ").strip().lower()
        if sel == 'a':
            return labels
        try:
            idxs = [int(x) for x in re.split(r'\s*,\s*', sel) if x]
            if all(1 <= i <= len(labels) for i in idxs):
                return [labels[i-1] for i in idxs]
        except Exception:
            pass
        print("Invalid selection; try again.")

def resolve_data_root(script_path: Path) -> Path:
    data_root = script_path.parent.parent / "Data"
    if not data_root.exists():
        raise FileNotFoundError(f"Expected Data directory at: {data_root}")
    return data_root

def list_datasets(data_root: Path) -> List[Path]:
    return sorted([p for p in data_root.iterdir() if p.is_dir() and p.name.lower().startswith("dataset")])

def list_wilcoxon_dirs(dataset_dir: Path) -> List[Path]:
    return sorted([p for p in dataset_dir.iterdir() if p.is_dir() and p.name.lower().startswith("wilcoxon")])

def list_cue_dirs(wdir: Path) -> Tuple[Dict[str, Path], Dict[str, Path]]:
    """
    List directories matching msc_mean and msc_cat patterns with either Cue or mov prefix.
    Returns tuple of (mean_dirs dict, cat_dirs dict) keyed by frequency string.
    """
    mean_dirs, cat_dirs = {}, {}
    for p in wdir.iterdir():
        if not p.is_dir(): continue
        # Match both Cue and mov (case-insensitive) prefixes
        m_mean = re.match(r"^(Cue|mov)_msc_mean_(\d+)Hz$", p.name, re.IGNORECASE)
        m_cat  = re.match(r"^(Cue|mov)_msc_cat_(\d+)Hz$", p.name, re.IGNORECASE)
        if m_mean: mean_dirs[m_mean.group(2)] = p
        if m_cat:  cat_dirs[m_cat.group(2)]  = p
    return mean_dirs, cat_dirs

def load_numeric_array(csv_path: Path) -> np.ndarray:
    df = pd.read_csv(csv_path, index_col=0)
    if df.shape[0] == df.shape[1]:
        arr = df.to_numpy()
        r, c = np.triu_indices(arr.shape[0], k=1)
        v = arr[r, c].astype(float)
        v = v[np.isfinite(v)]
        return v
    num = df.select_dtypes(include=[np.number]).to_numpy().astype(float).ravel()
    num = num[np.isfinite(num)]
    return num

def fisher_z_ci(r: float, n_pairs: int, alpha: float = 0.05):
    if not np.isfinite(r) or n_pairs <= 3:
        return float('nan'), float('nan')
    z = np.arctanh(np.clip(r, -0.999999, 0.999999))
    se = 1.0 / np.sqrt(n_pairs - 3.0)
    zcrit = stats.norm.ppf(1 - alpha/2)
    lo = np.tanh(z - zcrit * se)
    hi = np.tanh(z + zcrit * se)
    return float(lo), float(hi)

def collect_subject_files(root_dir: Path, target_filename: str) -> Dict[str, Path]:
    mapping = {}
    for p in root_dir.rglob(target_filename):
        mapping[p.parent.name] = p
    return mapping

def create_scatterplot(mean_values: np.ndarray, cat_values: np.ndarray, 
                      rho: float, p_value: float, freq: str, 
                      out_path: Path, title_suffix: str = "") -> None:
    """
    Create a scatterplot of mean vs cat values with correlation statistics.
    
    Args:
        mean_values: Array of mean values
        cat_values: Array of cat values
        rho: Spearman correlation coefficient
        p_value: P-value of the correlation
        freq: Frequency string (e.g., "10")
        out_path: Path to save the plot
        title_suffix: Optional suffix for the title (e.g., "GROUP" or subject name)
    """
    # Ensure arrays are the same length and finite
    m = min(mean_values.size, cat_values.size)
    mean_vals = mean_values[:m]
    cat_vals = cat_values[:m]
    
    # Filter out non-finite values
    valid = np.isfinite(mean_vals) & np.isfinite(cat_vals)
    mean_vals = mean_vals[valid]
    cat_vals = cat_vals[valid]
    
    if mean_vals.size < 3:
        return  # Skip plotting if not enough data points
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create scatter plot
    ax.scatter(mean_vals, cat_vals, alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
    
    # Labels and title
    ax.set_xlabel('MSC Mean', fontsize=12, fontweight='bold')
    ax.set_ylabel('MSC Cat', fontsize=12, fontweight='bold')
    
    title = f'Correlation: MSC Mean vs MSC Cat ({freq}Hz)'
    if title_suffix:
        title += f' - {title_suffix}'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add correlation statistics as text
    stats_text = f'Spearman ρ = {rho:.4f}\np = {p_value:.4e}\nn = {mean_vals.size}'
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def compute_group_stats_from_subjects(rhos: List[float], alpha: float = 0.05) -> dict:
    arr = np.asarray([x for x in rhos if np.isfinite(x)], dtype=float)
    n = arr.size
    if n == 0:
        return {
            "n_subjects": 0, "mean_rho": float('nan'), "std_rho": float('nan'),
            "ci95_low": float('nan'), "ci95_high": float('nan'),
            "t_pvalue": float('nan'), "wilcoxon_pvalue": float('nan')
        }
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if n > 1 else float('nan')
    if n > 1 and np.isfinite(std):
        tcrit = stats.t.ppf(1 - 0.05/2, df=n-1)
        se = std / np.sqrt(n)
        lo, hi = mean - tcrit * se, mean + tcrit * se
        t_stat, t_p = stats.ttest_1samp(arr, popmean=0.0)
        try:
            _, w_p = stats.wilcoxon(arr, zero_method='wilcox')
        except Exception:
            w_p = float('nan')
    else:
        lo = hi = mean
        t_p = float('nan'); w_p = float('nan')
    return {
        "n_subjects": int(n),
        "mean_rho": mean, "std_rho": std,
        "ci95_low": lo, "ci95_high": hi,
        "t_pvalue": float(t_p), "wilcoxon_pvalue": float(w_p)
    }

def analyze_frequency(freq: str, mean_dir: Path, cat_dir: Path, out_dir: Path) -> None:
    # Try both naming conventions: "L-R" (V2 style) and "Left-Right" (V3 style)
    mean_file_lr = f"msc_mean_{freq}Hz_L-R_overall.csv"
    cat_file_lr  = f"msc_cat_{freq}Hz_L-R_overall.csv"
    mean_file_leftright = f"msc_mean_{freq}Hz_Left-Right_overall.csv"
    cat_file_leftright  = f"msc_cat_{freq}Hz_Left-Right_overall.csv"
    
    # Determine which naming convention is used
    mean_map = collect_subject_files(mean_dir, mean_file_lr)
    cat_map  = collect_subject_files(cat_dir,  cat_file_lr)
    
    # If no files found with L-R, try Left-Right
    if not mean_map and not cat_map:
        mean_map = collect_subject_files(mean_dir, mean_file_leftright)
        cat_map  = collect_subject_files(cat_dir,  cat_file_leftright)
        if mean_map or cat_map:
            mean_file = mean_file_leftright
            cat_file  = cat_file_leftright
        else:
            mean_file = None
            cat_file  = None
    else:
        mean_file = mean_file_lr
        cat_file  = cat_file_lr

    common_subjects = sorted(set(mean_map) & set(cat_map))

    if common_subjects:
        rows = []; rhos = []
        all_mean_vals = []
        all_cat_vals = []
        
        for subj in common_subjects:
            a = load_numeric_array(mean_map[subj])
            b = load_numeric_array(cat_map[subj])
            m = min(a.size, b.size)
            if m < 3: continue
            rho, p = stats.spearmanr(a[:m], b[:m], nan_policy='omit')
            rows.append({"subject": subj, "rho_spearman": float(rho), "p_value": float(p), "n_pairs": int(m)})
            if np.isfinite(rho): rhos.append(float(rho))
            
            # Collect data for combined plot
            valid = np.isfinite(a[:m]) & np.isfinite(b[:m])
            all_mean_vals.extend(a[:m][valid].tolist())
            all_cat_vals.extend(b[:m][valid].tolist())
            
            # Create individual subject scatterplot
            create_scatterplot(a[:m], b[:m], rho, p, freq, 
                             out_dir / f"scatterplot_{freq}Hz_{subj}.png", 
                             title_suffix=subj)
        
        # Create combined scatterplot with all subjects
        if all_mean_vals and all_cat_vals:
            all_mean_arr = np.array(all_mean_vals)
            all_cat_arr = np.array(all_cat_vals)
            if all_mean_arr.size >= 3:
                rho_combined, p_combined = stats.spearmanr(all_mean_arr, all_cat_arr, nan_policy='omit')
                create_scatterplot(all_mean_arr, all_cat_arr, rho_combined, p_combined, freq,
                                 out_dir / f"scatterplot_{freq}Hz_ALL_SUBJECTS.png",
                                 title_suffix="ALL SUBJECTS")
        
        pd.DataFrame(rows).sort_values("subject").to_csv(out_dir / f"individual_correlations_{freq}Hz.csv", index=False)
        stats_dict = compute_group_stats_from_subjects(rhos)
        pd.DataFrame([{
            "frequency_Hz": int(freq), **stats_dict,
            "spearman_edgewise": "NA", "spearman_pvalue": "NA", "n_pairs": "NA",
        }]).to_csv(out_dir / f"group_stats_{freq}Hz.csv", index=False)
        return

    # If no subject files, check root files with both naming conventions
    if mean_file is None:
        # Try to find which naming convention is used
        root_mean_lr = mean_dir / mean_file_lr
        root_cat_lr  = cat_dir  / cat_file_lr
        root_mean_leftright = mean_dir / mean_file_leftright
        root_cat_leftright  = cat_dir  / cat_file_leftright
        
        if root_mean_lr.exists() and root_cat_lr.exists():
            root_mean = root_mean_lr
            root_cat  = root_cat_lr
        elif root_mean_leftright.exists() and root_cat_leftright.exists():
            root_mean = root_mean_leftright
            root_cat  = root_cat_leftright
        else:
            raise FileNotFoundError(
                f"Expected files matching either '{mean_file_lr}' or '{mean_file_leftright}' "
                f"and '{cat_file_lr}' or '{cat_file_leftright}' in their respective folders."
            )
    else:
        root_mean = mean_dir / mean_file
        root_cat  = cat_dir  / cat_file
        if not root_mean.exists() or not root_cat.exists():
            raise FileNotFoundError(f"Expected {root_mean.name} and {root_cat.name} in their respective folders.")

    a = load_numeric_array(root_mean)
    b = load_numeric_array(root_cat)
    m = min(a.size, b.size)
    if m < 3:
        raise ValueError(f"Not enough paired elements at {freq}Hz (found {m}).")

    rho, p = stats.spearmanr(a[:m], b[:m], nan_policy='omit')
    lo, hi = fisher_z_ci(rho, n_pairs=m)

    # Create scatterplot for group-level analysis
    create_scatterplot(a[:m], b[:m], rho, p, freq,
                     out_dir / f"scatterplot_{freq}Hz_GROUP.png",
                     title_suffix="GROUP")

    pd.DataFrame([{
        "subject": "GROUP", "rho_spearman": float(rho), "p_value": float(p), "n_pairs": int(m)
    }]).to_csv(out_dir / f"individual_correlations_{freq}Hz.csv", index=False)

    pd.DataFrame([{
        "frequency_Hz": int(freq),
        "n_subjects": "NA", "mean_rho": "NA", "std_rho": "NA",
        "ci95_low": "NA", "ci95_high": "NA", "t_pvalue": "NA", "wilcoxon_pvalue": "NA",
        "spearman_edgewise": float(rho), "spearman_pvalue": float(p),
        "n_pairs": int(m), "spearman_ci95_low": float(lo), "spearman_ci95_high": float(hi),
    }]).to_csv(out_dir / f"group_stats_{freq}Hz.csv", index=False)

def main(argv=None) -> int:
    script_path = Path(__file__).resolve()
    data_root = resolve_data_root(script_path)
    print(f"Data root: {data_root}")

    datasets = list_datasets(data_root)
    if not datasets:
        print("No DataSet* folders found under Data."); return 2
    dataset = choose_from_list(datasets, "Select a DataSet to analyze:")

    wdirs = list_wilcoxon_dirs(dataset)
    if not wdirs:
        print("No Wilcoxon* folders inside the selected DataSet."); return 3
    wilcoxon_dir = choose_from_list(wdirs, f"Select a Wilcoxon folder inside {dataset.name}:")

    mean_dirs, cat_dirs = list_cue_dirs(wilcoxon_dir)
    common = sorted(set(mean_dirs) & set(cat_dirs), key=lambda x: int(x))
    if not common:
        print("No common frequencies with both mean and cat."); return 4

    selected = choose_many([f"{f}Hz" for f in common], "Select frequency(ies):")
    selected = [s.replace("Hz","") for s in selected]

    out_dir = wilcoxon_dir / "correlation_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    for f in selected:
        print(f"\n--- Frequency {f}Hz ---")
        analyze_frequency(f, mean_dirs[f], cat_dirs[f], out_dir)
        print("Saved outputs to:", out_dir)

    print("\nDone."); return 0

if __name__ == "__main__":
    raise SystemExit(main())
