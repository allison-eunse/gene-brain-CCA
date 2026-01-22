#!/usr/bin/env python3
"""
Phase 3 Results Gatherer - Auto-collect and summarize benchmark results.

Per lab guidelines (Tip 3 - The "Sweeper" Script):
- Iterates through results directory
- Finds experiments where results.json exists (FINISHED status)
- Creates Pandas DataFrame
- Saves as CSV and Markdown for easy reporting

Usage:
    python scripts/gather_phase3_results.py
    python scripts/gather_phase3_results.py --output-dir gene-brain-cca-2/derived/phase3_results
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


def gather_experiments(root_dir: Path | str) -> pd.DataFrame:
    """
    Collect all completed experiment results from the output directory.
    
    Args:
        root_dir: Directory containing *_results.json files
    
    Returns:
        DataFrame with all experiment results
    """
    root_dir = Path(root_dir)
    experiments = []
    
    # Find all result files
    result_files = list(root_dir.glob("*_results.json"))
    print(f"Found {len(result_files)} result files in {root_dir}")
    
    for file_path in sorted(result_files):
        try:
            data = json.loads(file_path.read_text())
            
            # Only include finished experiments
            if data.get("status") != "FINISHED":
                print(f"  [skip] {file_path.name} - status: {data.get('status', 'unknown')}")
                continue
            
            # Add file info
            data["result_file"] = file_path.name
            experiments.append(data)
            print(f"  [ok] {file_path.name}")
            
        except json.JSONDecodeError as e:
            print(f"  [error] {file_path.name} - invalid JSON: {e}")
        except Exception as e:
            print(f"  [error] {file_path.name} - {e}")
    
    if not experiments:
        print("\nNo completed experiments found.")
        return pd.DataFrame()
    
    # Create DataFrame
    df = pd.DataFrame(experiments)
    
    # Reorder columns for readability
    priority_cols = [
        "fm_model", "modality", 
        "mdd_holdout_cc1", "ctrl_holdout_cc1", "r_diff",
        "gene_cosine_mdd_ctrl", "brain_cosine_mdd_ctrl",
        "mdd_best_config", "ctrl_best_config",
        "mdd_n", "ctrl_n",
    ]
    
    existing_priority = [c for c in priority_cols if c in df.columns]
    other_cols = [c for c in df.columns if c not in priority_cols]
    df = df[existing_priority + other_cols]
    
    return df


def generate_summary_table(df: pd.DataFrame) -> str:
    """Generate a formatted summary table."""
    if df.empty:
        return "No completed experiments."
    
    # Select key columns for summary
    summary_cols = [
        "fm_model", "modality",
        "mdd_holdout_cc1", "ctrl_holdout_cc1", "r_diff",
        "gene_cosine_mdd_ctrl", "brain_cosine_mdd_ctrl",
    ]
    
    existing_cols = [c for c in summary_cols if c in df.columns]
    summary = df[existing_cols].copy()
    
    # Round numeric columns
    for col in summary.columns:
        if summary[col].dtype in ['float64', 'float32']:
            summary[col] = summary[col].round(4)
    
    # Sort by r_diff (MDD - Control)
    if "r_diff" in summary.columns:
        summary = summary.sort_values("r_diff", ascending=False)
    
    return summary.to_markdown(index=False)


def main():
    ap = argparse.ArgumentParser(description="Gather Phase 3 benchmark results")
    ap.add_argument("--output-dir", default="gene-brain-cca-2/derived/phase3_results",
                    help="Directory containing result files")
    args = ap.parse_args()
    
    root_dir = Path(args.output_dir)
    
    print("=" * 60)
    print("Phase 3 Results Gatherer")
    print("=" * 60)
    print(f"Scanning: {root_dir}")
    print()
    
    # Gather experiments
    df = gather_experiments(root_dir)
    
    if df.empty:
        print("\nNo results to summarize.")
        return
    
    print(f"\nCollected {len(df)} experiments.")
    
    # Save CSV
    csv_path = root_dir / "summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"[save] {csv_path}")
    
    # Save Markdown
    md_path = root_dir / "summary.md"
    
    md_content = f"""# Phase 3 CCA Benchmark Results

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary

{generate_summary_table(df)}

## Interpretation

- **mdd_holdout_cc1**: Correlation between gene and brain canonical variates in MDD patients (holdout set)
- **ctrl_holdout_cc1**: Same for healthy controls
- **r_diff**: MDD - Control difference (positive = stronger coupling in MDD)
- **gene_cosine_mdd_ctrl**: Similarity of gene weights between groups (1 = identical, 0 = orthogonal)
- **brain_cosine_mdd_ctrl**: Similarity of brain weights between groups

## Best Configurations

"""
    
    for _, row in df.iterrows():
        md_content += f"### {row.get('fm_model', 'unknown')} + {row.get('modality', 'unknown')}\n"
        md_content += f"- MDD best: `{row.get('mdd_best_config', 'N/A')}`\n"
        md_content += f"- Control best: `{row.get('ctrl_best_config', 'N/A')}`\n"
        md_content += f"- MDD holdout r: {row.get('mdd_holdout_cc1', 0):.4f}\n"
        md_content += f"- Control holdout r: {row.get('ctrl_holdout_cc1', 0):.4f}\n"
        md_content += "\n"
    
    md_path.write_text(md_content)
    print(f"[save] {md_path}")
    
    # Print summary to console
    print("\n" + "=" * 60)
    print("Summary Table")
    print("=" * 60)
    print(generate_summary_table(df))
    
    # Highlight best result
    if "r_diff" in df.columns and len(df) > 0:
        best_idx = df["mdd_holdout_cc1"].abs().idxmax()
        best = df.loc[best_idx]
        print("\n" + "=" * 60)
        print("Best MDD Coupling")
        print("=" * 60)
        print(f"FM Model:    {best.get('fm_model', 'N/A')}")
        print(f"Modality:    {best.get('modality', 'N/A')}")
        print(f"MDD r:       {best.get('mdd_holdout_cc1', 0):.4f}")
        print(f"Control r:   {best.get('ctrl_holdout_cc1', 0):.4f}")
        print(f"Difference:  {best.get('r_diff', 0):.4f}")


if __name__ == "__main__":
    main()
