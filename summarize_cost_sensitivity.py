#!/usr/bin/env python3

"""
python3 summarize_cost_sensitivity.py \
  --low_dir out_cost_low/results \
  --base_dir out_cost_base/results \
  --high_dir out_cost_high/results \
  --out_csv cost_sensitivity_summary.csv
"""

import argparse
import os
import re
from typing import Dict, List, Tuple

import pandas as pd


def read_final_metrics(results_dir: str) -> pd.DataFrame:
    rows = []

    for fname in os.listdir(results_dir):
        if not fname.endswith(".csv"):
            continue

        metric = None
        if fname.endswith("_utility.csv"):
            metric = "utility"
            user_type = fname[:-len("_utility.csv")]
        elif fname.endswith("_cumregret.csv"):
            metric = "cumregret"
            user_type = fname[:-len("_cumregret.csv")]
        else:
            continue

        path = os.path.join(results_dir, fname)
        df = pd.read_csv(path)

        last = df.iloc[-1].to_dict()
        row = {"user_type": user_type, "metric": metric}

        for col, val in last.items():
            if col == "t":
                row[col] = val
            else:
                row[col] = val

        rows.append(row)

    if not rows:
        raise RuntimeError(f"No matching CSV files found in {results_dir}")

    long_df = pd.DataFrame(rows)

    merged_rows = []
    for user_type in sorted(long_df["user_type"].unique()):
        util = long_df[(long_df["user_type"] == user_type) & (long_df["metric"] == "utility")]
        reg = long_df[(long_df["user_type"] == user_type) & (long_df["metric"] == "cumregret")]

        if util.empty or reg.empty:
            continue

        util_row = util.iloc[0].to_dict()
        reg_row = reg.iloc[0].to_dict()

        merged = {"user_type": user_type, "t_final": util_row.get("t", None)}

        for k, v in util_row.items():
            if k not in {"user_type", "metric", "t"}:
                merged[f"{k}_utility"] = v

        for k, v in reg_row.items():
            if k not in {"user_type", "metric", "t"}:
                merged[f"{k}_cumregret"] = v

        merged_rows.append(merged)

    return pd.DataFrame(merged_rows)


def find_policy_mean_cols(columns: List[str], suffix: str) -> Dict[str, str]:
    out = {}
    pattern = re.compile(r"^(.*)_mean_" + re.escape(suffix) + r"$")
    # fallback for existing naming style: policy_mean_utility or policy_mean_cumregret
    # if your CSV names are like TS_mean, AlwaysExplain_mean, then adjust below
    for c in columns:
        m = pattern.match(c)
        if m:
            out[m.group(1)] = c

    if out:
        return out

    # fallback to current save_csv_timeseries style: <policy>_mean
    # after merge they become <policy>_mean_utility or <policy>_mean_cumregret
    pattern2 = re.compile(r"^(.*)_mean_" + re.escape(suffix) + r"$")
    for c in columns:
        m = pattern2.match(c)
        if m:
            out[m.group(1)] = c
    return out


def extract_policy_columns(df: pd.DataFrame, metric_suffix: str) -> Dict[str, str]:
    cols = {}
    for c in df.columns:
        if c.endswith(f"_mean_{metric_suffix}"):
            pol = c[: -len(f"_mean_{metric_suffix}")]
            cols[pol] = c
    return cols


def summarize_run(label: str, results_dir: str) -> pd.DataFrame:
    df = read_final_metrics(results_dir)

    util_cols = extract_policy_columns(df, "utility")
    reg_cols = extract_policy_columns(df, "cumregret")

    if not util_cols or not reg_cols:
        raise RuntimeError(
            f"Could not infer policy columns in {results_dir}. "
            f"Found columns: {list(df.columns)}"
        )

    # prefer these names if present
    ts_candidates = [
        p for p in util_cols
        if "adaptive" in p.lower() or "ts" in p.lower() or "thompson" in p.lower()
    ]
    ts_policy = ts_candidates[0] if ts_candidates else None

    if ts_policy is None:
        raise RuntimeError(
            f"Could not identify the adaptive policy from columns: {list(util_cols.keys())}"
        )

    fixed_policies = [p for p in util_cols if p != ts_policy]

    rows = []
    for _, row in df.iterrows():
        user_type = row["user_type"]

        best_fixed = max(fixed_policies, key=lambda p: row[util_cols[p]])
        best_fixed_utility = row[util_cols[best_fixed]]
        ts_utility = row[util_cols[ts_policy]]

        best_fixed_regret = row[reg_cols[best_fixed]]
        ts_regret = row[reg_cols[ts_policy]]

        rows.append({
            "cost_setting": label,
            "user_type": user_type,
            "adaptive_policy": ts_policy,
            "adaptive_final_utility": ts_utility,
            "best_fixed_policy": best_fixed,
            "best_fixed_final_utility": best_fixed_utility,
            "utility_diff_adaptive_minus_best_fixed": ts_utility - best_fixed_utility,
            "adaptive_final_cumregret": ts_regret,
            "best_fixed_final_cumregret": best_fixed_regret,
            "cumregret_diff_adaptive_minus_best_fixed": ts_regret - best_fixed_regret,
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", required=True, help="path to baseline run results dir")
    parser.add_argument("--low_dir", required=True, help="path to low-cost run results dir")
    parser.add_argument("--high_dir", required=True, help="path to high-cost run results dir")
    parser.add_argument("--out_csv", default="cost_sensitivity_summary.csv")
    args = parser.parse_args()

    dfs = [
        summarize_run("low", args.low_dir),
        summarize_run("base", args.base_dir),
        summarize_run("high", args.high_dir),
    ]
    out = pd.concat(dfs, ignore_index=True)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 50)
    pd.set_option("display.float_format", lambda x: f"{x:.3f}")

    print("\n=== Cost sensitivity summary ===\n")
    print(out.sort_values(["user_type", "cost_setting"]).to_string(index=False))

    out.to_csv(args.out_csv, index=False)
    print(f"\nSaved summary to: {args.out_csv}")


if __name__ == "__main__":
    main()