#!/usr/bin/env python3
"""
ICSR 2026 illustrative simulation: explanation strategy selection under uncertain human response.

- Situations: Delay, Failure, Reordering
- Actions: Explain, Update, Silent
- Human response: y ~ Bernoulli(theta[s,a])
- Belief: Beta(alpha, beta) per (s,a)
- Adaptive policy: Thompson sampling

Outputs:
- results/*.csv (per user type, per policy aggregated over runs)
- figures/*.pdf

Baseline (Types 1-3 only):
python3 simulate_icsr.py --outdir out_icsr_static --T 300 --runs 30 --seed 42

Dynamic preference regime (add Type 4):
python3 simulate_icsr.py \
  --outdir out_icsr_dynamic \
  --T 300 --runs 30 --seed 42 \
  --switch_prob 0.03 \
  --switch_mode toggle \
  --dynamic_pair Type1_ExplanationOriented+Type2_Minimalist


--- How to run the sensitivity analysis ---
- Baseline:  
python3 simulate_icsr_2.py --outdir out_cost_base --T 300 --runs 30 --seed 42 \
  --cost_explain 0.15 --cost_update 0.05 --cost_silent 0.00

- How to run the sensitivity analysis:
python3 simulate_icsr_2.py --outdir out_cost_low --T 300 --runs 30 --seed 42 \
  --cost_explain 0.10 --cost_update 0.05 --cost_silent 0.00

Higher explanation cost:
python3 simulate_icsr_2.py --outdir out_cost_high --T 300 --runs 30 --seed 42 \
  --cost_explain 0.25 --cost_update 0.05 --cost_silent 0.00

- Dynamic preference regime:
python3 simulate_icsr_2.py \
  --outdir out_cost_dynamic \
  --T 300 --runs 30 --seed 42 \
  --switch_prob 0.03 \
  --switch_mode toggle \
  --dynamic_pair Type1_ExplanationOriented+Type2_Minimalist \
  --cost_explain 0.25 \
  --cost_update 0.05 \
  --cost_silent 0.00
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# -----------------------------
# Domain definitions
# -----------------------------

SITUATIONS = ["Delay", "Failure", "Reordering"]
ACTIONS = ["Explain", "Update", "Silent"]

S_IDX = {s: i for i, s in enumerate(SITUATIONS)}
A_IDX = {a: i for i, a in enumerate(ACTIONS)}

# Communication costs
# ACTION_COST = np.array([0.15, 0.05, 0.00], dtype=float)  # Explain, Update, Silent
# Default communication costs
DEFAULT_ACTION_COST = np.array([0.15, 0.05, 0.00], dtype=float)  # Explain, Update, Silent
ACTION_COST = DEFAULT_ACTION_COST.copy()

POLICY_DISPLAY = {
    "always_explain": "Always Explain",
    "always_update": "Always Update",
    "always_silent": "Always Silent",
    "random": "Random",
    "adaptive_thompson": "Adaptive (TS)",
}

from dataclasses import dataclass
from typing import Optional

@dataclass
class UserType:
    name: str
    theta: np.ndarray  # default / stationary theta, shape (|S|, |A|)

    # Optional: non-stationary user regime
    thetas: Optional[List[np.ndarray]] = None  # list of theta matrices
    switch_prob: float = 0.0                  # probability to switch per step
    switch_mode: str = "toggle"               # "toggle" or "random"


def make_user_types() -> List[UserType]:
    # User Type 1 — Explanation-oriented
    theta1 = np.array(
        [
            [0.85, 0.65, 0.30],  # Delay
            [0.90, 0.60, 0.25],  # Failure
            [0.80, 0.60, 0.40],  # Reordering
        ],
        dtype=float,
    )

    # User Type 2 — Minimalist
    theta2 = np.array(
        [
            [0.60, 0.80, 0.70],  # Delay
            [0.65, 0.85, 0.60],  # Failure
            [0.55, 0.75, 0.70],  # Reordering
        ],
        dtype=float,
    )

    # User Type 3 — Situation-sensitive
    theta3 = np.array(
        [
            [0.70, 0.85, 0.50],  # Delay
            [0.92, 0.55, 0.20],  # Failure
            [0.60, 0.65, 0.75],  # Reordering
        ],
        dtype=float,
    )

    return [
        UserType("Type1_ExplanationOriented", theta1),
        UserType("Type2_Minimalist", theta2),
        UserType("Type3_SituationSensitive", theta3),
    ]


# -----------------------------
# Policies
# -----------------------------

def select_action_fixed(policy: str) -> int:
    if policy == "always_explain":
        return A_IDX["Explain"]
    if policy == "always_update":
        return A_IDX["Update"]
    if policy == "always_silent":
        return A_IDX["Silent"]
    raise ValueError(f"Unknown fixed policy: {policy}")


def select_action_random(rng: np.random.Generator) -> int:
    return int(rng.integers(low=0, high=len(ACTIONS)))


def select_action_thompson(
    rng: np.random.Generator,
    s_idx: int,
    alpha: np.ndarray,
    beta: np.ndarray,
) -> int:
    """
    Thompson sampling:
    sample theta_tilde[s,a] ~ Beta(alpha[s,a], beta[s,a])
    choose action that maximizes expected utility: theta_tilde - cost(a)
    """
    theta_tilde = rng.beta(alpha[s_idx, :], beta[s_idx, :])
    expected_u = theta_tilde - ACTION_COST
    return int(np.argmax(expected_u))


# -----------------------------
# Simulation core
# -----------------------------

@dataclass
class RunTrace:
    utility: np.ndarray          # (T,)
    regret: np.ndarray           # (T,)
    situation_idx: np.ndarray    # (T,)
    action_idx: np.ndarray       # (T,)
    y: np.ndarray                # (T,)
    alpha_hist: np.ndarray       # (T, |S|, |A|) optional
    beta_hist: np.ndarray        # (T, |S|, |A|) optional


def step_utility(y: int, action_idx: int) -> float:
    # U = y - cost(action)
    return float(y) - float(ACTION_COST[action_idx])


def optimal_expected_utility(theta_true_row: np.ndarray) -> float:
    # max_a (theta[s,a] - cost(a))
    return float(np.max(theta_true_row - ACTION_COST))


def expected_utility_under_true(theta_true: float, action_idx: int) -> float:
    return float(theta_true) - float(ACTION_COST[action_idx])


def get_active_theta(
    rng: np.random.Generator,
    user: UserType,
    active_idx: int,
) -> Tuple[np.ndarray, int]:
    """
    Returns the currently active theta matrix and updated active_idx.
    If user.thetas is provided and switch_prob > 0, a switch may occur.
    """
    if user.thetas is None or user.switch_prob <= 0.0:
        return user.theta, active_idx

    if rng.random() < user.switch_prob:
        if user.switch_mode == "toggle" and len(user.thetas) == 2:
            active_idx = 1 - active_idx
        else:
            active_idx = int(rng.integers(low=0, high=len(user.thetas)))

    return user.thetas[active_idx], active_idx


def simulate_one_run(
    rng: np.random.Generator,
    user: UserType,
    policy: str,
    T: int,
    log_beliefs: bool = False,
) -> RunTrace:
    # Beta priors
    alpha = np.ones((len(SITUATIONS), len(ACTIONS)), dtype=float)
    beta = np.ones((len(SITUATIONS), len(ACTIONS)), dtype=float)

    utility = np.zeros(T, dtype=float)
    regret = np.zeros(T, dtype=float)
    situation_idx = np.zeros(T, dtype=int)
    action_idx = np.zeros(T, dtype=int)
    y_arr = np.zeros(T, dtype=int)

    if log_beliefs:
        alpha_hist = np.zeros((T, len(SITUATIONS), len(ACTIONS)), dtype=float)
        beta_hist = np.zeros((T, len(SITUATIONS), len(ACTIONS)), dtype=float)
    else:
        alpha_hist = np.zeros((0, 0, 0), dtype=float)
        beta_hist = np.zeros((0, 0, 0), dtype=float)

    active_idx = 0

    for t in range(T):
        s = int(rng.integers(low=0, high=len(SITUATIONS)))  # uniform situations
        situation_idx[t] = s

        # choose action
        if policy in {"always_explain", "always_update", "always_silent"}:
            a = select_action_fixed(policy)
        elif policy == "random":
            a = select_action_random(rng)
        elif policy == "adaptive_thompson":
            a = select_action_thompson(rng, s, alpha, beta)
        else:
            raise ValueError(f"Unknown policy: {policy}")

        action_idx[t] = a

        # select current (possibly switching) true theta
        theta_true, active_idx = get_active_theta(rng, user, active_idx)

        # sample human response under the active true model
        theta_sa = float(theta_true[s, a])
        y = int(rng.random() < theta_sa)
        y_arr[t] = y

        # utility
        utility[t] = step_utility(y, a)

        # regret under the active true model
        opt_u = optimal_expected_utility(theta_true[s, :])
        chosen_u = expected_utility_under_true(theta_sa, a)
        regret[t] = opt_u - chosen_u  # expected regret

        # update belief
        alpha[s, a] += y
        beta[s, a] += (1 - y)

        if log_beliefs:
            alpha_hist[t] = alpha
            beta_hist[t] = beta

    return RunTrace(
        utility=utility,
        regret=regret,
        situation_idx=situation_idx,
        action_idx=action_idx,
        y=y_arr,
        alpha_hist=alpha_hist,
        beta_hist=beta_hist,
    )


# -----------------------------
# Aggregation + plotting
# -----------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def mean_std_over_runs(arr_runs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    arr_runs: (R, T)
    returns mean(T), std(T)
    """
    return arr_runs.mean(axis=0), arr_runs.std(axis=0)


def save_csv_timeseries(
    filepath: str,
    x: np.ndarray,
    series: Dict[str, Tuple[np.ndarray, np.ndarray]],
    x_name: str = "t",
) -> None:
    """
    series maps label -> (mean, std) arrays
    """
    labels = list(series.keys())
    with open(filepath, "w", newline="") as f:
        w = csv.writer(f)
        header = [x_name]
        for lab in labels:
            header += [f"{lab}_mean", f"{lab}_std"]
        w.writerow(header)
        for i in range(len(x)):
            row = [int(x[i])]
            for lab in labels:
                m, s = series[lab]
                row += [float(m[i]), float(s[i])]
            w.writerow(row)


def plot_mean_std_seaborn(
    runs_dict: Dict[str, np.ndarray],  # policy_code -> (R,T)
    y_label: str,
    outpath_pdf: str,
    title: str | None = None,
    legend_title: str | None = None,
) -> None:
    """
    Plot mean with SD band using seaborn.
    runs_dict maps policy_code -> array (R, T)
    """
    rows = []

    for pol_code, arr in runs_dict.items():
        label = POLICY_DISPLAY.get(pol_code, pol_code)
        R, T = arr.shape
        # fast-ish: append rows in python, OK at your sizes (30*300*5 = 45k)
        for r in range(R):
            for t in range(T):
                rows.append({"step": t, "value": float(arr[r, t]), "policy": label})

    df = pd.DataFrame(rows)

    plt.figure()
    ax = sns.lineplot(
        data=df,
        x="step",
        y="value",
        hue="policy",
        errorbar="sd",
        linewidth=2.0,
    )

    ax.set_xlabel("Interaction step")
    ax.set_ylabel(y_label)

    if title:
        ax.set_title(title)

    # cleaner legend
    leg = ax.legend_
    if leg is not None:
        leg.set_title(legend_title if legend_title is not None else "")
        # move legend outside if you want:
        # ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)

    plt.tight_layout()
    plt.savefig(outpath_pdf)
    plt.close()


def plot_belief_convergence_example(
    trace: RunTrace,
    s_name: str,
    a_name: str,
    theta_true: float,
    outpath_pdf: str,
) -> None:

    s = S_IDX[s_name]
    a = A_IDX[a_name]

    T = trace.alpha_hist.shape[0]
    posterior_mean = trace.alpha_hist[:, s, a] / (
        trace.alpha_hist[:, s, a] + trace.beta_hist[:, s, a]
    )

    df = pd.DataFrame({
        "step": np.arange(T),
        "Posterior mean": posterior_mean
    })

    plt.figure()

    ax = sns.lineplot(
        data=df,
        x="step",
        y="Posterior mean",
        linewidth=2.0
    )

    # True theta reference line
    ax.axhline(
        theta_true,
        linestyle="--",
        linewidth=2.0,
        label="True acceptance probability"
    )

    ax.set_xlabel("Interaction step")
    ax.set_ylabel("Acceptance probability")

    # Clean legend
    leg = ax.legend()
    if leg is not None:
        leg.set_title("")

    sns.despine()
    plt.tight_layout()
    plt.savefig(outpath_pdf)
    plt.close()


def save_run_metadata(filepath: str, args: argparse.Namespace) -> None:
    with open(filepath, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["parameter", "value"])
        w.writerow(["T", args.T])
        w.writerow(["runs", args.runs])
        w.writerow(["seed", args.seed])
        w.writerow(["cost_explain", args.cost_explain])
        w.writerow(["cost_update", args.cost_update])
        w.writerow(["cost_silent", args.cost_silent])
        w.writerow(["switch_prob", args.switch_prob])
        w.writerow(["switch_mode", args.switch_mode])
        w.writerow(["dynamic_pair", args.dynamic_pair])


def main() -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=300, help="interaction steps per run")
    parser.add_argument("--runs", type=int, default=30, help="runs per user type")
    parser.add_argument("--seed", type=int, default=42, help="base random seed")
    parser.add_argument("--outdir", type=str, default="out_icrs", help="output directory")
    parser.add_argument("--switch_prob", type=float, default=0.0,
                    help="probability of switching user response model per step (0 disables)")
    parser.add_argument("--switch_mode", type=str, default="toggle", choices=["toggle", "random"],
                    help="how to switch between models: toggle (0<->1) or random")
    parser.add_argument("--dynamic_pair", type=str, default="Type1_ExplanationOriented+Type2_Minimalist",
                    help="two user types to switch between, e.g., Type1_ExplanationOriented+Type2_Minimalist")
    parser.add_argument("--cost_explain", type=float, default=0.15,
                        help="communication cost for Explain")
    parser.add_argument("--cost_update", type=float, default=0.05,
                        help="communication cost for Update")
    parser.add_argument("--cost_silent", type=float, default=0.00,
                        help="communication cost for Silent")
    
    args = parser.parse_args()

    global ACTION_COST
    ACTION_COST = np.array(
        [args.cost_explain, args.cost_update, args.cost_silent],
        dtype=float
    )

    outdir = args.outdir
    figdir = os.path.join(outdir, "figures")
    resdir = os.path.join(outdir, "results")
    ensure_dir(figdir)
    ensure_dir(resdir)

    save_run_metadata(os.path.join(outdir, "run_metadata.csv"), args)

    user_types = make_user_types()

    # Optional: add dynamic switching user type (Type4)
    if args.switch_prob > 0.0:
        left_name, right_name = args.dynamic_pair.split("+")
        u_left = [u for u in user_types if u.name == left_name][0]
        u_right = [u for u in user_types if u.name == right_name][0]

        user_types.append(
            UserType(
                name="Type4_DynamicSwitching",
                theta=u_left.theta,  # fallback; not used when thetas is set
                thetas=[u_left.theta, u_right.theta],
                switch_prob=float(args.switch_prob),
                switch_mode=str(args.switch_mode),
            )
        )

    policies = [
        ("Always Explain", "always_explain"),
        ("Always Update", "always_update"),
        ("Always Silent", "always_silent"),
        ("Random", "random"),
        ("Adaptive (TS)", "adaptive_thompson"),
    ]

    x = np.arange(args.T)

    # Store one trace for belief convergence plot (Type3 + adaptive)
    example_trace: RunTrace | None = None

    for user in user_types:
        # Collect run arrays: dict policy_code -> (R, T)
        util_runs: Dict[str, np.ndarray] = {}
        reg_runs: Dict[str, np.ndarray] = {}

        for label, pol in policies:
            util_runs[pol] = np.zeros((args.runs, args.T), dtype=float)
            reg_runs[pol] = np.zeros((args.runs, args.T), dtype=float)

            for r in range(args.runs):
                rng = np.random.default_rng(args.seed + 1000 * hash(user.name) % 10_000 + 17 * r + 3 * hash(pol) % 10_000)

                log_beliefs = (user.name == "Type3_SituationSensitive" and pol == "adaptive_thompson" and r == 0)
                trace = simulate_one_run(rng, user, pol, args.T, log_beliefs=log_beliefs)

                util_runs[pol][r, :] = trace.utility
                reg_runs[pol][r, :] = np.cumsum(trace.regret)  # cumulative regret

                if log_beliefs:
                    example_trace = trace

        # Aggregate mean/std
        util_series: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        reg_series: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for label, pol in policies:
            m_u, s_u = mean_std_over_runs(util_runs[pol])
            m_r, s_r = mean_std_over_runs(reg_runs[pol])
            util_series[label] = (m_u, s_u)
            reg_series[label] = (m_r, s_r)

        # Save CSV
        save_csv_timeseries(
            os.path.join(resdir, f"{user.name}_utility.csv"),
            x,
            util_series,
            x_name="t",
        )
        save_csv_timeseries(
            os.path.join(resdir, f"{user.name}_cumregret.csv"),
            x,
            reg_series,
            x_name="t",
        )

        # Plot utility
        plot_mean_std_seaborn(
            util_runs,
            y_label="Utility",
            title=None,  # keep plot clean; describe in caption
            legend_title=None,
            outpath_pdf=os.path.join(figdir, f"{user.name}_utility.pdf"),
        )

        # Plot cumulative regret
        plot_mean_std_seaborn(
            reg_runs,
            y_label="Cumulative regret",
            title=None,
            legend_title=None,
            outpath_pdf=os.path.join(figdir, f"{user.name}_cumregret.pdf"),
        )

    # Belief convergence example (Type3, Failure–Explain)
    if example_trace is not None and example_trace.alpha_hist.shape[0] > 0:
        user3 = [u for u in user_types if u.name == "Type3_SituationSensitive"][0]
        theta_true = float(user3.theta[S_IDX["Failure"], A_IDX["Explain"]])
        plot_belief_convergence_example(
            example_trace,
            s_name="Failure",
            a_name="Explain",
            theta_true=theta_true,
            outpath_pdf=os.path.join(figdir, "Type3_belief_convergence_Failure_Explain.pdf"),
        )

    print(f"Done. Outputs written to: {outdir}/")
    print(f"- Figures: {outdir}/figures/")
    print(f"- CSVs:    {outdir}/results/")


if __name__ == "__main__":
    main()