import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

# ----------------------------
# Load data
# ----------------------------
with_explanations = pd.read_csv("./With_Explanations.csv", sep=";")
without_explanations = pd.read_csv("./Without_Explanations.csv", sep=";")

# Add group label (only for robust column selection)
tmp = pd.concat(
    [
        with_explanations.assign(Explanations="Group 2 (G2)"),
        without_explanations.assign(Explanations="Group 1 (G1)"),
    ],
    ignore_index=True,
)

# Select the 7 satisfaction Likert columns exactly as in your plotting pipeline
question_cols = tmp.columns.drop("Explanations")[-7:]

# Ensure numeric
g2 = with_explanations[question_cols].apply(pd.to_numeric, errors="coerce")
g1 = without_explanations[question_cols].apply(pd.to_numeric, errors="coerce")

# ----------------------------
# Per-question Welch t-tests
# ----------------------------
t_tests = {}
for col in question_cols:
    x2 = g2[col].dropna().to_numpy()
    x1 = g1[col].dropna().to_numpy()
    t_stat, p_val = ttest_ind(x2, x1, equal_var=False)
    t_tests[col] = (t_stat, p_val)

# ----------------------------
# CORRECT Overall test:
# per-participant overall satisfaction = mean across Q1..Q7
# ----------------------------
g2_overall = g2.mean(axis=1).dropna().to_numpy()
g1_overall = g1.mean(axis=1).dropna().to_numpy()

overall_t_stat, overall_p_val = ttest_ind(g2_overall, g1_overall, equal_var=False)

# ----------------------------
# Effect size (Cohen's d) on per-participant overall score
# ----------------------------
n1, n2 = len(g1_overall), len(g2_overall)
m1, m2 = float(np.mean(g1_overall)), float(np.mean(g2_overall))
s1, s2 = float(np.std(g1_overall, ddof=1)), float(np.std(g2_overall, ddof=1))

sp = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
d = (m2 - m1) / sp if sp > 0 else np.nan  # positive => G2 > G1

def interpret(effect: float) -> str:
    a = abs(effect)
    if a < 0.2:
        return "negligible"
    if a < 0.5:
        return "small"
    if a < 0.8:
        return "medium"
    return "large"

# ----------------------------
# Print results (human-readable)
# ----------------------------
print("Per-question Welch t-tests (G2 vs G1):")
for i, col in enumerate(question_cols, start=1):
    t_stat, p_val = t_tests[col]
    print(f"  Q{i}: t={t_stat:.5f}, p={p_val:.5g}")

print("\nOverall satisfaction (per-participant mean across Q1–Q7):")
print(f"  G1: n={n1}, mean={m1:.2f}, sd={s1:.2f}")
print(f"  G2: n={n2}, mean={m2:.2f}, sd={s2:.2f}")
print(f"  Overall: t={overall_t_stat:.2f}, p={overall_p_val:.2g}")
print(f"  Cohen's d = {d:.2f} ({interpret(d)} effect)")

# ----------------------------
# Save t-test results to CSV (Q1..Q7 + Overall)
# ----------------------------
t_test_results = pd.DataFrame(
    {
        "Question": [f"Q{i+1}" for i in range(len(question_cols))] + ["Overall"],
        "t_stat": [round(t_tests[col][0],2) for col in question_cols] + [overall_t_stat],
        "p_value": [round(t_tests[col][1],2) for col in question_cols] + [overall_p_val],
    }
)
t_test_results.to_csv("t_test_results.csv", index=False)
print("\nSaved: t_test_results.csv")