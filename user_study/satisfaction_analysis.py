import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

# Load data
with_explanations = pd.read_csv("./With_Explanations.csv", sep=";")
without_explanations = pd.read_csv("Without_Explanations.csv", sep=";")

# Add column for explanations
df_no_explanations = without_explanations.copy()
df_no_explanations["Explanations"] = "Group 1 (G1)" #"Condition 1"
df_explanations = with_explanations.copy()
df_explanations["Explanations"] = "Group 2 (G2)" # "Condition 2"

# Combine
combined_df = pd.concat([df_explanations, df_no_explanations], ignore_index=True)

# Fix: ignore the Explanations column when selecting the last 7 Likert-scale columns
question_cols = combined_df.columns.drop("Explanations")[-7:]

# Drop outliers
def drop_outliers(df, columns):
    for col in columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

# Drop outliers for each subset separately
df_explanations = drop_outliers(df_explanations, question_cols)
df_no_explanations = drop_outliers(df_no_explanations, question_cols)

# Combine again after dropping outliers
combined_df = pd.concat([df_explanations, df_no_explanations], ignore_index=True)


# Melt for plotting
df_long = combined_df.melt(
    id_vars=["Explanations"],
    value_vars=question_cols,
    var_name="Question",
    value_name="Response"
)

# Rename columns in df_long
question_mapping = {old_name: f"Q{i+1}" for i, old_name in enumerate(question_cols)}
df_long["Question"] = df_long["Question"].map(question_mapping)


def plot_user_satisfaction_boxplot(
    df_long,
    outpath_pdf="user_satisfaction_responses.pdf",
    outpath_svg=None,
    title=None,
):
    """
    df_long columns expected:
      - "Question"  (e.g., Q1..Q7)
      - "Response"  (Likert 1..5)
      - "Explanations" (e.g., "Group 1 (G1)" / "Group 2 (G2)")
    """

    # Match the style used in your new experiment plots
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

    # Optional: order questions numerically if they’re strings like "Q1", "Q2", ...
    q_order = sorted(df_long["Question"].unique(), key=lambda q: int(str(q).lstrip("Q")))

    plt.figure(figsize=(10, 4.8))
    ax = sns.boxplot(
        data=df_long,
        x="Question",
        y="Response",
        hue="Explanations",
        order=q_order,
        hue_order=["Group 1 (G1)", "Group 2 (G2)"],
        width=0.65,
        fliersize=2,
        linewidth=1.2,
    )

    # Clean axis labels and limits (Likert 1–5)
    ax.set_xlabel("Question")
    ax.set_ylabel("Response (1–5 Likert)")
    ax.set_ylim(1, 5)

    # Titles: keep plots clean; prefer caption in paper
    if title:
        ax.set_title(title)

    # Legend: consistent, minimal
    leg = ax.legend(title="")
    if leg is not None:
        leg.set_frame_on(True)
        leg._loc = 0  # best

    sns.despine(trim=True)
    plt.tight_layout()

    plt.savefig(outpath_pdf)
    if outpath_svg:
        plt.savefig(outpath_svg, format="svg")

    plt.close()


plot_user_satisfaction_boxplot(
    df_long,
    outpath_pdf="./user_satisfaction_responses.pdf",
    outpath_svg="./user_satisfaction_responses.svg",  # optional
    title=None  # keep None for camera-ready
)