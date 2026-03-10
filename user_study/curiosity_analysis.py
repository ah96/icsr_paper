import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
with_explanations = pd.read_csv("./With_Explanations.csv", sep=";")
without_explanations = pd.read_csv("Without_Explanations.csv", sep=";")

checkbox_columns = with_explanations.columns[-16:-7]
print(checkbox_columns)

with_checkbox_means = with_explanations[checkbox_columns].mean() * 100
without_checkbox_means = without_explanations[checkbox_columns].mean() * 100

def plot_curiosity_bars(
    without_checkbox_means,
    with_checkbox_means,
    questions,
    outpath_pdf="curiosity.pdf",
    outpath_svg=None,
    title=None,
    # optional raw data for uncertainty
    with_explanations_df: pd.DataFrame | None = None,
    without_explanations_df: pd.DataFrame | None = None,
    checkbox_columns=None,            # list/Index, same order as questions
    uncertainty: str | None = "ci",   # "ci" | "se" | "sd" | None
    ci_level: int = 95,
):
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

    without_checkbox_means = np.asarray(without_checkbox_means, dtype=float)
    with_checkbox_means = np.asarray(with_checkbox_means, dtype=float)

    Q = len(questions)
    if len(without_checkbox_means) != Q or len(with_checkbox_means) != Q:
        raise ValueError("Means arrays and questions must have the same length.")

    df = pd.DataFrame({
        "Question": list(questions) * 2,
        "Score (%)": np.concatenate([without_checkbox_means, with_checkbox_means]),
        "Group": (["Group 1 (G1)"] * Q) + (["Group 2 (G2)"] * Q),
    })

    plt.figure(figsize=(10, 4.8))
    hue_order = ["Group 1 (G1)", "Group 2 (G2)"]
    ax = sns.barplot(
        data=df,
        x="Question",
        y="Score (%)",
        hue="Group",
        order=list(questions),
        hue_order=hue_order,
        errorbar=None,
    )

    def _err_from_raw(df_raw: pd.DataFrame) -> np.ndarray:
        errs = []
        z = 1.96 if ci_level == 95 else 1.645
        for col in checkbox_columns:
            x = pd.to_numeric(df_raw[col], errors="coerce").dropna().to_numpy(dtype=float)
            n = len(x)
            if n <= 1:
                errs.append(0.0)
                continue
            sd = float(np.std(x, ddof=1))
            se = sd / np.sqrt(n)
            if uncertainty == "sd":
                e = sd
            elif uncertainty == "se":
                e = se
            elif uncertainty == "ci":
                e = z * se
            else:
                e = 0.0
            errs.append(100.0 * e)
        return np.asarray(errs, dtype=float)

    # ---- robust manual error bars (no patch counting) ----
    if (
        uncertainty is not None
        and with_explanations_df is not None
        and without_explanations_df is not None
        and checkbox_columns is not None
    ):
        if len(checkbox_columns) != Q:
            raise ValueError("checkbox_columns must have the same length/order as questions.")

        err_g1 = _err_from_raw(without_explanations_df)
        err_g2 = _err_from_raw(with_explanations_df)

        # x positions for categories are 0..Q-1
        x_centers = np.arange(Q, dtype=float)

        # seaborn bar width defaults vary; use the actual drawn bar width if possible
        # We'll estimate offsets based on typical grouped-bar geometry.
        group_width = 0.8              # width reserved for each category
        bar_width = group_width / len(hue_order)
        offsets = np.array([-0.5, 0.5]) * bar_width  # for 2 groups: left/right within the category

        # Add errorbars: Group1 on the left, Group2 on the right
        for i in range(Q):
            # Group 1
            ax.errorbar(
                x_centers[i] + offsets[0],
                without_checkbox_means[i],
                yerr=err_g1[i],
                fmt="none",
                capsize=3,
                linewidth=1.2,
                color="black",
                zorder=10,
            )
            # Group 2
            ax.errorbar(
                x_centers[i] + offsets[1],
                with_checkbox_means[i],
                yerr=err_g2[i],
                fmt="none",
                capsize=3,
                linewidth=1.2,
                color="black",
                zorder=10,
            )

    ax.set_xlabel("Question")
    ax.set_ylabel("Average checkbox score (%)")
    if title:
        ax.set_title(title)
    ax.set_ylim(0, 100)

    leg = ax.legend(title="")
    if leg is not None:
        leg.set_frame_on(True)

    sns.despine(trim=True)
    plt.tight_layout()
    plt.savefig(outpath_pdf)
    if outpath_svg:
        plt.savefig(outpath_svg, format="svg")
    plt.close()


new_column_names = ['Q1','Q2','Q3','Q4','Q5','Q6','Q7','Q8','Q9']

plot_curiosity_bars(
    without_checkbox_means=without_checkbox_means,
    with_checkbox_means=with_checkbox_means,
    questions=new_column_names,
    outpath_pdf="./user_curiosity_responses.pdf",
    outpath_svg="./user_curiosity_responses.svg",
    title=None,
    # NEW:
    with_explanations_df=with_explanations,
    without_explanations_df=without_explanations,
    checkbox_columns=checkbox_columns,
    uncertainty="ci",   # "ci" (95%), or "se", or "sd", or None
    ci_level=95,
)