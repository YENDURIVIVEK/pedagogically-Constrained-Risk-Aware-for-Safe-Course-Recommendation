"""Standalone Gradio Advisor Dashboard for PCRA-HGNN (OULAD).

Displays computed Top-5 curriculum recommendations across evaluation seeds
(side-by-side Unconstrained HGNN vs. Risk-Aware HGNN, plus single-model view)
and, when pooled OOF risk predictions are provided, the Day-28 early-warning
at-risk probability from the classification head.
"""

from pathlib import Path
import os
import pandas as pd
import gradio as gr

BASE_DIR = Path(__file__).resolve().parent

TOP5_CANDIDATES = [
    BASE_DIR / "oulad_temporal_ranking_top5_by_user_model.csv",
    BASE_DIR / "data" / "oulad_temporal_ranking_top5_by_user_model.csv",
    Path("/kaggle/working/oulad_hgnn_revised_outputs/oulad_temporal_ranking_top5_by_user_model.csv"),
]
OOF_CANDIDATES = [
    BASE_DIR / "nested_grouped_cv_oof_predictions.csv",
    BASE_DIR / "data" / "nested_grouped_cv_oof_predictions.csv",
    Path("/kaggle/working/oulad_hgnn_revised_outputs/nested_grouped_cv_oof_predictions.csv"),
]

# Training-period module non-completion rates (2013B, 2013J, 2014B) from OULAD.
TRAINING_MODULE_FAILURE_RATES = {
    "AAA": 27.4,
    "BBB": 53.7,
    "CCC": 65.8,
    "DDD": 59.3,
    "EEE": 44.7,
    "FFF": 53.1,
    "GGG": 40.1,
}


def _find_existing(paths):
    for p in paths:
        if p.is_file():
            return p
    return None


def load_data():
    top5_path = _find_existing(TOP5_CANDIDATES)
    if top5_path is None:
        return pd.DataFrame(), pd.DataFrame(), None

    top5_df = pd.read_csv(top5_path)
    user_col = next((c for c in ["Student_ID", "User_ID"] if c in top5_df.columns), None)
    item_col = next((c for c in ["Recommended_Module", "Item_ID"] if c in top5_df.columns), None)
    if user_col is None or item_col is None:
        raise ValueError("Top-5 CSV must contain Student_ID/User_ID and Recommended_Module/Item_ID.")

    top5_df = top5_df.rename(columns={user_col: "Student_ID", item_col: "Recommended_Module"})
    top5_df["Student_ID"] = top5_df["Student_ID"].astype(str).str.strip()
    top5_df["Recommended_Module"] = top5_df["Recommended_Module"].astype(str).str.strip()

    oof_path = _find_existing(OOF_CANDIDATES)
    oof_df = pd.read_csv(oof_path) if oof_path is not None else pd.DataFrame()
    if not oof_df.empty and "id_student" in oof_df.columns:
        oof_df["id_student"] = oof_df["id_student"].astype(str).str.strip()
    return top5_df, oof_df, top5_path


TOP5_DF, OOF_DF, TOP5_PATH = load_data()

MODELS_AVAILABLE = (
    sorted(TOP5_DF["Model"].dropna().astype(str).unique().tolist())
    if not TOP5_DF.empty else []
)
USER_CHOICES = (
    sorted(TOP5_DF["Student_ID"].dropna().astype(str).unique().tolist())
    if not TOP5_DF.empty else []
)
DEFAULT_MODEL = "HGNN (Risk-Aware)" if "HGNN (Risk-Aware)" in MODELS_AVAILABLE else (
    MODELS_AVAILABLE[0] if MODELS_AVAILABLE else None
)


def _consensus_table(student_id: str, model_name: str) -> pd.DataFrame:
    cols = ["Rank", "Recommended_Module", "Seed_Votes", "Mean_Rank", "Train_Fail_Rate_%"]
    if TOP5_DF.empty or not student_id or not model_name:
        return pd.DataFrame(columns=cols)
    rows = TOP5_DF.loc[
        TOP5_DF["Student_ID"].eq(str(student_id)) & TOP5_DF["Model"].astype(str).eq(str(model_name))
    ]
    if rows.empty:
        return pd.DataFrame(columns=cols)
    consensus = (
        rows.groupby("Recommended_Module", as_index=False)
        .agg(Seed_Votes=("Seed", "nunique"), Mean_Rank=("Rank", "mean"))
        .sort_values(["Seed_Votes", "Mean_Rank", "Recommended_Module"], ascending=[False, True, True])
        .head(5)
    )
    return pd.DataFrame({
        "Rank": range(1, len(consensus) + 1),
        "Recommended_Module": consensus["Recommended_Module"].to_numpy(),
        "Seed_Votes": consensus["Seed_Votes"].to_numpy(),
        "Mean_Rank": consensus["Mean_Rank"].round(2).to_numpy(),
        "Train_Fail_Rate_%": [
            TRAINING_MODULE_FAILURE_RATES.get(m, float("nan")) for m in consensus["Recommended_Module"]
        ],
    })


def _risk_summary_md(student_id: str) -> str:
    if TOP5_DF.empty:
        return (
            "> **Missing data file:** Place `oulad_temporal_ranking_top5_by_user_model.csv` "
            "next to `app.py` (exported from `OUTPUT_DIR` on Kaggle) and restart the app."
        )
    if not student_id:
        return "Select a student ID to view recommendations."

    if OOF_DF.empty or "id_student" not in OOF_DF.columns:
        return (
            f"**Selected Learner:** `{student_id}` | **Target Presentation:** `2014J`  \n"
            "*Optional: add `nested_grouped_cv_oof_predictions.csv` beside `app.py` to show Day-28 HGNN risk probabilities.*"
        )

    s_rows = OOF_DF.loc[OOF_DF["id_student"].eq(str(student_id))]
    if s_rows.empty:
        return f"**Selected Learner:** `{student_id}` (no Day-28 OOF profile rows found)."

    hgnn_cols = [c for c in OOF_DF.columns if c.startswith("OOF_HGNN_seed_")]
    lines = [f"**Selected Learner:** `{student_id}` | **Enrollments in OULAD:** {len(s_rows)}"]
    if hgnn_cols:
        probs = s_rows[hgnn_cols].mean(axis=1)
        max_p = float(probs.max())
        flag = "High Day-28 Risk (>= 0.50)" if max_p >= 0.50 else "Low/Moderate Day-28 Risk (< 0.50)"
        lines.append(f"**Max Day-28 HGNN Predicted At-Risk Prob.:** `{max_p:.3f}` ({flag})")
    hist_rows = s_rows.loc[s_rows["code_presentation"].astype(str).ne("2014J")]
    if not hist_rows.empty and "At_Risk" in hist_rows.columns:
        rho_u = float(hist_rows["At_Risk"].mean())
        branch = "Branch B: Vulnerable (safety penalty active)" if rho_u >= 0.50 else "Branch A: Well-prepared"
        lines.append(f"**Historical At-Risk Rate (ρ_u):** `{rho_u:.2f}` — *{branch}*")
    return "  \n".join(lines)


def update_dashboard(student_id: str, model_name: str):
    return (
        _risk_summary_md(student_id),
        _consensus_table(student_id, model_name),
        _consensus_table(student_id, "HGNN"),
        _consensus_table(student_id, "HGNN (Risk-Aware)"),
    )


with gr.Blocks(title="PCRA-HGNN Advisor Dashboard") as demo:
    gr.Markdown(
        "# PCRA-HGNN: Student Risk Profiling & Safe Course Recommendation Dashboard\n"
        "Inspect consensus Top-5 recommendations (`2014J` test presentation across evaluation seeds) "
        "and compare **Unconstrained HGNN** vs. **Risk-Aware HGNN** (λ = 0.35)."
    )
    with gr.Row():
        user_selector = gr.Dropdown(
            choices=USER_CHOICES,
            value=USER_CHOICES[0] if USER_CHOICES else None,
            label="Select Student ID",
            filterable=True,
        )
        model_selector = gr.Dropdown(
            choices=MODELS_AVAILABLE,
            value=DEFAULT_MODEL,
            label="Primary Model Selector",
        )

    risk_banner = gr.Markdown(
        value=_risk_summary_md(USER_CHOICES[0] if USER_CHOICES else "")
    )

    gr.Markdown("### Side-by-Side Comparison (Unconstrained vs. Risk-Aware HGNN)")
    with gr.Row():
        uncons_table = gr.Dataframe(
            value=_consensus_table(USER_CHOICES[0] if USER_CHOICES else "", "HGNN"),
            label="Unconstrained HGNN (Top-5 Consensus)",
            interactive=False,
        )
        risk_aware_table = gr.Dataframe(
            value=_consensus_table(USER_CHOICES[0] if USER_CHOICES else "", "HGNN (Risk-Aware)"),
            label="Risk-Aware HGNN (Top-5 Consensus, λ = 0.35)",
            interactive=False,
        )

    gr.Markdown("### Selected Model Detail")
    selected_table = gr.Dataframe(
        value=_consensus_table(USER_CHOICES[0] if USER_CHOICES else "", DEFAULT_MODEL or ""),
        label="Computed Top-5 Recommendations (Selected Model)",
        interactive=False,
    )

    for trigger in (user_selector, model_selector):
        trigger.change(
            update_dashboard,
            inputs=[user_selector, model_selector],
            outputs=[risk_banner, selected_table, uncons_table, risk_aware_table],
        )

if __name__ == "__main__":
    demo.launch(show_error=True)
