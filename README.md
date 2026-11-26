---
title: PCRA-HGNN Advisor Dashboard
emoji: 🎓
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
---

# PCRA-HGNN: Early Student Profiling & Safe Course Recommendation

Official code and interactive advisor dashboard for **Pedagogically Constrained, Risk-Aware Heterogeneous Graph Neural Networks for Early Student Profiling and Safe Course Recommendation** (OULAD benchmark, 32,593 enrollments).

## Repository Contents

- `stdcrs.ipynb` — Complete reproducible Kaggle experiment notebook (nested learner-grouped 5-fold CV, 80/20 grouped holdout, relation ablations, temporal `2014J` curriculum recommendation, safety analysis, and all paper tables/figures).
- `app.py` — Standalone interactive Gradio advisor dashboard showing side-by-side Unconstrained HGNN vs. Risk-Aware HGNN (`λ = 0.35`) Top-5 consensus recommendations and optional Day-28 HGNN risk profiles.
- `requirements.txt` — Minimal runtime dependencies for the dashboard.
- `oulad_temporal_ranking_top5_by_user_model.csv` — *(Export from Kaggle `OUTPUT_DIR`)* Top-5 recommendations per `(Seed, Model, Student_ID)` for the `2014J` evaluation presentation.
- `nested_grouped_cv_oof_predictions.csv` — *(Optional, export from Kaggle `OUTPUT_DIR`)* Pooled out-of-fold Day-28 risk probabilities for displaying early-warning scores in the dashboard.

## Quick Start (Local Dashboard)

1. Download `oulad_temporal_ranking_top5_by_user_model.csv` (and optionally `nested_grouped_cv_oof_predictions.csv`) from Kaggle's `/kaggle/working/oulad_hgnn_revised_outputs/` into this folder.
2. Install dependencies and launch:
   ```bash
   pip install -r requirements.txt
   python app.py
   ```
3. Open `http://127.0.0.1:7860` in your browser.

## Deploying Permanently to Hugging Face Spaces

1. Create a new **Gradio** Space on Hugging Face.
2. Upload `app.py`, `requirements.txt`, `README.md`, and `oulad_temporal_ranking_top5_by_user_model.csv` (plus `nested_grouped_cv_oof_predictions.csv` if desired).
   - Or run `gradio deploy` from inside this directory after logging in with your Hugging Face token.
3. The Space builds automatically using the YAML header in `README.md`.
