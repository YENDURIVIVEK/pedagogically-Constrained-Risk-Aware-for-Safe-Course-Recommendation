---
title: PCRA-HGNN Advisor Dashboard
emoji: 🎓
colorFrom: blue
colorTo: indigo
sdk: static
pinned: false
---

# PCRA-HGNN: Early Student Profiling & Safe Course Recommendation

Official code and interactive advisor dashboard for **Pedagogically Constrained, Risk-Aware Heterogeneous Graph Neural Networks for Early Student Profiling and Safe Course Recommendation** (OULAD benchmark, 32,593 enrollments).

## Repository Contents

- `index.html` — Zero-backend static web dashboard (works directly on **GitHub Pages** and **Hugging Face Static Spaces**; loads both CSVs client-side).
- `stdcrs.ipynb` — Complete reproducible Kaggle experiment notebook (nested learner-grouped 5-fold CV, 80/20 grouped holdout, relation ablations, temporal `2014J` curriculum recommendation, safety analysis, and all paper tables/figures).
- `app.py` — Standalone Python/Gradio dashboard for local execution.
- `requirements.txt` — Minimal dependencies for `app.py`.
- `oulad_temporal_ranking_top5_by_user_model.csv` — Top-5 recommendations per `(Seed, Model, Student_ID)` for the `2014J` evaluation presentation.
- `nested_grouped_cv_oof_predictions.csv` — Pooled out-of-fold Day-28 risk probabilities for early-warning display.

## Option A: Live Static Demo (Free, No Backend Required)

- **GitHub Pages:** Upload `index.html` to this repository, then enable **Settings → Pages → Deploy from a branch (`main` / `/root`)**.
- **Hugging Face Static Space:** Create a **Static** Space (`sdk: static` in `README.md`) and upload `index.html` and `README.md` (it loads CSVs from `./` if uploaded, or falls back to this public GitHub repository).

## Option B: Run Locally with Python / Gradio

```bash
pip install -r requirements.txt
python app.py
```
Then open `http://127.0.0.1:7860` in your browser.
