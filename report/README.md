# Project Report — LaTeX (Overleaf)

## Quick Start (Overleaf)

1. Go to [overleaf.com](https://www.overleaf.com) and create a new project.
2. Upload **all** files from this `report/` folder, including:
   - `main.tex` — the report source
   - `figures/` — all PNG figure files (16 files)
3. Set the compiler to **pdfLaTeX** (Menu → Settings → Compiler).
4. Click **Recompile**. Run it **twice** to resolve cross-references and TOC.

## What to Fill In

The following placeholders need to be replaced with your details (search for `[` in `main.tex`):

| Placeholder | Replace with |
|---|---|
| `[INSTITUTE NAME]` | Your institute/university name |
| `[DEPARTMENT NAME]` | Your department |
| `[PROJECT TITLE]` | Already filled — change if needed |
| `[PROGRAMME / DEPARTMENT]` | Your programme (e.g., Computer Science) |
| `[STUDENT NAME – ROLL NUMBER]` | Your name and roll number |
| `[SUPERVISOR NAME, DESIGNATION]` | Your supervisor's name and title |
| `[MONTH, YEAR]` | Submission month and year |
| `[ACADEMIC YEAR]` | e.g., 2024–2025 |
| `[SUPERVISOR]` | Supervisor name (in Declaration section) |

## Document Structure

The report follows the UG project template with these sections:

- **Front matter** (Roman page numbers): Title page, Certificate, Declaration, Acknowledgements, Abstract, TOC, List of Figures, List of Tables, Abbreviations
- **Chapters** (Arabic page numbers):
  1. Introduction
  2. Literature Review
  3. Requirements and Problem Formulation
  4. Proposed Methodology / System Design
  5. Implementation and Experimental Setup
  6. Results and Discussion
  7. Conclusion and Future Work
- **Back matter**: References, Appendices A–D

## Formatting

- Paper: A4, margins 1.25" left / 1" others
- Font: Times New Roman (via `mathptmx`)
- Spacing: 1.5 line spacing
- Body: justified
- Citation style: IEEE (numbered, square brackets)

## Figures

All 16 figure PNGs are in `figures/`. The LaTeX uses `\graphicspath` to find them automatically. Figures referenced in the report:

| Figure | File |
|---|---|
| Pipeline architecture | TikZ (drawn in LaTeX) |
| Model comparison | `05_model_comparison.png` |
| Feature importance | `07_feature_importance.png` |
| Actual vs predicted | `08_actual_vs_predicted_week.png` |
| Error by hour | `09_error_by_hour_and_weather.png` |
| Net flow | `10_net_flow_pred_vs_actual.png` |
| Rebalancing strategies | `11_rebalancing_strategies.png` |

## Local Compilation (if you have LaTeX installed)

```bash
cd report
pdflatex main.tex
pdflatex main.tex   # second pass for TOC/refs
```

## Data Sources

All metrics and numbers in the report are sourced from actual pipeline outputs:
- `outputs/metrics/evaluation_report.json`
- `outputs/metrics/training_report.json`
- `outputs/metrics/preprocessing_report.json`
- `outputs/metrics/rebalancing_report.json`
- `outputs/metrics/feature_report.json`
