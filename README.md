# MambaFormer-TPEF

Official code for **“MambaFormer-TPEF: A Hybrid State Space-Transformer Framework with Two-stage Peak Enhancement for Ultra-long Sequence Photovoltaic Power Forecasting”**.

## Overview

This repository contains the implementation and experiment materials for **MambaFormer-TPEF**, a photovoltaic (PV) power forecasting framework designed for **ultra-long sequence forecasting**.

The main Task-4 setting in this project is:

- **Input horizon:** 720 hours (30 days)
- **Prediction horizon:** 168 hours (7 days)
- **Forecast granularity:** hourly
- **Feature dimension:** 25
- **Target variable:** `InvPAC_kW_Avg`

The overall technical pipeline can be summarized as:

**Mamba backbone fitting + tree-model-based peak enhancement + MLEF ensemble fusion**

## Key Features

- Hybrid long-sequence forecasting framework for PV power prediction
- Task-4 setting for **720h → 168h** sequence-to-sequence forecasting
- Peak-oriented optimization for improving daytime peak fitting
- Multi-Level Ensemble Fusion (MLEF)
- SHAP-based interpretability analysis
- Supplementary project documents for experiments, ablations, and peak-loss design

## Method Summary

The repository focuses on a Task-4 PV forecasting setup where the model predicts the next 168 hourly power values from 720 hours of historical observations.

The implemented framework combines:

1. **Backbone sequence modeling** using Mamba-based long-sequence modeling
2. **Peak enhancement modules** for peak magnitude and timing refinement
3. **MLEF ensemble fusion** to combine the strengths of sequence models and tree-based models

According to the uploaded project summaries, the repository is organized around a Mamba-based forecasting pipeline with peak enhancement and meta-learning ensemble design. The experiment notes describe the best integrated setting as a Mamba-based base predictor together with MLEF ensemble fusion. :contentReference[oaicite:4]{index=4} :contentReference[oaicite:5]{index=5}

## Peak Optimization

This project includes dedicated peak-optimization components for improving **daytime peak prediction**, including:

- **Peak-aware loss functions**
- **Weighted peak-aware loss**
- **Peak extraction and evaluation**
- **Peak visualization**
- **Peak-aware ensemble logic**
- **Optional peak enhancement tools for tree-model-assisted correction**

The project documents describe APIs and usage patterns for modules such as `WeightedPeakAwareLoss`, `PeakExtractor`, `PeakVisualizer`, `PeakAwareEnsemble`, and `MLEFPeakEnhancer`. 

## Experimental Notes

The uploaded experiment summary describes the Task-4 setting as a long-horizon PV forecasting problem and reports that the Mamba-based pipeline outperformed Transformer in the long-sequence setup, while MLEF further improved the overall performance. The same notes summarize a best overall result around **R² = 0.558** for the integrated MLEF setting. :contentReference[oaicite:7]{index=7}

The ablation summary also records several negative trials, including unstable or ineffective peak-loss variants and feature-engineering alternatives, which are retained in `docs/` for reference and future development. :contentReference[oaicite:8]{index=8}

## Repository Structure

```text
MambaFormer-TPEF/
├─ data/
│  ├─ sample/
│  └─ README.md
├─ docs/
│  ├─ PEAK_OPTIMIZATION_GUIDE.md
│  ├─ ablation_study.md
│  ├─ peak_loss_functions.md
│  ├─ task4_experiment_report.md
│  └─ task4_model_config_summary.md
├─ figure/
├─ output/
│  ├─ paper_figures/
│  ├─ shap_analysis/
│  └─ task4_peak_mlef_*/
├─ scripts/
│  ├─ README.md
│  ├─ generate_paper_figures.py
│  ├─ shap_analysis_task4.py
│  └─ train_task4_peak_mlef.py
├─ src/
│  └─ mtm_mlef/
├─ .gitignore
├─ LICENSE
├─ README.md
└─ config.yaml

