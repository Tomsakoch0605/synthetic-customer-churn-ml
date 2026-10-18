# Recruiter-facing portfolio material

**GitHub description:** End-to-end synthetic churn ML: Python EDA, preprocessing, cross-validation, baseline comparison, explainability, reproducibility and CI.

**Topics:** `data-science`, `machine-learning`, `python`, `scikit-learn`, `classification`, `exploratory-data-analysis`, `cross-validation`, `model-evaluation`, `synthetic-data`, `github-actions`

**Resume bullets — portfolio facts only (not employer achievements):**
- Built a reproducible synthetic churn-classification pipeline with validation, preprocessing, 5-fold training cross-validation, and held-out evaluation.
- Benchmarked logistic regression against a prior-probability baseline and exported classification metrics and permutation importances.
- Automated Python-version testing and linting with GitHub Actions and documented results and limitations in English and Spanish.

**Interview pitch (60 seconds):**
"This is an independently built Data Science portfolio demonstration rather than a reproduction of an employer system. I defined a fictional monthly subscription churn problem and generated reproducible synthetic data, which avoids any corporate information. I explored class balance and plan-level differences, then used a stratified holdout and a preprocessing pipeline nested within cross-validation. I compared logistic regression with a baseline, selected the model based only on training folds, and evaluated it once on the reserved test data. I report ROC-AUC, average precision, classification scores, calibration via Brier loss, and permutation importance. The repository includes tests and CI. These synthetic results demonstrate my method, not performance on real customers."

**Technical interview questions:** Why is a baseline necessary? Why stratification? Where could leakage enter preprocessing? Why not optimize the threshold on the test set? Can permutation importance infer causality? How would you validate on time-based real data? What changes are required for production or fairness audits?
