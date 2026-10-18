# Measured results | Interpretación de resultados

All values below refer **exclusively to artificial, seeded synthetic data** (1,500 fictional subscription records; seed 42). They cannot establish performance, customer behavior, or economic impact in any real organization.

| Metric | Measured value |
|---|---:|
| Training samples | 1,200 |
| Held-out test samples | 300 |
| Test churn prevalence | 30.67% |
| Prior-baseline 5-fold CV ROC-AUC | 0.500 |
| Logistic regression 5-fold CV ROC-AUC | 0.719 |
| Test ROC-AUC | 0.723 |
| Test average precision | 0.574 |
| Test precision, threshold 0.5 | 0.667 |
| Test recall, threshold 0.5 | 0.239 |
| Test F1, threshold 0.5 | 0.352 |
| Test Brier loss | 0.182 |
| Confusion matrix (TN, FP, FN, TP) | 197, 11, 70, 22 |

**English:** Cross-validated discrimination exceeded the constant-prior baseline, but at the predeclared decision threshold, recall was low: 70 of 92 positive synthetic cases were missed. This demonstrates why ROC-AUC alone does not specify operational usefulness. Test-set permutation importance ranked tenure, monthly fee, and simulated payment failures highest; feature permutation scores express sensitivity, not causal effects. The fixed threshold avoids optimizing on the held-out set. Reported uncertainty and model behavior would need independent evaluation on authorized real data before deployment.

**Español:** La capacidad discriminativa en validación cruzada superó el modelo base, pero con el umbral de decisión fijado previamente, la sensibilidad fue baja: se omitieron 70 de 92 casos positivos sintéticos. Esto muestra por qué un ROC-AUC aceptable no garantiza utilidad operativa. La importancia por permutación situó primero la antigüedad, la cuota mensual y los fallos de pago ficticios; estos valores no demuestran causalidad. Antes de cualquier despliegue serían necesarias nuevas evaluaciones con datos reales autorizados y una revisión de riesgos.

For exact unrounded metrics, see [`../artifacts/metrics.json`](../artifacts/metrics.json).
