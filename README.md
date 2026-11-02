# Loan Default Risk Prediction using Machine Learning & Pipeline

This project demonstrates an **end-to-end machine learning workflow** for predicting whether a personal-loan applicant has a **high probability of defaulting** (Low risk vs High risk). It compares **Logistic Regression, Random Forest, Gradient Boosting and a Multi-Layer Perceptron (MLP)** with one shared pipeline and one fair process, then saves the selected model for reuse.

It follows **industrial best practices** by:

- Putting **scaling and encoding inside a Pipeline**, so they are fitted on training data only (no data leakage)
- Using **stratified splitting** because only about a quarter of customers default
- Giving every algorithm the **same cross-validation folds and a similar tuning effort**
- Choosing the final model with **cross-validation on training data only**, so the test set is never used to pick a winner
- Judging models with **precision, recall, F1 and ROC-AUC**, not accuracy alone
- Explaining the model with **permutation feature importance**
- Saving the trained pipeline with **Joblib** and reusing it from a separate **prediction script**
- Saving plots, tables and metrics to files for **reproducibility**

---

## Dependencies

Requires **Python 3.9+** (tested on Python 3.12). Install the required Python packages before running the project:

```bash
pip install -r requirements.txt
```

or directly:

```bash
pip install pandas numpy matplotlib scikit-learn joblib
```

---

## Dataset Information

**Source:** `Loan_Default.csv` (1,200 customers, 10 features, no missing values, no duplicate rows)

**Features (10):**

| # | Feature | Description |
|---|---------|-------------|
| 1 | Age | Applicant age |
| 2 | Income | Annual income |
| 3 | LoanAmount | Requested loan |
| 4 | CreditScore | Credit score |
| 5 | EmploymentYears | Years employed |
| 6 | ExistingLoans | Number of existing loans |
| 7 | MonthlyDebt | Existing monthly debt |
| 8 | LoanTerm | Loan duration |
| 9 | PreviousDefault | Yes / No |
| 10 | HomeOwnership | Rent / Own / Mortgage |

**Target:**

- **0** = Low default risk (879 customers, 73.2%)
- **1** = High default risk (321 customers, 26.8%)

An "always predict 0" model is already **73.3% accurate** on the test set, so accuracy alone is a weak measure here.

---

## Workflow

### Data Preparation

- Load the CSV and check that all required columns exist and the target contains only 0 and 1
- Check missing values and duplicates (none found)
- Check class balance (73 / 27)

### Train-Test Split

- Split into **70% train (840 rows)** and **30% test (360 rows)**
- Use **stratified sampling**: both sets keep the same 26.8% default rate, so the test set has enough defaulters (96) to measure recall reliably

### Pipeline Construction

- **Step 1:** `ColumnTransformer`
  - `StandardScaler` for the 8 numeric features
  - `OneHotEncoder` for `PreviousDefault` and `HomeOwnership`
- **Step 2:** the classifier (any of the five models below)

### Models Compared

| Model | Tuning (5-fold CV on training data) |
|-------|-------------------------------------|
| Logistic Regression | `C` in 0.01, 0.1, 1, 10 |
| Random Forest (300 trees) | `max_depth` in 4, 8, None and `min_samples_leaf` in 1, 5, 10 |
| Gradient Boosting | `n_estimators`, `learning_rate`, `max_depth` (8 combinations) |
| MLP (baseline from the assignment) | `(32, 16)`, relu, adam, `max_iter=1000` |
| MLP (tuned) | best single change from the MLP experiments below |

### MLP Hyperparameter Experiments

One parameter is changed at a time, everything else stays at the baseline. Each value is scored with 5-fold cross-validation on the **training data only**.

| Experiment | Values tried |
|------------|--------------|
| Activation | identity, logistic, tanh, relu |
| Hidden Layers | (10,), (20, 10), (50, 25), (100, 50, 25) |
| Learning Rate | 0.0001, 0.001, 0.01, 0.1 |
| Regularization (extra) | alpha = 0.0001, 0.01, 0.1, 1.0 |

`identity` activation is kept as a diagnostic but cannot be the tuned MLP, because it turns the network into a linear model.

### Model Selection

- The final model is the one with the best **cross-validated ROC-AUC** (ties broken by F1)
- The test set is evaluated **once**, after the choice is made

### Decision Threshold

- The default threshold is 0.5. For each model a second threshold is chosen from **training-data cross-validation** to maximize F1, trading precision for higher recall

### Model Saving & Loading

- Save the full pipeline with **Joblib** (`joblib.dump`)
- Save a **metadata file** (expected columns, allowed category values, library versions, metrics)
- Load both in a separate script (`joblib.load`) and predict without retraining

---

## Running the Project

```bash
python loan_default_pipeline.py
```

The run takes about 3 to 4 minutes. Expected output (shortened):

```text
Dataset Shape :: (1200, 11)
Total missing values :: 0
Class 0 (Low risk) :: 879 rows (73.2%)
Class 1 (High risk) :: 321 rows (26.8%)
Train_x Shape ::  (840, 10)
Test_x Shape  ::  (360, 10)
...
  Logistic Regression  C=1                      CV AUC 0.796 | F1 0.484
  Random Forest        max_depth=4, ...         CV AUC 0.789 | F1 0.295
  Gradient Boosting    learning_rate=0.1, ...   CV AUC 0.786 | F1 0.491

Final model (best CV ROC-AUC) :: Logistic Regression (CV AUC 0.796)
...
Model saved to outputs/loan_default_model.joblib
Loaded model prediction for first test sample: 1
```

**Generated files** (in `outputs/`):

```text
outputs/
├── figures/
│   ├── eda_overview.png
│   ├── model_comparison.png
│   ├── roc_curves.png
│   ├── feature_importance.png
│   ├── confusion_matrix_final.png
│   ├── confusion_matrix_baseline_mlp.png
│   ├── training_loss.png
│   └── mlp_hyperparameter_experiments.png
├── model_comparison.csv
├── mlp_experiment_results.csv
├── metrics.json
├── model_metadata.json
└── loan_default_model.joblib
```

---

## Results

All numbers below are from `random_state=42`. The test set has 360 rows, of which 96 are high-risk customers.

**Model comparison (test set, threshold 0.5):**

| Model | CV ROC-AUC | Train Acc | Test Acc | Precision | Recall | F1 | Test ROC-AUC |
|-------|-----------|-----------|----------|-----------|--------|----|--------------|
| Always predict "Low risk" | - | - | 0.733 | - | 0.000 | - | 0.500 |
| **Logistic Regression (selected)** | **0.796** | 0.795 | 0.783 | 0.622 | 0.479 | 0.541 | **0.789** |
| Random Forest | 0.789 | 0.805 | 0.753 | 0.640 | 0.167 | 0.264 | 0.761 |
| Gradient Boosting | 0.786 | 0.849 | 0.794 | 0.667 | 0.458 | 0.543 | 0.778 |
| MLP (baseline 32-16 relu) | 0.704 | 0.996 | 0.706 | 0.451 | 0.479 | 0.465 | 0.707 |
| MLP (tuned, logistic activation) | 0.793 | 0.788 | 0.792 | 0.630 | 0.531 | 0.576 | 0.785 |

**Each model at its own tuned threshold (chosen on training data):**

| Model | Threshold | Test Recall | Test F1 |
|-------|-----------|-------------|---------|
| Logistic Regression | 0.24 | 0.708 | 0.571 |
| Random Forest | 0.29 | 0.667 | 0.540 |
| Gradient Boosting | 0.24 | 0.698 | 0.547 |
| MLP (baseline) | 0.10 | 0.604 | 0.511 |
| MLP (tuned) | 0.23 | 0.688 | 0.557 |

**Selected model (Logistic Regression), confusion matrix** (rows = actual, columns = predicted):

```text
Threshold 0.50                       Threshold 0.24
                 Pred Low  Pred High                 Pred Low  Pred High
Actual Low          236        28      Actual Low       190        74
Actual High          50        46      Actual High       28        68
```

**Feature importance** (drop in ROC-AUC when a feature is shuffled, test set):

| Feature | Importance |
|---------|-----------|
| CreditScore | 0.175 |
| PreviousDefault | 0.060 |
| Income | 0.035 |
| LoanAmount | 0.009 |
| HomeOwnership, MonthlyDebt, Age | about 0.000 to 0.002 |
| EmploymentYears, LoanTerm, ExistingLoans | slightly below 0 (no measurable signal) |

**MLP experiments (cross-validation on training data only):**

| Experiment | Value | CV Acc | CV F1 | CV ROC-AUC | Train Acc |
|------------|-------|--------|-------|------------|-----------|
| Baseline | (32, 16) relu, lr 0.001 | 0.710 | 0.431 | 0.704 | 1.000 |
| Activation | identity | 0.776 | 0.492 | 0.793 | 0.795 |
| Activation | **logistic** | 0.771 | 0.506 | 0.793 | 0.793 |
| Activation | tanh | 0.706 | 0.424 | 0.689 | 0.999 |
| Activation | relu | 0.710 | 0.431 | 0.704 | 1.000 |
| Hidden Layers | (10,) | 0.768 | 0.481 | 0.767 | 0.823 |
| Hidden Layers | (20, 10) | 0.730 | 0.475 | 0.718 | 0.968 |
| Hidden Layers | (50, 25) | 0.717 | 0.444 | 0.705 | 1.000 |
| Hidden Layers | (100, 50, 25) | 0.736 | 0.470 | 0.716 | 1.000 |
| Learning Rate | 0.0001 | 0.768 | 0.497 | 0.775 | 0.820 |
| Learning Rate | 0.001 | 0.710 | 0.431 | 0.704 | 1.000 |
| Learning Rate | 0.01 | 0.711 | 0.464 | 0.677 | 1.000 |
| Learning Rate | 0.1 | 0.700 | 0.496 | 0.694 | 0.897 |
| Regularization | alpha 0.0001 | 0.710 | 0.431 | 0.704 | 1.000 |
| Regularization | alpha 0.01 | 0.692 | 0.426 | 0.696 | 0.999 |
| Regularization | alpha 0.1 | 0.701 | 0.442 | 0.682 | 0.996 |
| Regularization | alpha 1.0 | 0.757 | 0.490 | 0.756 | 0.860 |

**Key findings:**

1. **No complex model clearly beats a simple one.** After tuning, all four sensible models land within 0.01 of each other in cross-validated ROC-AUC (0.786 to 0.796). Logistic Regression ranks first, so it is selected. It is also the fastest, the easiest to explain, and the easiest to audit, which matters for credit decisions.
2. **The starting MLP overfits.** It reaches 99.6% train accuracy but only 70.6% test accuracy, which is below the 73.3% of always predicting "Low risk". Its training loss in `training_loss.png` keeps falling toward zero while it memorizes the 840 training rows.
3. **Credit score drives the decision.** CreditScore, PreviousDefault and Income carry almost all the signal. Several features add nothing measurable, so a leaner model could drop them.
4. **Accuracy hides the real trade-off.** Random Forest has a fair accuracy (0.753) but finds only 17% of defaulters at threshold 0.5. At its own tuned threshold it finds 67%.
5. **The threshold is a business decision.** For Logistic Regression, lowering it to 0.24 catches 68 of 96 defaulters instead of 46, but flags 74 good customers instead of 28. The right choice depends on the cost of a missed default versus a rejected good customer.
6. **Small differences are noise.** With 96 defaulters in the test set, gaps of about 0.01 are not meaningful. This is why the choice is made by cross-validation and not by one test score.

---

## Visualizations

- **Exploratory overview:** `outputs/figures/eda_overview.png`
- **Model comparison and ROC curves:** `outputs/figures/model_comparison.png`, `outputs/figures/roc_curves.png`
- **Feature importance:** `outputs/figures/feature_importance.png`
- **Confusion matrices:** `outputs/figures/confusion_matrix_final.png`, `outputs/figures/confusion_matrix_baseline_mlp.png`
- **MLP training loss and experiments:** `outputs/figures/training_loss.png`, `outputs/figures/mlp_hyperparameter_experiments.png`

Set `SHOW_PLOTS = True` at the top of the script to also open the plots in a window.

---

## Model Persistence with Joblib (Train Once, Use Many Times)

**The idea:** training takes time and needs the labelled dataset. Using the model should take milliseconds and should not need either. So the trained model is written to a file with `joblib.dump`, and any other program can bring it back with `joblib.load` and call `predict`.

```text
 TRAINING (run once)                          PREDICTION (run any time)
 loan_default_pipeline.py                     predict_loan_default.py
 ┌──────────────────────┐                     ┌──────────────────────────┐
 │ Loan_Default.csv     │                     │ new_applicants.csv       │
 │ tune + compare models│                     │ load model + metadata    │
 │ joblib.dump(model)   │ ──► .joblib  ──►    │ check columns/categories │
 │ write metadata.json  │ ──► .json    ──►    │ predict_proba            │
 └──────────────────────┘                     └──────────────────────────┘
```

**What is stored:**

- `outputs/loan_default_model.joblib`: the **whole pipeline** (fitted scaler, fitted encoder and the selected model). Because the preprocessing travels with the model, new applicants get exactly the same scaling and encoding as the training data.
- `outputs/model_metadata.json`: a small "model card" with the model name and settings, the expected feature columns, allowed values for `PreviousDefault` and `HomeOwnership`, the scikit-learn and Python versions used, training date, default threshold (0.5), the tuned threshold (0.24), and the test metrics.

**Why the metadata file matters:** the prediction script uses it to reject bad input (missing columns, empty values, unknown categories such as `yes` instead of `Yes`) and to warn if the installed scikit-learn version differs from the one that trained the model.

**Predict for new applicants (no training, no dataset needed):**

```bash
python predict_loan_default.py new_applicants.csv
```

The file name is optional: running `python predict_loan_default.py` on its own scores `new_applicants.csv`. Train first (`python loan_default_pipeline.py`) so the `outputs/` folder with the model exists.

```text
   DefaultProbability  Prediction  RiskLabel
0               0.010           0   Low risk
1               0.211           0   Low risk
2               0.974           1  High risk
3               0.454           0   Low risk
```

Use the tuned threshold to catch more defaulters (applicant 3 flips to High risk) and save the result:

```bash
python predict_loan_default.py new_applicants.csv --threshold 0.24 --output scored_applicants.csv
```

**Rules for using saved models:**

1. **Same library version:** load the model with the scikit-learn version that trained it (the script warns if they differ).
2. **Trusted files only:** Joblib files can run code when opened, so load only files you created yourself.
3. **Retrain when data changes:** a saved model does not learn. When new loan outcomes arrive, rerun `loan_default_pipeline.py` to create a new model file.
4. **Git:** `.joblib` files are in `.gitignore`. After cloning, run `python loan_default_pipeline.py` once to create the model.

---

## Sample Prediction (from Python code)

```python
import pandas as pd
from loan_default_pipeline import load_model, FEATURES

model = load_model("outputs/loan_default_model.joblib")

applicant = pd.DataFrame(
    [[34, 650000, 500000, 650, 8, 2, 22000, 48, "No", "Mortgage"]],
    columns=FEATURES,
)
probability = model.predict_proba(applicant)[0, 1]
print("Default probability:", round(probability, 3))
print("Prediction:", int(probability >= 0.5))  # 0 (Low risk) or 1 (High risk)
```

> **Disclaimer:** This is an educational project on a small dataset. It must not be used for real lending decisions. A real credit model would also need fairness checks (for example on `Age`), calibration, and regulatory review.

---

## Author

**Prathmesh Girme**
GitHub: [@Prathmesh-Girme](https://github.com/Prathmesh-Girme)
**Date:** 01/11/2026
