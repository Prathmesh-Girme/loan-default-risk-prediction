########################################################
# Loan Default Risk Prediction
#
# Compares five models with one shared pipeline and one fair process:
#   Logistic Regression, Random Forest, Gradient Boosting,
#   MLP (baseline from the assignment) and MLP (tuned)
#
# Pipeline : ColumnTransformer (scale + encode) -> model
# Target   : Default (0 = Low default risk, 1 = High default risk)
########################################################

########################################################
# Required Python Packages
########################################################
import json
import os
import platform
import warnings
from datetime import datetime

import joblib
import matplotlib
import numpy as np
import pandas as pd
import sklearn

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

########################################################
# Settings
########################################################

# Large networks may hit max_iter while the loss is still creeping down.
# That is expected here and is visible in the loss plot, so the warning is muted.
warnings.filterwarnings("ignore", category=ConvergenceWarning)

SHOW_PLOTS = False  # False: only save plots to files (works without a display)

if not SHOW_PLOTS:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402 - must come after backend selection

########################################################
# File Paths
########################################################

DATA_PATH = "Loan_Default.csv"
OUTPUT_DIR = "outputs"
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
MODEL_PATH = os.path.join(OUTPUT_DIR, "loan_default_model.joblib")
METADATA_PATH = os.path.join(OUTPUT_DIR, "model_metadata.json")
EXPERIMENTS_PATH = os.path.join(OUTPUT_DIR, "mlp_experiment_results.csv")
COMPARISON_PATH = os.path.join(OUTPUT_DIR, "model_comparison.csv")
METRICS_PATH = os.path.join(OUTPUT_DIR, "metrics.json")

########################################################
# Columns
########################################################

TARGET = "Default"
NUMERIC_FEATURES = [
    "Age", "Income", "LoanAmount", "CreditScore",
    "EmploymentYears", "ExistingLoans", "MonthlyDebt", "LoanTerm",
]
CATEGORICAL_FEATURES = ["PreviousDefault", "HomeOwnership"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
CLASS_LABELS = ["Low risk", "High risk"]
CLASS_NAMES = ["Low risk (0)", "High risk (1)"]

########################################################
# Model Settings
########################################################

RANDOM_STATE = 42
TRAIN_PERCENTAGE = 0.7
CV_FOLDS = 5
MODEL_STEP = "model"  # name of the final step in every pipeline

# Starting MLP given in the assignment
BASELINE_MLP_PARAMS = {
    "hidden_layer_sizes": (32, 16),
    "activation": "relu",
    "solver": "adam",
    "max_iter": 1000,
    "random_state": RANDOM_STATE,
}

# MLP experiments: one parameter is changed at a time, the rest stay at baseline.
MLP_EXPERIMENTS = {
    "Activation": ("activation", ["identity", "logistic", "tanh", "relu"]),
    "Hidden Layers": ("hidden_layer_sizes", [(10,), (20, 10), (50, 25), (100, 50, 25)]),
    "Learning Rate": ("learning_rate_init", [0.0001, 0.001, 0.01, 0.1]),
    "Regularization (extra)": ("alpha", [0.0001, 0.01, 0.1, 1.0]),
}

# Small, equal-effort search grids for the other algorithms
CANDIDATE_MODELS = {
    "Logistic Regression": (
        LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        {f"{MODEL_STEP}__C": [0.01, 0.1, 1, 10]},
    ),
    "Random Forest": (
        RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=1),
        {
            f"{MODEL_STEP}__max_depth": [4, 8, None],
            f"{MODEL_STEP}__min_samples_leaf": [1, 5, 10],
        },
    ),
    "Gradient Boosting": (
        GradientBoostingClassifier(random_state=RANDOM_STATE),
        {
            f"{MODEL_STEP}__n_estimators": [100, 200],
            f"{MODEL_STEP}__learning_rate": [0.03, 0.1],
            f"{MODEL_STEP}__max_depth": [2, 3],
        },
    ),
}

# 'identity' activation makes the network a linear model. It stays in the
# experiment as a diagnostic, but it is not allowed to be the tuned MLP.
LINEAR_ACTIVATION = "identity"

SCORING = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
    "roc_auc": "roc_auc",
}
CV_COLUMNS = ["CV_Accuracy", "CV_Precision", "CV_Recall", "CV_F1", "CV_ROC_AUC"]


########################################################
# Function name : read_data
# Description   : Read the loan dataset into a pandas dataframe
# Input         : Path of CSV file
# Output        : Dataframe
# Author        : Prathmesh Sunil Girme
# Date          : 01/11/2026
########################################################
def read_data(path):
    """Read the data into pandas dataframe"""
    return pd.read_csv(path)


########################################################
# Function name : validate_dataset
# Description   : Check that all required columns exist and the target is 0/1
# Input         : Dataframe
# Output        : Nothing (raises ValueError if the data is not usable)
# Author        : Prathmesh Sunil Girme
# Date          : 01/11/2026
########################################################
def validate_dataset(dataset):
    """Stop early with a clear message if the dataset is not as expected"""
    missing_columns = [c for c in FEATURES + [TARGET] if c not in dataset.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing columns: {missing_columns}")

    if set(dataset[TARGET].unique()) - {0, 1}:
        raise ValueError(f"Target '{TARGET}' must contain only 0 and 1")


########################################################
# Function name : dataset_statistics
# Description   : Display shape, data types, and basic statistics
# Input         : Dataframe
# Output        : Nothing (prints)
# Author        : Prathmesh Sunil Girme
# Date          : 01/11/2026
########################################################
def dataset_statistics(dataset):
    """Print basic stats and each feature's correlation with the target"""
    print("Dataset Shape ::", dataset.shape)
    print("\nData Types:\n", dataset.dtypes)
    print("\nStatistics:\n", dataset.describe(include="all").T)

    encoded = dataset.copy()
    encoded["PreviousDefault"] = (encoded["PreviousDefault"] == "Yes").astype(int)
    correlation = encoded[NUMERIC_FEATURES + ["PreviousDefault", TARGET]].corr()[TARGET]
    print("\nCorrelation with Default:\n", correlation.drop(TARGET).sort_values())

    print("\nDefault rate by HomeOwnership:\n", dataset.groupby("HomeOwnership")[TARGET].mean())
    print("\nDefault rate by PreviousDefault:\n", dataset.groupby("PreviousDefault")[TARGET].mean())


########################################################
# Function name : check_missing_values
# Description   : Count missing values and duplicate rows
# Input         : Dataframe
# Output        : Total missing values
# Author        : Prathmesh Sunil Girme
# Date          : 01/11/2026
########################################################
def check_missing_values(dataset):
    """Report missing values per column and duplicate rows"""
    missing = dataset.isnull().sum()
    print("Missing values per column:\n", missing)
    print("Total missing values ::", int(missing.sum()))
    print("Duplicate rows       ::", int(dataset.duplicated().sum()))
    return int(missing.sum())


########################################################
# Function name : check_class_balance
# Description   : Show how many customers are in each target class
# Input         : Dataframe
# Output        : Share of high-risk customers (0 to 1)
# Author        : Prathmesh Sunil Girme
# Date          : 01/11/2026
########################################################
def check_class_balance(dataset):
    """Print class counts and percentages"""
    counts = dataset[TARGET].value_counts().sort_index()
    shares = dataset[TARGET].value_counts(normalize=True).sort_index()

    for label in counts.index:
        print(f"Class {label} ({CLASS_LABELS[label]}) :: {counts[label]} rows ({shares[label]:.1%})")

    print(f"Majority-class baseline accuracy (always predict 0) :: {shares.max():.3f}")
    return float(shares.get(1, 0.0))


########################################################
# Function name : finish_plot
# Description   : Save the current figure (and show it if SHOW_PLOTS is True)
# Input         : File name
# Output        : Nothing (writes a PNG)
# Author        : Prathmesh Sunil Girme
# Date          : 01/11/2026
########################################################
def finish_plot(file_name):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, file_name)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    if SHOW_PLOTS:
        plt.show()
    plt.close()
    print(f"Plot saved to {path}")


########################################################
# Function name : plot_eda
# Description   : Exploratory analysis plots
# Input         : Dataframe
# Output        : outputs/figures/eda_overview.png
# Author        : Prathmesh Sunil Girme
# Date          : 01/11/2026
########################################################
def plot_eda(dataset):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    counts = dataset[TARGET].value_counts().sort_index()
    axes[0, 0].bar(CLASS_NAMES, counts.values, color=["#4c78a8", "#e45756"])
    for i, value in enumerate(counts.values):
        axes[0, 0].text(i, value, str(value), ha="center", va="bottom")
    axes[0, 0].set_title("Target Class Balance")
    axes[0, 0].set_ylabel("Customers")

    for ax, column in ((axes[0, 1], "PreviousDefault"), (axes[1, 0], "HomeOwnership")):
        rate = dataset.groupby(column)[TARGET].mean()
        ax.bar(rate.index, rate.values, color="#72b7b2")
        ax.axhline(dataset[TARGET].mean(), linestyle="--", color="gray", label="Overall rate")
        ax.set_title(f"Default Rate by {column}")
        ax.set_ylabel("Default rate")
        ax.legend()

    for label, group in dataset.groupby(TARGET):
        axes[1, 1].hist(group["CreditScore"], bins=20, alpha=0.6, label=CLASS_NAMES[label])
    axes[1, 1].set_title("Credit Score by Class")
    axes[1, 1].set_xlabel("CreditScore")
    axes[1, 1].legend()

    finish_plot("eda_overview.png")


########################################################
# Function name : split_dataset
# Description   : Split into train/test using stratified sampling
# Input         : Dataframe, train percentage
# Output        : train_x, test_x, train_y, test_y
# Author        : Prathmesh Sunil Girme
# Date          : 01/11/2026
########################################################
def split_dataset(dataset, train_percentage, random_state=RANDOM_STATE):
    """Stratified split so both sets keep the same default rate"""
    train_x, test_x, train_y, test_y = train_test_split(
        dataset[FEATURES],
        dataset[TARGET],
        train_size=train_percentage,
        random_state=random_state,
        stratify=dataset[TARGET],
    )
    return train_x, test_x, train_y, test_y


########################################################
# Function name : make_cv
# Description   : Cross-validation splitter shared by every comparison
# Input         : Nothing
# Output        : StratifiedKFold object
# Author        : Prathmesh Sunil Girme
# Date          : 01/11/2026
########################################################
def make_cv():
    return StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)


########################################################
# Function name : build_pipeline
# Description   : Build a Pipeline:
#                 ColumnTransformer : StandardScaler for numbers,
#                                     OneHotEncoder for categories
#                 model             : any scikit-learn classifier
# Input         : Classifier
# Output        : Untrained pipeline
# Author        : Prathmesh Sunil Girme
# Date          : 01/11/2026
########################################################
def build_pipeline(estimator):
    """Scaling and encoding live inside the pipeline, so they are fitted
    on training data only (no data leakage) and reused for new applicants."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(drop="if_binary", handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    return Pipeline(steps=[("preprocessor", preprocessor), (MODEL_STEP, estimator)])


########################################################
# Function name : build_mlp_pipeline
# Description   : Pipeline with an MLPClassifier (baseline plus any changes)
# Input         : Dictionary of MLP parameters to change
# Output        : Untrained pipeline
# Author        : Prathmesh Sunil Girme
# Date          : 01/11/2026
########################################################
def build_mlp_pipeline(mlp_params=None):
    params = dict(BASELINE_MLP_PARAMS)
    if mlp_params:
        params.update(mlp_params)
    return build_pipeline(MLPClassifier(**params))


########################################################
# Function name : train_pipeline
# Description   : Train a Pipeline
# Input         : Pipeline, training data
# Output        : Trained pipeline
# Author        : Prathmesh Sunil Girme
# Date          : 01/11/2026
########################################################
def train_pipeline(pipeline, train_x, train_y):
    pipeline.fit(train_x, train_y)
    return pipeline


########################################################
# Function name : evaluate_model
# Description   : Accuracy, precision, recall, F1, ROC-AUC, confusion matrix
# Input         : Model, features, labels, decision threshold
# Output        : Dictionary of metrics, predicted labels
# Author        : Prathmesh Sunil Girme
# Date          : 01/11/2026
########################################################
def evaluate_model(model, x, y, threshold=0.5):
    """Class 1 (High risk) is the positive class for precision/recall/F1"""
    probability = model.predict_proba(x)[:, 1]
    predictions = (probability >= threshold).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y, predictions)),
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probability)),
        "confusion_matrix": confusion_matrix(y, predictions).tolist(),
    }
    return metrics, predictions


########################################################
# Function name : report_model
# Description   : Print the full evaluation of one model on the test set
# Input         : Title, model, train and test data, threshold
# Output        : Test metrics dictionary, test predictions
# Author        : Prathmesh Sunil Girme
# Date          : 01/11/2026
########################################################
def report_model(title, model, train_x, train_y, test_x, test_y, threshold=0.5):
    train_metrics, _ = evaluate_model(model, train_x, train_y, threshold)
    test_metrics, predictions = evaluate_model(model, test_x, test_y, threshold)
    test_metrics["train_accuracy"] = train_metrics["accuracy"]
    test_metrics["threshold"] = float(threshold)

    print(f"\n===== {title} (threshold = {threshold:.2f}) =====")
    print("Train Accuracy ::", round(train_metrics["accuracy"], 4))
    print("Test Accuracy  ::", round(test_metrics["accuracy"], 4))
    print("Precision      ::", round(test_metrics["precision"], 4))
    print("Recall         ::", round(test_metrics["recall"], 4))
    print("F1-score       ::", round(test_metrics["f1"], 4))
    print("ROC-AUC        ::", round(test_metrics["roc_auc"], 4))
    print("Confusion Matrix:\n", confusion_matrix(test_y, predictions))
    print("Classification Report:\n",
          classification_report(test_y, predictions, target_names=CLASS_NAMES, zero_division=0))
    return test_metrics, predictions


########################################################
# Function name : plot_confusion_matrix
# Description   : Display Confusion Matrix
# Input         : True labels, predicted labels, title, file name
# Output        : PNG file
# Author        : Prathmesh Sunil Girme
# Date          : 01/11/2026
########################################################
def plot_confusion_matrix(y_true, y_pred, title, file_name):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    image = ax.matshow(cm, cmap="Blues")
    fig.colorbar(image)
    for (i, j), value in np.ndenumerate(cm):
        ax.text(j, i, str(value), ha="center", va="center", fontsize=13)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title, pad=40)
    finish_plot(file_name)


########################################################
# Function name : plot_training_loss
# Description   : Plot the training loss curve of one or more MLP pipelines
# Input         : Dictionary {name: trained MLP pipeline}, file name
# Output        : PNG file
# Author        : Prathmesh Sunil Girme
# Date          : 01/11/2026
########################################################
def plot_training_loss(models, file_name):
    plt.figure(figsize=(8, 4.5))
    for name, model in models.items():
        plt.plot(model.named_steps[MODEL_STEP].loss_curve_, label=name)
    plt.xlabel("Iteration (epoch)")
    plt.ylabel("Training loss")
    plt.title("MLP Training Loss")
    plt.legend()
    finish_plot(file_name)


########################################################
# Function name : run_mlp_experiment
# Description   : Cross-validate one MLP parameter at a time on TRAINING data
#                 only, so the test set is never used to choose a model
# Input         : Experiment name, parameter name (None = baseline), values,
#                 training data
# Output        : Dataframe with one row per value
# Author        : Prathmesh Sunil Girme
# Date          : 01/11/2026
########################################################
def run_mlp_experiment(experiment_name, parameter, values, train_x, train_y):
    rows = []

    for value in values:
        params = {parameter: value} if parameter else {}
        scores = cross_validate(
            build_mlp_pipeline(params), train_x, train_y,
            cv=make_cv(), scoring=SCORING, return_train_score=True, n_jobs=-1,
        )
        row = {
            "Experiment": experiment_name,
            "Parameter": parameter or "baseline",
            "Value": str(value),
            "CV_Accuracy": scores["test_accuracy"].mean(),
            "CV_Precision": scores["test_precision"].mean(),
            "CV_Recall": scores["test_recall"].mean(),
            "CV_F1": scores["test_f1"].mean(),
            "CV_ROC_AUC": scores["test_roc_auc"].mean(),
            "Train_Accuracy": scores["train_accuracy"].mean(),
            "Params": params,
        }
        rows.append(row)
        print(f"  {row['Parameter']} = {str(value):<14} "
              f"CV acc {row['CV_Accuracy']:.3f} | F1 {row['CV_F1']:.3f} | "
              f"AUC {row['CV_ROC_AUC']:.3f} | train acc {row['Train_Accuracy']:.3f}")

    return pd.DataFrame(rows)


########################################################
# Function name : plot_mlp_experiments
# Description   : Bar charts of cross-validated F1 and ROC-AUC per experiment
# Input         : Dataframe from run_mlp_experiment
# Output        : PNG file
# Author        : Prathmesh Sunil Girme
# Date          : 01/11/2026
########################################################
def plot_mlp_experiments(results):
    names = list(results["Experiment"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    for ax, name in zip(axes.ravel(), names):
        part = results[results["Experiment"] == name]
        positions = np.arange(len(part))
        ax.bar(positions - 0.2, part["CV_F1"], width=0.4, label="F1")
        ax.bar(positions + 0.2, part["CV_ROC_AUC"], width=0.4, label="ROC-AUC")
        ax.set_xticks(positions)
        ax.set_xticklabels(part["Value"], rotation=20)
        ax.set_ylim(0, 1)
        ax.set_title(f"{name} ({part['Parameter'].iloc[0]})")
        ax.legend()

    fig.suptitle("MLP Hyperparameter Experiments (5-fold CV on training data)")
    finish_plot("mlp_hyperparameter_experiments.png")


########################################################
# Function name : select_best_mlp_config
# Description   : Pick the MLP experiment row with the best CV ROC-AUC
#                 (ties are broken by CV F1). Linear (identity) networks are
#                 skipped because they are not a real multi-layer perceptron.
# Input         : Dataframe of MLP experiment results
# Output        : Best row
# Author        : Prathmesh Sunil Girme
# Date          : 01/11/2026
########################################################
def select_best_mlp_config(results):
    is_linear = (results["Parameter"] == "activation") & (results["Value"] == LINEAR_ACTIVATION)
    ranked = results[~is_linear].sort_values(["CV_ROC_AUC", "CV_F1"], ascending=False)
    return ranked.iloc[0]


########################################################
# Function name : tune_candidate
# Description   : Grid search one algorithm with 5-fold CV on TRAINING data,
#                 refit the best setting on the full training set
# Input         : Model name, classifier, parameter grid, training data
# Output        : Fitted best pipeline, dictionary of CV scores
# Author        : Prathmesh Sunil Girme
# Date          : 01/11/2026
########################################################
def tune_candidate(name, estimator, grid, train_x, train_y):
    search = GridSearchCV(
        build_pipeline(estimator), grid,
        scoring=SCORING, refit="roc_auc", cv=make_cv(), n_jobs=-1,
    )
    search.fit(train_x, train_y)

    best = search.best_index_
    results = search.cv_results_
    best_params = {key.split("__", 1)[1]: value for key, value in search.best_params_.items()}

    row = {
        "Model": name,
        "Settings": ", ".join(f"{k}={v}" for k, v in best_params.items()),
        "CV_Accuracy": results["mean_test_accuracy"][best],
        "CV_Precision": results["mean_test_precision"][best],
        "CV_Recall": results["mean_test_recall"][best],
        "CV_F1": results["mean_test_f1"][best],
        "CV_ROC_AUC": results["mean_test_roc_auc"][best],
    }
    print(f"  {name:<20} {row['Settings']:<48} "
          f"CV AUC {row['CV_ROC_AUC']:.3f} | F1 {row['CV_F1']:.3f}")
    return search.best_estimator_, row


########################################################
# Function name : tune_threshold
# Description   : Find the decision threshold with the best F1 using
#                 cross-validated predictions on TRAINING data only
# Input         : Pipeline (it is cloned, not modified), training data
# Output        : Threshold between 0.10 and 0.90
# Author        : Prathmesh Sunil Girme
# Date          : 01/11/2026
########################################################
def tune_threshold(pipeline, train_x, train_y):
    probability = cross_val_predict(
        clone(pipeline), train_x, train_y, cv=make_cv(), method="predict_proba", n_jobs=-1
    )[:, 1]

    thresholds = np.arange(0.10, 0.91, 0.01)
    scores = [f1_score(train_y, probability >= t, zero_division=0) for t in thresholds]
    return round(float(thresholds[int(np.argmax(scores))]), 2)


########################################################
# Function name : compare_models
# Description   : Evaluate every candidate on the test set ONCE, at the normal
#                 0.5 threshold and at its own cross-validated threshold
# Input         : Dictionary {name: fitted pipeline}, CV rows, train/test data
# Output        : Comparison dataframe
# Author        : Prathmesh Sunil Girme
# Date          : 01/11/2026
########################################################
def compare_models(models, cv_rows, train_x, train_y, test_x, test_y):
    rows = []
    for row in cv_rows:
        model = models[row["Model"]]
        at_default, _ = evaluate_model(model, test_x, test_y, 0.5)
        train_default, _ = evaluate_model(model, train_x, train_y, 0.5)
        threshold = tune_threshold(model, train_x, train_y)
        at_tuned, _ = evaluate_model(model, test_x, test_y, threshold)

        rows.append({
            **row,
            "Train_Accuracy": train_default["accuracy"],
            "Test_Accuracy": at_default["accuracy"],
            "Test_Precision": at_default["precision"],
            "Test_Recall": at_default["recall"],
            "Test_F1": at_default["f1"],
            "Test_ROC_AUC": at_default["roc_auc"],
            "Tuned_Threshold": threshold,
            "Test_Recall_At_Tuned": at_tuned["recall"],
            "Test_F1_At_Tuned": at_tuned["f1"],
        })
    return pd.DataFrame(rows)


########################################################
# Function name : select_best_model
# Description   : Choose the final model by CV ROC-AUC (ties: CV F1).
#                 The test set plays no part in this choice.
# Input         : Comparison dataframe
# Output        : Name of the best model
# Author        : Prathmesh Sunil Girme
# Date          : 01/11/2026
########################################################
def select_best_model(comparison):
    ranked = comparison.sort_values(["CV_ROC_AUC", "CV_F1"], ascending=False)
    return ranked.iloc[0]["Model"]


########################################################
# Function name : plot_model_comparison
# Description   : CV ROC-AUC next to test ROC-AUC for every model
# Input         : Comparison dataframe
# Output        : PNG file
# Author        : Prathmesh Sunil Girme
# Date          : 01/11/2026
########################################################
def plot_model_comparison(comparison):
    positions = np.arange(len(comparison))
    plt.figure(figsize=(10, 4.8))
    plt.bar(positions - 0.2, comparison["CV_ROC_AUC"], width=0.4, label="Cross-validation (train)")
    plt.bar(positions + 0.2, comparison["Test_ROC_AUC"], width=0.4, label="Test set")
    plt.xticks(positions, [m.replace(" (", "\n(") for m in comparison["Model"]], fontsize=8)
    plt.ylim(0.5, 0.9)
    plt.ylabel("ROC-AUC (0.5 = random guessing)")
    plt.title("Model Comparison: ROC-AUC")
    plt.legend()
    finish_plot("model_comparison.png")


########################################################
# Function name : plot_roc_curves
# Description   : ROC curves of all models on the test set
# Input         : Dictionary {name: fitted pipeline}, test data
# Output        : PNG file
# Author        : Prathmesh Sunil Girme
# Date          : 01/11/2026
########################################################
def plot_roc_curves(models, test_x, test_y):
    plt.figure(figsize=(6.5, 5.5))
    for name, model in models.items():
        probability = model.predict_proba(test_x)[:, 1]
        fpr, tpr, _ = roc_curve(test_y, probability)
        plt.plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(test_y, probability):.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guessing")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate (recall)")
    plt.title("ROC Curves (test set)")
    plt.legend(fontsize=7, loc="lower right")
    finish_plot("roc_curves.png")


########################################################
# Function name : plot_feature_importance
# Description   : Permutation importance: how much ROC-AUC drops when one
#                 feature is shuffled. Works for any model.
# Input         : Fitted pipeline, test data
# Output        : PNG file, importance series
# Author        : Prathmesh Sunil Girme
# Date          : 01/11/2026
########################################################
def plot_feature_importance(model, test_x, test_y, title):
    result = permutation_importance(
        model, test_x, test_y, scoring="roc_auc",
        n_repeats=20, random_state=RANDOM_STATE, n_jobs=-1,
    )
    order = np.argsort(result.importances_mean)
    names = [FEATURES[i] for i in order]

    plt.figure(figsize=(7, 4.8))
    plt.barh(names, result.importances_mean[order], xerr=result.importances_std[order])
    plt.xlabel("Drop in ROC-AUC when the feature is shuffled")
    plt.title(title)
    finish_plot("feature_importance.png")
    return pd.Series(result.importances_mean, index=FEATURES).sort_values(ascending=False)


########################################################
# Function name : predict_new_applicants
# Description   : Score new loan applicants (made-up examples for testing)
# Input         : Trained model, decision threshold
# Output        : Dataframe with default probability and risk label
# Author        : Prathmesh Sunil Girme
# Date          : 01/11/2026
########################################################
def predict_new_applicants(model, threshold=0.5):
    applicants = pd.DataFrame(
        [
            [45, 1200000, 300000, 790, 20, 1, 8000, 36, "No", "Own"],
            [34, 650000, 500000, 650, 8, 2, 22000, 48, "No", "Mortgage"],
            [23, 250000, 900000, 520, 1, 4, 45000, 60, "Yes", "Rent"],
        ],
        columns=FEATURES,
        index=["Applicant A (strong profile)", "Applicant B (average profile)",
               "Applicant C (weak profile)"],
    )
    probability = model.predict_proba(applicants)[:, 1]
    result = applicants.copy()
    result["DefaultProbability"] = probability.round(3)
    result["Prediction"] = np.where(probability >= threshold, "1 (High risk)", "0 (Low risk)")
    return result


########################################################
# Function name : save_model / load_model
# Description   : Save or load the trained Pipeline with joblib.
#                 Only load model files you created yourself.
# Input         : Model and path / path
# Output        : Nothing / model
# Author        : Prathmesh Sunil Girme
# Date          : 01/11/2026
########################################################
def save_model(model, path=MODEL_PATH):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved to {path}")


def load_model(path=MODEL_PATH):
    model = joblib.load(path)
    print(f"Model loaded from {path}")
    return model


########################################################
# Function name : json_default
# Description   : Let json.dump handle numpy numbers
# Author        : Prathmesh Sunil Girme
# Date          : 01/11/2026
########################################################
def json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f"Cannot save {type(value)} to JSON")


########################################################
# Function name : save_model_metadata
# Description   : Save a "model card" next to the .joblib file so the model can
#                 be used safely later: which columns it expects, which library
#                 versions built it, how well it performed, and which values
#                 each category column may contain.
# Input         : Model name, model, dataset, number of training rows,
#                 test metrics, tuned threshold, CV ROC-AUC
# Output        : outputs/model_metadata.json
# Author        : Prathmesh Sunil Girme
# Date          : 01/11/2026
########################################################
def save_model_metadata(model_name, model, dataset, train_rows, test_metrics,
                        tuned_threshold, cv_roc_auc, path=METADATA_PATH):
    estimator = model.named_steps[MODEL_STEP]
    simple_types = (int, float, str, bool, type(None))
    parameters = {
        key: value for key, value in estimator.get_params().items()
        if isinstance(value, simple_types)
    }
    metadata = {
        "model_name": model_name,
        "model_class": type(estimator).__name__,
        "model_parameters": parameters,
        "selection": {"rule": "best cross-validated ROC-AUC on training data",
                      "cv_roc_auc": cv_roc_auc},
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "python_version": platform.python_version(),
        "scikit_learn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "training_rows": int(train_rows),
        "features": FEATURES,
        "categories": {c: sorted(dataset[c].unique().tolist()) for c in CATEGORICAL_FEATURES},
        "target": TARGET,
        "class_labels": {"0": CLASS_LABELS[0], "1": CLASS_LABELS[1]},
        "decision_threshold": 0.5,
        "tuned_threshold_higher_recall": tuned_threshold,
        "test_metrics": test_metrics,
    }
    with open(path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, default=json_default)
    print(f"Model metadata saved to {path}")


########################################################
# Function name : main
# Description   : Main function from where execution starts
# Author        : Prathmesh Sunil Girme
# Date          : 01/11/2026
########################################################
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1) Load and understand the dataset
    dataset = read_data(DATA_PATH)
    validate_dataset(dataset)
    print("\n--- 1) Dataset ---")
    dataset_statistics(dataset)

    # 2) Missing values
    print("\n--- 2) Missing values ---")
    check_missing_values(dataset)

    # 3) Class balance
    print("\n--- 3) Class balance ---")
    check_class_balance(dataset)

    # 4) Exploratory plots
    print("\n--- 4) Exploratory analysis ---")
    plot_eda(dataset)

    # 5) Split (stratified, because the classes are imbalanced)
    print("\n--- 5) Train/test split ---")
    train_x, test_x, train_y, test_y = split_dataset(dataset, TRAIN_PERCENTAGE)
    print("Train_x Shape :: ", train_x.shape)
    print("Test_x Shape  :: ", test_x.shape)
    print(f"High-risk share  -> train: {train_y.mean():.3f} | test: {test_y.mean():.3f}")
    print("Stratified splitting is used: with only about a quarter of customers in the "
          "high-risk class, a random split could give the test set too few defaulters "
          "to measure recall reliably.")

    # 6) Baseline MLP exactly as given in the assignment
    print("\n--- 6) Baseline MLP ---")
    baseline_mlp = train_pipeline(build_mlp_pipeline(), train_x, train_y)
    encoded_names = baseline_mlp.named_steps["preprocessor"].get_feature_names_out()
    print("Model input columns after scaling and encoding ::", list(encoded_names))
    baseline_metrics, baseline_predictions = report_model(
        "Baseline MLP (32, 16) relu", baseline_mlp, train_x, train_y, test_x, test_y
    )
    plot_confusion_matrix(test_y, baseline_predictions,
                          "Baseline MLP - Confusion Matrix", "confusion_matrix_baseline_mlp.png")

    # 7) MLP hyperparameter experiments (cross-validation on training data only)
    print("\n--- 7) MLP hyperparameter experiments (5-fold CV, training data only) ---")
    all_results = [run_mlp_experiment("Baseline", None, ["(32, 16) relu lr=0.001"],
                                      train_x, train_y)]
    for experiment_name, (parameter, values) in MLP_EXPERIMENTS.items():
        print(f"\nExperiment: {experiment_name}")
        all_results.append(run_mlp_experiment(experiment_name, parameter, values,
                                              train_x, train_y))

    mlp_results = pd.concat(all_results, ignore_index=True)
    mlp_results.drop(columns="Params").round(4).to_csv(EXPERIMENTS_PATH, index=False)
    print(f"\nMLP experiment table saved to {EXPERIMENTS_PATH}")
    plot_mlp_experiments(mlp_results[mlp_results["Experiment"] != "Baseline"])

    best_mlp = select_best_mlp_config(mlp_results)
    tuned_mlp_name = f"MLP (tuned: {best_mlp['Parameter']}={best_mlp['Value']})"
    print(f"Best MLP configuration (CV ROC-AUC) :: {tuned_mlp_name} "
          f"(AUC {best_mlp['CV_ROC_AUC']:.3f})")
    tuned_mlp = train_pipeline(build_mlp_pipeline(best_mlp["Params"]), train_x, train_y)
    plot_training_loss(
        {"Baseline (32, 16) relu": baseline_mlp, tuned_mlp_name: tuned_mlp},
        "training_loss.png",
    )

    # 8) Tune the other algorithms with the same CV folds and the same effort
    print("\n--- 8) Tuning Logistic Regression, Random Forest, Gradient Boosting ---")
    models = {}
    cv_rows = []
    for name, (estimator, grid) in CANDIDATE_MODELS.items():
        models[name], row = tune_candidate(name, estimator, grid, train_x, train_y)
        cv_rows.append(row)

    baseline_row = mlp_results[mlp_results["Experiment"] == "Baseline"].iloc[0]
    baseline_name = "MLP (baseline 32-16 relu)"
    models[baseline_name] = baseline_mlp
    models[tuned_mlp_name] = tuned_mlp
    for name, source, settings in (
        (baseline_name, baseline_row, "hidden=(32, 16), activation=relu"),
        (tuned_mlp_name, best_mlp, f"{best_mlp['Parameter']}={best_mlp['Value']}"),
    ):
        cv_rows.append({"Model": name, "Settings": settings,
                        **{column: source[column] for column in CV_COLUMNS}})

    # 9) Compare all models on the test set (once) and choose the final model
    print("\n--- 9) Model comparison ---")
    comparison = compare_models(models, cv_rows, train_x, train_y, test_x, test_y)
    comparison.round(4).to_csv(COMPARISON_PATH, index=False)
    print(comparison[["Model", "CV_ROC_AUC", "CV_F1", "Train_Accuracy", "Test_Accuracy",
                      "Test_Precision", "Test_Recall", "Test_F1", "Test_ROC_AUC"]]
          .round(3).to_string(index=False))
    print(comparison[["Model", "Tuned_Threshold", "Test_Recall_At_Tuned", "Test_F1_At_Tuned"]]
          .round(3).to_string(index=False))
    print(f"\nComparison table saved to {COMPARISON_PATH}")
    plot_model_comparison(comparison)
    plot_roc_curves(models, test_x, test_y)

    final_name = select_best_model(comparison)
    final_model = models[final_name]
    final_row = comparison[comparison["Model"] == final_name].iloc[0]
    threshold = float(final_row["Tuned_Threshold"])
    print(f"\nFinal model (best CV ROC-AUC) :: {final_name} (CV AUC {final_row['CV_ROC_AUC']:.3f})")

    # 10) Detailed report for the final model
    final_metrics, final_predictions = report_model(
        f"FINAL: {final_name}", final_model, train_x, train_y, test_x, test_y
    )
    plot_confusion_matrix(test_y, final_predictions, f"{final_name} - Confusion Matrix",
                          "confusion_matrix_final.png")
    tuned_metrics, _ = report_model(
        f"FINAL: {final_name} + tuned threshold", final_model, train_x, train_y,
        test_x, test_y, threshold
    )

    # 11) Which features drive the decision?
    print("\n--- 11) Feature importance (permutation, test set) ---")
    importance = plot_feature_importance(
        final_model, test_x, test_y, f"Feature Importance - {final_name}"
    )
    print(importance.round(4).to_string())

    # 12) Test on new loan applicants
    print("\n--- 12) New loan applicants (illustrative examples) ---")
    print(predict_new_applicants(final_model).to_string())

    # 13) Save the model and metadata, load the model, and test a sample
    print("\n--- 13) Save and load ---")
    save_model(final_model, MODEL_PATH)
    save_model_metadata(final_name, final_model, dataset, len(train_x), final_metrics,
                        threshold, float(final_row["CV_ROC_AUC"]))
    loaded = load_model(MODEL_PATH)
    sample = test_x.iloc[[0]]
    print(f"Loaded model prediction for first test sample: {loaded.predict(sample)[0]}")

    summary = {
        "final_model": final_name,
        "selection_rule": "best cross-validated ROC-AUC on training data (test set not used)",
        "majority_class_accuracy": float(1 - test_y.mean()),
        "baseline_mlp_test": baseline_metrics,
        "final_model_test": final_metrics,
        "final_model_test_tuned_threshold": tuned_metrics,
        "feature_importance_roc_auc_drop": importance.to_dict(),
        "comparison": comparison.to_dict(orient="records"),
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, default=json_default)
    print(f"Metrics saved to {METRICS_PATH}")


########################################################
# Application starter
########################################################

if __name__ == "__main__":
    main()
