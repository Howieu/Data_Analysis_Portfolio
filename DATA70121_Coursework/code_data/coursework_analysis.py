# ===== Section 1: Imports & Settings =====
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

warnings.simplefilter(action="ignore", category=FutureWarning)
sns.set_style("whitegrid", {"axes.grid": False})

BASE_DIR = Path(__file__).resolve().parent
FIG_DIR = BASE_DIR / "Figures"
DATA_PATH = BASE_DIR / "MavenRail_cleaned2.csv"
PREDICTION_PATH = BASE_DIR / "ToPredict.csv"
RANDOM_STATE = 123
np.random.seed(RANDOM_STATE)

if not FIG_DIR.exists():
    FIG_DIR.mkdir(parents=True)
print(f"Figures directory: {FIG_DIR.resolve()}")

# ===== Section 2: Dataset Loading =====
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH.resolve()}")
rail_df = pd.read_csv(DATA_PATH)
print(f"Loaded dataset shape: {rail_df.shape}")

duplicate_count = rail_df.duplicated().sum()
print(f"Duplicate rows retained: {duplicate_count}")

critical_cols = [
    "Price",
    "Journey.Status",
    "Refund.Request",
    "Departure",
    "Scheduled.Arrival",
    "Actual.Arrival",
]
missing_summary = rail_df[critical_cols].isna().sum().to_frame(name="missing_count")
missing_summary["missing_pct"] = (
    missing_summary["missing_count"] / len(rail_df) * 100
).round(2)
print("Missingness summary for critical columns:\n", missing_summary)

rows_before = len(rail_df)
rail_df = rail_df.dropna(subset=["Price", "Journey.Status", "Refund.Request"])
print(
    f"Rows removed due to missing critical fields: {rows_before - len(rail_df)}"
)
print(f"Dataset after cleaning shape: {rail_df.shape}")

# ===== Section 3: Data Cleaning & Feature Engineering =====
datetime_cols = ["Departure", "Scheduled.Arrival", "Actual.Arrival"]
for col in datetime_cols:
    rail_df[col] = pd.to_datetime(rail_df[col], format="%Y-%m-%d %H:%M")

durations = rail_df["Actual.Arrival"] - rail_df["Scheduled.Arrival"]
delay_minutes = durations.dt.total_seconds() / 60
rail_df["DelayInMinutes"] = np.where(delay_minutes > 0, delay_minutes, np.nan)

print(f"Total journeys: {len(rail_df):,}")
print(f"Journeys on time or early: {(delay_minutes <= 0).sum():,}")
print(f"DelayInMinutes NA count: {rail_df['DelayInMinutes'].isna().sum():,}")

rail_df["ScheduledMinutes"] = (
    rail_df["Scheduled.Arrival"] - rail_df["Departure"]
).dt.total_seconds() / 60
rail_df["ActualMinutes"] = (
    rail_df["Actual.Arrival"] - rail_df["Departure"]
).dt.total_seconds() / 60
rail_df["DepartureHour"] = rail_df["Departure"].dt.hour
rail_df["DepartureDayOfWeek"] = rail_df["Departure"].dt.dayofweek

print("Summary of key numeric variables:")
print(
    rail_df[["Price", "ScheduledMinutes", "ActualMinutes"]]
    .describe()
    .round(2)
)

numeric_specs = [
    ("Price (£)", "Price"),
    ("Scheduled.Arrival (min)", "ScheduledMinutes"),
    ("Actual.Arrival (min)", "ActualMinutes"),
    ("Delay (min, if delayed)", "DelayInMinutes"),
]
stats_rows = []
for label, col in numeric_specs:
    series = rail_df[col].dropna()
    if series.empty:
        continue
    stats_rows.append(
        {
            "Variable": label,
            "Mean": series.mean(),
            "Median": series.median(),
            "SD": series.std(),
            "Min": series.min(),
            "Max": series.max(),
        }
    )
stats_df = pd.DataFrame(stats_rows)
stats_df[["Mean", "Median", "SD", "Min", "Max"]] = stats_df[
    ["Mean", "Median", "SD", "Min", "Max"]
].round(2)
print("\nDescriptive statistics used in LaTeX table:\n", stats_df)

categorical_mapping = {
    "Journey Status": "Journey.Status",
    "Refund Request": "Refund.Request",
    "Ticket Type": "Ticket.Type",
    "Ticket Class": "Ticket.Class",
}
print("\nCategorical distributions (%):")
for label, col in categorical_mapping.items():
    distribution = rail_df[col].value_counts(normalize=True).mul(100).round(1)
    formatted = ", ".join(f"{idx} ({pct}%)" for idx, pct in distribution.items())
    print(f"{label}: {formatted}")

# ===== Section 4: Exploratory Data Analysis =====
scatter_df = rail_df[["ScheduledMinutes", "Price"]].dropna()
plt.figure(figsize=(8, 6))
plt.scatter(scatter_df["ScheduledMinutes"], scatter_df["Price"], s=6, color="darkslateblue")
plt.xlabel("Scheduled travel time (minutes)")
plt.ylabel("Ticket price (£)")
plt.title("Scheduled minutes vs price")
plt.tight_layout()
plt.savefig(FIG_DIR / "eda_sched_vs_price.pdf", format="pdf")
plt.close()
print("Saved eda_sched_vs_price.pdf")

hour_counts = rail_df["DepartureHour"].value_counts().sort_index()
plt.figure(figsize=(8, 4))
plt.plot(hour_counts.index, hour_counts.values, marker="o", color="darkorange")
plt.xlabel("Departure hour")
plt.ylabel("Journeys")
plt.title("Hourly departure profile")
plt.xticks(range(0, 24, 2))
plt.tight_layout()
plt.savefig(FIG_DIR / "eda_hour_profile.pdf", format="pdf")
plt.close()
print("Saved eda_hour_profile.pdf")

delay_only = rail_df["DelayInMinutes"].dropna()
plt.figure(figsize=(8, 5))
plt.hist(delay_only, bins=40, color="firebrick", alpha=0.7)
plt.xlabel("Delay minutes")
plt.ylabel("Delayed journeys")
plt.title("Distribution of delays (DelayInMinutes)")
plt.tight_layout()
plt.savefig(FIG_DIR / "eda_delay_hist.pdf", format="pdf")
plt.close()
print("Saved eda_delay_hist.pdf")

status_counts = rail_df["Journey.Status"].value_counts()
plt.figure(figsize=(7, 5))
plt.bar(status_counts.index, status_counts.values, color="lightseagreen")
plt.xlabel("Journey status")
plt.ylabel("Journeys")
plt.title("Journey status distribution")
plt.tight_layout()
plt.savefig(FIG_DIR / "eda_status_counts.pdf", format="pdf")
plt.close()
print("Saved eda_status_counts.pdf")

numeric_cols = [
    "Price",
    "ScheduledMinutes",
    "ActualMinutes",
    "DelayInMinutes",
    "DepartureHour",
    "DepartureDayOfWeek",
]
pairplot_df = rail_df[numeric_cols].dropna()
plot_mat = pairplot_df.values
if plot_mat.shape[0] > 6000:
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(plot_mat.shape[0], size=6000, replace=False)
    plot_mat = plot_mat[idx]

labels = [
    "Price",
    "Scheduled minutes",
    "Actual minutes",
    "Delay minutes",
    "Departure hour",
    "Departure weekday",
]
n = len(labels)
plt.figure(figsize=(14, 14))
for i in range(n):
    for j in range(n):
        plt.subplot(n, n, 1 + i + (n * j))
        if i == j:
            sns.histplot(plot_mat[:, i], stat="density", color="lightseagreen", alpha=0.6)
            sns.kdeplot(plot_mat[:, i], color="black")
        else:
            plt.scatter(plot_mat[:, i], plot_mat[:, j], s=4, alpha=0.25, color="tab:blue")
        if j == n - 1:
            plt.xlabel(labels[i])
        else:
            plt.xlabel("")
        if i == 0:
            plt.ylabel(labels[j])
        else:
            plt.ylabel("")
plt.tight_layout()
plt.savefig(FIG_DIR / "eda_pairplot_custom.pdf", format="pdf")
plt.close()
print("Saved eda_pairplot_custom.pdf")

# ===== Section 5: Single Variable Logistic Regression =====
late_df = rail_df[rail_df["Journey.Status"] != "On Time"].copy()
late_df = late_df.dropna(subset=["Refund.Request"])
print(f"Non-punctual journeys for single-variable model: {len(late_df):,}")

late_df["RefundBinary"] = np.where(
    late_df["Refund.Request"].str.strip().str.lower() == "yes", 1, 0
)
late_df["MediumPrice"] = np.where(
    (late_df["Price"] > 10) & (late_df["Price"] <= 30), 1, 0
)

X_single = sm.add_constant(late_df["MediumPrice"])
logit_model_single = sm.Logit(late_df["RefundBinary"], X_single)
logit_result_single = logit_model_single.fit(disp=False)
print(logit_result_single.summary2())

coef_table_single = pd.DataFrame(
    {
        "Coefficient": logit_result_single.params,
        "Std.Err": logit_result_single.bse,
        "p-value": logit_result_single.pvalues,
        "Odds Ratio": np.exp(logit_result_single.params),
    }
).round(3)
print("\nSingle-variable logistic regression coefficients:\n", coef_table_single)

beta0 = logit_result_single.params["const"]
beta1 = logit_result_single.params["MediumPrice"]
prob_table = pd.DataFrame(
    {
        "Price (£)": [5, 25],
        "MediumPrice": [0, 1],
        "Predicted Refund Probability": [
            1 / (1 + np.exp(-beta0)),
            1 / (1 + np.exp(-(beta0 + beta1))),
        ],
    }
)
print("\nReference refund probabilities:\n", prob_table.round(3))

# ===== Section 6: Multivariable Models =====
FEATURE_COLS = [
    "Price",
    "DelayInMinutes_filled",
    "Journey.Status",
    "Ticket.Type",
    "Ticket.Class",
    "Payment.Method",
    "Railcard",
    "DepartureHour",
    "DepartureDayOfWeek",
]
CATEGORICAL_FEATURES = [
    "Journey.Status",
    "Ticket.Type",
    "Ticket.Class",
    "Payment.Method",
    "Railcard",
]

rail_df["RefundBinary"] = (rail_df["Refund.Request"] == "Yes").astype(int)
rail_df["DelayInMinutes_filled"] = rail_df["DelayInMinutes"].fillna(0)

X_raw = rail_df[FEATURE_COLS].copy()
X_raw[CATEGORICAL_FEATURES] = X_raw[CATEGORICAL_FEATURES].fillna("Unknown")
X_raw["DepartureHour"] = X_raw["DepartureHour"].fillna(
    X_raw["DepartureHour"].median()
)
X_raw["DepartureDayOfWeek"] = X_raw["DepartureDayOfWeek"].fillna(
    X_raw["DepartureDayOfWeek"].median()
)

X = pd.get_dummies(X_raw, columns=CATEGORICAL_FEATURES, drop_first=True)
y = rail_df["RefundBinary"]
print(f"Feature matrix shape: {X.shape}")
print(f"Refund request rate: {y.mean():.3f}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(
    f"Training rows: {X_train.shape[0]:,}, Testing rows: {X_test.shape[0]:,}"
)

models = [
    (
        "Logistic Regression",
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
    ),
    (
        "Decision Tree",
        DecisionTreeClassifier(max_depth=4, class_weight="balanced", random_state=RANDOM_STATE),
    ),
    (
        "Random Forest",
        RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    ),
]

model_results = []
for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_proba),
        "PR-AUC": average_precision_score(y_test, y_proba),
        "Probas": y_proba,
        "Estimator": model,
    }
    model_results.append(metrics)

results_df = pd.DataFrame(model_results)[
    ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]
]
print("\nTest set metrics:\n", results_df.round(3).to_string(index=False))

plt.figure(figsize=(7, 6))
for res in model_results:
    fpr, tpr, _ = roc_curve(y_test, res["Probas"])
    plt.plot(fpr, tpr, label=f"{res['Model']} (AUC = {res['ROC-AUC']:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves: Refund Prediction")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(FIG_DIR / "model_roc_comparison.pdf", format="pdf")
plt.close()
print("Saved model_roc_comparison.pdf")

baseline = y_test.mean()
plt.figure(figsize=(7, 6))
for res in model_results:
    precision, recall, _ = precision_recall_curve(y_test, res["Probas"])
    plt.plot(recall, precision, label=f"{res['Model']} (AP = {res['PR-AUC']:.3f})")
plt.hlines(baseline, xmin=0, xmax=1, linestyles="--", colors="gray", label=f"Baseline (AP = {baseline:.3f})")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curves: Refund Prediction")
plt.legend(loc="best")
plt.tight_layout()
plt.savefig(FIG_DIR / "model_pr_comparison.pdf", format="pdf")
plt.close()
print("Saved model_pr_comparison.pdf")

best_by_auc = results_df.loc[results_df["ROC-AUC"].idxmax()]
logit_auc = results_df.loc[
    results_df["Model"] == "Logistic Regression", "ROC-AUC"
].iloc[0]
if (best_by_auc["Model"] != "Logistic Regression") and (
    best_by_auc["ROC-AUC"] - logit_auc > 0.01
):
    final_model_name = best_by_auc["Model"]
else:
    final_model_name = "Logistic Regression"
final_model = next(res["Estimator"] for res in model_results if res["Model"] == final_model_name)
final_model.fit(X_train, y_train)
final_feature_columns = X.columns.tolist()
print(f"Final model selected: {final_model_name}")

# ===== Section 7: Final Outputs =====
if not PREDICTION_PATH.exists():
    raise FileNotFoundError(
        f"Prediction dataset not found at {PREDICTION_PATH.resolve()}"
    )
df_pred = pd.read_csv(PREDICTION_PATH)

time_cols = ["Departure", "Scheduled.Arrival", "Actual.Arrival"]
for col in time_cols:
    if col in df_pred.columns:
        df_pred[col] = pd.to_datetime(
            df_pred[col], format="%Y-%m-%d %H:%M", errors="coerce"
        )

if "DelayInMinutes" not in df_pred.columns:
    delay_minutes_pred = (
        df_pred["Actual.Arrival"] - df_pred["Scheduled.Arrival"]
    ).dt.total_seconds() / 60
    df_pred["DelayInMinutes"] = delay_minutes_pred.where(delay_minutes_pred > 0)

if "DepartureHour" not in df_pred.columns:
    df_pred["DepartureHour"] = df_pred["Departure"].dt.hour
if "DepartureDayOfWeek" not in df_pred.columns:
    df_pred["DepartureDayOfWeek"] = df_pred["Departure"].dt.dayofweek

df_pred["DelayInMinutes_filled"] = df_pred["DelayInMinutes"].fillna(0)
df_pred["DepartureHour"] = df_pred["DepartureHour"].fillna(
    df_pred["DepartureHour"].median()
)
df_pred["DepartureDayOfWeek"] = df_pred["DepartureDayOfWeek"].fillna(
    df_pred["DepartureDayOfWeek"].median()
)

missing_cols = set(FEATURE_COLS) - set(df_pred.columns)
if missing_cols:
    raise ValueError(
        f"ToPredict dataset is missing required columns: {sorted(missing_cols)}"
    )

X_pred_raw = df_pred[FEATURE_COLS].copy()
X_pred_raw[CATEGORICAL_FEATURES] = X_pred_raw[CATEGORICAL_FEATURES].fillna("Unknown")
X_pred = pd.get_dummies(X_pred_raw, columns=CATEGORICAL_FEATURES, drop_first=True)
for col in final_feature_columns:
    if col not in X_pred.columns:
        X_pred[col] = 0
X_pred = X_pred[final_feature_columns]

pred_proba = final_model.predict_proba(X_pred)[:, 1]
df_pred["PredRefundProb"] = pred_proba

pred_display_cols = [
    "Price",
    "Journey.Status",
    "Ticket.Type",
    "Ticket.Class",
    "Payment.Method",
    "PredRefundProb",
]
print(
    "\nPredicted refund probabilities (decimal):\n",
    df_pred[pred_display_cols].round({"PredRefundProb": 4}).to_string(index=False),
)
percent_view = df_pred[pred_display_cols[:-1]].copy()
percent_view["Predicted Probability (%)"] = (
    df_pred["PredRefundProb"] * 100
).round(1)
print(
    "\nPredicted refund probabilities (%):\n",
    percent_view.to_string(index=False),
)
