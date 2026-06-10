"""
Student Performance Analytics System
Machine Learning Pipeline
Usage: python student_ml_model.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score)
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")


# ── 1. Load Data ──────────────────────────────────────────────────────────────
def load_data(filepath="student_analytics_dataset.xlsx"):
    df = pd.read_excel(filepath, sheet_name="Student Data", header=1)
    df.columns = df.columns.str.strip()
    return df


# ── 2. Preprocess ─────────────────────────────────────────────────────────────
def preprocess(df):
    subjects = ["Maths", "Science", "English", "History", "Computer"]
    feature_cols = ["Attendance (%)"] + subjects

    df["Weak_Subject_Score"] = df[subjects].min(axis=1)
    df["Subject_Range"] = df[subjects].max(axis=1) - df[subjects].min(axis=1)
    df["Attendance_Band"] = pd.cut(
        df["Attendance (%)"], bins=[0, 60, 75, 90, 100],
        labels=[0, 1, 2, 3]).astype(int)

    feature_cols += ["Weak_Subject_Score", "Subject_Range", "Attendance_Band"]

    X = df[feature_cols]
    y = df["Pass/Fail (1=Pass)"]
    return X, y, df


# ── 3. Attendance vs Marks Correlation ───────────────────────────────────────
def attendance_correlation(df):
    subjects = ["Maths", "Science", "English", "History", "Computer"]
    print("\n── Attendance vs Marks Correlation ──────────────────────────────")
    corr_matrix = df[["Attendance (%)"] + subjects + ["Average Marks"]].corr()
    att_corr = corr_matrix["Attendance (%)"].drop("Attendance (%)")
    for col, val in att_corr.items():
        strength = "Strong" if abs(val) > 0.6 else "Moderate" if abs(val) > 0.3 else "Weak"
        print(f"  {col:<20} r = {val:.3f}  ({strength})")
    return att_corr


# ── 4. Train Models ───────────────────────────────────────────────────────────
def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=6, random_state=42, class_weight="balanced"),
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(random_state=42, max_iter=500, class_weight="balanced"))
        ])
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        results[name] = {
            "model": model,
            "accuracy": accuracy_score(y_test, y_pred),
            "auc": roc_auc_score(y_test, y_prob),
            "report": classification_report(y_test, y_pred, target_names=["Fail","Pass"]),
            "confusion": confusion_matrix(y_test, y_pred),
            "X_test": X_test, "y_test": y_test, "y_pred": y_pred
        }

    return results, X_train, X_test, y_train, y_test


# ── 5. Feature Importance ─────────────────────────────────────────────────────
def feature_importance(rf_model, feature_names):
    clf = rf_model if hasattr(rf_model, "feature_importances_") else rf_model.named_steps["clf"]
    importances = clf.feature_importances_
    fi = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    print("\n── Feature Importance (Random Forest) ───────────────────────────")
    for feat, imp in fi.items():
        bar = "█" * int(imp * 40)
        print(f"  {feat:<25} {imp:.3f}  {bar}")
    return fi


# ── 6. Identify At-Risk Students ──────────────────────────────────────────────
def identify_risk(df, rf_model, feature_cols):
    X_all = df[feature_cols]
    df = df.copy()
    df["Pass_Probability"] = rf_model.predict_proba(X_all)[:, 1] * 100
    df["ML_Prediction"] = rf_model.predict(X_all)
    df["ML_Prediction_Label"] = df["ML_Prediction"].map({1: "Pass", 0: "Fail"})

    at_risk = df[df["Risk Level"] == "High"].sort_values("Pass_Probability")
    print(f"\n── At-Risk Students (High Risk) — {len(at_risk)} students ──────────────")
    print(f"  {'Name':<22} {'Attend':>7} {'AvgMark':>8} {'Pass Prob':>10} {'Weakest Subject'}")
    subjects = ["Maths", "Science", "English", "History", "Computer"]
    for _, row in at_risk.iterrows():
        weakest = min(subjects, key=lambda s: row[s])
        print(f"  {row['Name']:<22} {row['Attendance (%)']:>6.1f}% "
              f"{row['Average Marks']:>7.1f}  "
              f"{row['Pass_Probability']:>8.1f}%  {weakest} ({row[weakest]:.0f})")
    return df


# ── 7. Suggest Improvement ────────────────────────────────────────────────────
def suggest_improvement(df):
    subjects = ["Maths", "Science", "English", "History", "Computer"]
    print("\n── Personalised Improvement Suggestions (sample — 10 students) ──")
    sample = df[df["Risk Level"].isin(["High","Medium"])].head(10)
    for _, row in sample.iterrows():
        suggestions = []
        if row["Attendance (%)"] < 75:
            suggestions.append(f"Improve attendance (currently {row['Attendance (%)']:.0f}%)")
        weak_subs = [s for s in subjects if row[s] < 50]
        if weak_subs:
            suggestions.append(f"Revise: {', '.join(weak_subs)}")
        if not suggestions:
            suggestions.append("Keep up current effort")
        print(f"  {row['Name']:<22} → {'; '.join(suggestions)}")


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  STUDENT PERFORMANCE ANALYTICS — ML PIPELINE")
    print("=" * 60)

    df = load_data()
    print(f"\nLoaded {len(df)} student records | Columns: {list(df.columns)}")

    X, y, df = preprocess(df)
    feature_cols = list(X.columns)

    att_corr = attendance_correlation(df)

    results, X_train, X_test, y_train, y_test = train_models(X, y)

    print("\n── Model Performance ─────────────────────────────────────────────")
    for name, res in results.items():
        print(f"\n  {name}")
        print(f"    Accuracy : {res['accuracy']*100:.1f}%")
        print(f"    AUC-ROC  : {res['auc']:.3f}")
        print(f"    Confusion Matrix:\n{res['confusion']}")
        print(f"\n    Classification Report:\n{res['report']}")

    rf_model = results["Random Forest"]["model"]
    feature_importance(rf_model, feature_cols)

    df = identify_risk(df, rf_model, feature_cols)
    suggest_improvement(df)

    print("\n✓ Pipeline complete.")
