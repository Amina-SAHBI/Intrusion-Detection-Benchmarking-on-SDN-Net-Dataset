#!/usr/bin/env python3
"""
sgd_binary.py

Stochastic Gradient Descent (Linear classifier) 7-variant benchmarking script
for binary intrusion detection on the SDN-Net dataset.

Why 7 models?
 → Each model tests a different data balancing + feature selection strategy
    to analyze robustness, class-imbalance impact, convergence, and ranking stability.

Why many ML / DL algorithms?
 → Network intrusion behavior is complex and heterogeneous. No single model is
    universally optimal. Evaluating diverse algorithms prevents biased conclusions,
    reveals generalization limits, and identifies the best trade-off for SDN flows.

Variants:
 1) Unbalanced
 2) Unbalanced + RF Feature Selection (RF-FS)
 3) Unbalanced + Information Gain FS (IG-FS)
 4) SMOTE
 5) SMOTE + RF-FS
 6) SMOTE + IG-FS
 7) SMOTE-Tomek
"""
import argparse
from pathlib import Path
from datetime import datetime
import time
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc
)

from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.utils.class_weight import compute_class_weight

from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek

RND = 42
sns.set(style="whitegrid")


def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(d):
    Path(d).mkdir(parents=True, exist_ok=True)


def rf_feature_importances(X_train, y_train):
    """Compute RF feature importance for selection."""
    rf = RandomForestClassifier(n_estimators=200, random_state=RND, n_jobs=-1)
    rf.fit(X_train, y_train)
    return pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)


def rf_feature_selection(X_train, y_train, k):
    """Return top-K RF-ranked features and the full importance series."""
    feats = rf_feature_importances(X_train, y_train)
    return list(feats.index[:k]), feats


def ig_feature_selection_with_scores(X_train, y_train, k):
    """Return top-K IG-ranked features and full scores."""
    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    selector.fit(X_train, y_train)
    ig = pd.Series(selector.scores_, index=X_train.columns).sort_values(ascending=False)
    return list(ig.index[:k]), ig


def plot_fs_comparison(rf_scores, ig_scores, top_k=20, suffix=""):
    """Plot RF importance vs IG ranking overlap."""
    rf_top = rf_scores.head(top_k)
    ig_top = ig_scores.head(top_k)
    shared = set(rf_top.index) & set(ig_top.index)

    plt.figure(figsize=(10, 5))
    rf_top.sort_values().plot.barh(color='tab:blue')
    plt.title(f"RF Feature Importances {suffix}")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    ig_top.sort_values().plot.barh(color='tab:green')
    plt.title(f"IG Feature Scores {suffix}")
    plt.xlabel("Mutual Info Score")
    plt.tight_layout()
    plt.show()

    print(f"Top {top_k} overlap (RF ∩ IG): {len(shared)}")
    if shared:
        print("Shared top-ranked features:", sorted(shared))


def plot_learning_curve(clf, X, y, suffix=""):
    """Plot learning curve using accuracy."""
    try:
        sizes, train_scores, val_scores = learning_curve(
            clf, X, y, cv=5, scoring='accuracy', n_jobs=-1,
            train_sizes=[0.1, 0.3, 0.5, 0.7, 1.0]
        )
        plt.figure(figsize=(7, 5))
        plt.plot(sizes, np.mean(train_scores, 1), 'o-', label="Train")
        plt.plot(sizes, np.mean(val_scores, 1), 'o-', label="Validation")
        plt.title(f"Learning Curve – SGD {suffix}")
        plt.xlabel("Training examples")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print("Unable to plot learning curve:", e)


def run_sgd(name, X_train, X_test, y_train, y_test, *,
            loss='log_loss', penalty='l2', alpha=1e-4, max_iter=1000,
            class_weight=None, algo="SGD", save=False, outdir="outputs"):
    """Run one experiment using SGDClassifier and return metrics dict."""
    stamp = timestamp()
    model_dir = Path(outdir) / f"{name.replace(' ', '_')}_{stamp}"
    if save:
        ensure_dir(model_dir)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    clf = SGDClassifier(loss=loss, penalty=penalty, alpha=alpha, max_iter=max_iter, random_state=RND, class_weight=class_weight)
    clf.fit(X_train_sc, y_train)
    y_pred = clf.predict(X_test_sc)

    # ROC score if possible
    score = None
    if hasattr(clf, "predict_proba"):
        try:
            score = clf.predict_proba(X_test_sc)[:, 1]
        except Exception:
            score = None
    if score is None and hasattr(clf, "decision_function"):
        try:
            score = clf.decision_function(X_test_sc)
        except Exception:
            score = None
    if score is None:
        score = y_pred

    acc = accuracy_score(y_test, y_pred)
    pr = precision_score(y_test, y_pred, zero_division=0)
    rc = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    report = classification_report(y_test, y_pred, zero_division=0)

    print(f"\n=== {name} ({algo}) ===")
    print(f"Accuracy: {acc:.4f}  Precision: {pr:.4f}  Recall: {rc:.4f}  F1: {f1:.4f}")
    print("\nClassification report:\n", report)

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=[0, 1], yticklabels=[0, 1], cbar=False, ax=ax)
    ax.set_title(f"Confusion Matrix - {name} ({algo})")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.tight_layout(); plt.show()

    auc_score = None
    try:
        fpr, tpr, _ = roc_curve(y_test, score)
        auc_score = auc(fpr, tpr)
        plt.figure(figsize=(6, 4))
        plt.plot(fpr, tpr, label=f"AUC={auc_score:.3f}")
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.title(f"ROC - {name} ({algo})"); plt.legend(); plt.tight_layout(); plt.show()
    except Exception:
        print("ROC unavailable for this classifier/output.")

    plot_learning_curve(SGDClassifier(loss=loss, penalty=penalty, alpha=alpha, random_state=RND), X_train, y_train, suffix=f"({name})")

    # Save artifacts
    if save:
        try:
            joblib.dump(clf, model_dir / f"{algo}_model_{stamp}.joblib")
            joblib.dump(scaler, model_dir / f"scaler_{stamp}.joblib")
        except Exception as e:
            print("Unable to save model/scaler:", e)

    return {"config": name, "algorithm": algo, "loss": loss, "penalty": penalty, "alpha": alpha, "max_iter": int(max_iter), "accuracy": acc, "precision": pr, "recall": rc, "f1": f1, "auc": auc_score}


def load_and_prepare(csv_path):
    """Load CSV and prepare X, y (binary labels 0=normal,1=attack)."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path, skipinitialspace=True, low_memory=False)
    # Drop unnamed index columns
    unnamed = [c for c in df.columns if c.startswith("Unnamed")]
    if unnamed:
        df.drop(columns=unnamed, inplace=True)
    # Attack Type column
    if "Attack Type" not in df.columns:
        if "Class" not in df.columns:
            raise KeyError("CSV must contain 'Attack Type' or 'Class'.")
        df["Attack Type"] = df["Class"].astype(str).str.strip()
    df["Attack_Mapped"] = df["Attack Type"].astype(str).str.strip().str.upper().map(lambda x: "NORMAL" if x == "NORMAL" else "ATTACK")
    df["Attack_Binary_Label"] = df["Attack_Mapped"].map({"NORMAL": 0, "ATTACK": 1}).astype(int)

    exclude = {"Class", "Attack Type", "Attack_Mapped", "Attack_Binary_Label"}
    obj_cols = [c for c in df.select_dtypes(include=["object", "category"]).columns if c not in exclude]
    if obj_cols:
        df = pd.get_dummies(df, columns=obj_cols, drop_first=True)

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
        df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

    drop_cols = [c for c in ["Class", "Attack Type", "Attack_Mapped", "Attack_Binary_Label"] if c in df.columns]
    X = df.drop(columns=drop_cols, errors="ignore")
    y = df["Attack_Binary_Label"].astype(int)
    return X, y


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="../SDN-Net.csv")
    parser.add_argument("--outdir", default="outputs")
    parser.add_argument("--k", type=int, default=30)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--loss", default="log_loss", help="SGD loss (e.g. log_loss)")
    parser.add_argument("--penalty", default="l2", help="penalty (l2, l1, elasticnet)")
    parser.add_argument("--alpha", type=float, default=1e-4, help="regularization strength")
    parser.add_argument("--max_iter", type=int, default=1000)
    parser.add_argument("--class_weight", action="store_true", help="Use class_weight='balanced' in SGD")
    args = parser.parse_args(argv)

    X_full, y_full = load_and_prepare(args.csv)
    print("Dataset shape:", X_full.shape)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_full, y_full, test_size=0.33, random_state=RND, stratify=y_full
    )
    print("Train / Test shapes:", X_train_raw.shape, X_test_raw.shape)

    results = []

    # Model 1: Unbalanced full
    cw = "balanced" if args.class_weight else None
    results.append(run_sgd("Model1_Unbalanced_Full", X_train_raw, X_test_raw, y_train, y_test,
                           loss=args.loss, penalty=args.penalty, alpha=args.alpha, max_iter=args.max_iter,
                           class_weight=cw, save=args.save, outdir=args.outdir))

    # Model 2: Unbalanced + RF-FS
    sel2, rf_scores2 = rf_feature_selection(X_train_raw, y_train, args.k)
    if args.save:
        ensure_dir(args.outdir)
        pd.Series(sel2, name="feature").to_csv(Path(args.outdir) / f"features_model2_{timestamp()}.csv", index=False)
    results.append(run_sgd("Model2_Unbalanced_RF-FS", X_train_raw[sel2], X_test_raw[sel2], y_train, y_test,
                           loss=args.loss, penalty=args.penalty, alpha=args.alpha, max_iter=args.max_iter,
                           class_weight=cw, save=args.save, outdir=args.outdir))

    # Model 3: Unbalanced + IG-FS
    sel3, ig_scores3 = ig_feature_selection_with_scores(X_train_raw, y_train, args.k)
    if args.save:
        pd.Series(sel3, name="feature").to_csv(Path(args.outdir) / f"features_model3_{timestamp()}.csv", index=False)
    results.append(run_sgd("Model3_Unbalanced_IG-FS", X_train_raw[sel3], X_test_raw[sel3], y_train, y_test,
                           loss=args.loss, penalty=args.penalty, alpha=args.alpha, max_iter=args.max_iter,
                           class_weight=cw, save=args.save, outdir=args.outdir))

    # Model 4: SMOTE balanced (full)
    smote = SMOTE(random_state=RND)
    X4_arr, y4 = smote.fit_resample(X_train_raw, y_train)
    X_train_m4 = pd.DataFrame(X4_arr, columns=X_train_raw.columns)
    results.append(run_sgd("Model4_SMOTE_Full", X_train_m4, X_test_raw, y4, y_test,
                           loss=args.loss, penalty=args.penalty, alpha=args.alpha, max_iter=args.max_iter,
                           class_weight=None, save=args.save, outdir=args.outdir))

    # Model 5: SMOTE + RF-FS
    sel5, rf_scores5 = rf_feature_selection(X_train_m4, y4, args.k)
    if args.save:
        pd.Series(sel5, name="feature").to_csv(Path(args.outdir) / f"features_model5_{timestamp()}.csv", index=False)
    results.append(run_sgd("Model5_SMOTE_RF-FS", X_train_m4[sel5], X_test_raw[sel5], y4, y_test,
                           loss=args.loss, penalty=args.penalty, alpha=args.alpha, max_iter=args.max_iter,
                           class_weight=None, save=args.save, outdir=args.outdir))

    # Model 6: SMOTE + IG-FS
    sel6, ig_scores6 = ig_feature_selection_with_scores(X_train_m4, y4, args.k)
    if args.save:
        pd.Series(sel6, name="feature").to_csv(Path(args.outdir) / f"features_model6_{timestamp()}.csv", index=False)
    results.append(run_sgd("Model6_SMOTE_IG-FS", X_train_m4[sel6], X_test_raw[sel6], y4, y_test,
                           loss=args.loss, penalty=args.penalty, alpha=args.alpha, max_iter=args.max_iter,
                           class_weight=None, save=args.save, outdir=args.outdir))

    # Model 7: SMOTE-Tomek balanced (full)
    smt = SMOTETomek(random_state=RND)
    X7_arr, y7 = smt.fit_resample(X_train_raw, y_train)
    X_train_m7 = pd.DataFrame(X7_arr, columns=X_train_raw.columns)
    results.append(run_sgd("Model7_SMOTETomek_Full", X_train_m7, X_test_raw, y7, y_test,
                           loss=args.loss, penalty=args.penalty, alpha=args.alpha, max_iter=args.max_iter,
                           class_weight=None, save=args.save, outdir=args.outdir))

    summary = pd.DataFrame(results)
    print("\n==== FINAL SUMMARY ====\n", summary)

    if args.save:
        ensure_dir(args.outdir)
        summary.to_csv(Path(args.outdir) / f"summary_{timestamp()}.csv", index=False)


if __name__ == "__main__":
    main()
