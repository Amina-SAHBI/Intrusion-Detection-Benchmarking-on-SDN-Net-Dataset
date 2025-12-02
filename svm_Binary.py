#!/usr/bin/env python3
"""
svc_binary.py

Support Vector Machine (SVC)-based 7-variant benchmarking script for binary
intrusion detection on the SDN-Net dataset.

This script runs the same 7 experimental pipeline variants:
  - Unbalanced
  - Unbalanced + RF Feature Selection (RF-FS)
  - Unbalanced + Information Gain / Mutual Information FS (IG-FS)
  - SMOTE
  - SMOTE + RF-FS
  - SMOTE + IG-FS
  - SMOTE-Tomek

All variants use an SVC classifier as the core model for binary classification.
The `algo_name` label is included in printouts, plot titles and in the final
summary so that results remain traceable when comparing multiple scripts.

Features:
 - 7 experiment variants per run
 - Inline display of metrics, confusion matrix and ROC for each variant
 - For FS variants, prints selected features and compares RF importances vs IG scores
 - Learning curve (accuracy) per model as a proxy for convergence / capacity
 - "algorithm" field saved in metrics and summary (default: "SVC")
 - Optional saving (--save) and pausing between models (--pause)

Usage:
  python svc_binary.py --csv ../SDN-Net.csv --k 30
  python svc_binary.py --csv ../SDN-Net.csv --k 30 --save --outdir outputs --pause

Dependencies:
  pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn joblib
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime
import time

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight

from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek

sns.set(style="whitegrid")
RND = 42


def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(d):
    Path(d).mkdir(parents=True, exist_ok=True)


def rf_feature_importances(X_train, y_train):
    rf = RandomForestClassifier(n_estimators=200, random_state=RND, n_jobs=-1)
    rf.fit(X_train, y_train)
    importances = pd.Series(rf.feature_importances_, index=X_train.columns)
    return importances.sort_values(ascending=False)


def rf_feature_selection(X_train, y_train, k):
    importances = rf_feature_importances(X_train, y_train)
    return list(importances.index[:k])


def ig_feature_selection_with_scores(X_train, y_train, k):
    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    selector.fit(X_train, y_train)
    scores = pd.Series(selector.scores_, index=X_train.columns).sort_values(ascending=False)
    return list(scores.index[:k]), scores


def plot_feature_selection_comparison(rf_imp, ig_scores, top_k=20, title_suffix=""):
    rf_imp = rf_imp.sort_values(ascending=False)
    ig_scores = ig_scores.sort_values(ascending=False)
    rf_top = rf_imp.head(top_k)
    ig_top = ig_scores.head(top_k)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    rf_top.sort_values().plot.barh(ax=axes[0], color='tab:blue')
    axes[0].set_title("Feature Importances (Random Forest) " + title_suffix)
    axes[0].set_xlabel("Importance")

    ig_top.sort_values().plot.barh(ax=axes[1], color='tab:green')
    axes[1].set_title("Feature Scores (Information Gain) " + title_suffix)
    axes[1].set_xlabel("Mutual Information Score")

    plt.tight_layout()
    plt.show()

    rf_set = set(rf_top.index)
    ig_set = set(ig_top.index)
    overlap = rf_set & ig_set
    print(f"Top {top_k} RF features count: {len(rf_top)}; Top {top_k} IG features count: {len(ig_top)}")
    print(f"Overlap count: {len(overlap)}")
    if overlap:
        print("Overlapping features (RF ∩ IG):")
        print(sorted(list(overlap)))
    else:
        print("No overlap in top features.")


def plot_learning_curve_for_model(estimator, X, y, title="Learning Curve", cv=5, n_jobs=1,
                                  train_sizes=np.linspace(0.1, 1.0, 5)):
    try:
        plt.figure(figsize=(8, 6))
        train_sizes, train_scores, val_scores = learning_curve(
            estimator, X, y, cv=cv, scoring='accuracy',
            train_sizes=train_sizes, n_jobs=n_jobs
        )
        train_mean = np.mean(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        plt.plot(train_sizes, train_mean, 'o-', color='r', label='Training score')
        plt.plot(train_sizes, val_mean, 'o-', color='g', label='Cross-validation score')
        plt.title(title)
        plt.xlabel("Training examples")
        plt.ylabel("Score (accuracy)")
        plt.legend(loc="best")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print("Unable to plot learning curve:", e)


def load_and_prepare(csv_path):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path, skipinitialspace=True, low_memory=False)
    unnamed = [c for c in df.columns if c.startswith("Unnamed")]
    if unnamed:
        df.drop(columns=unnamed, inplace=True)
    if "Attack Type" not in df.columns:
        if "Class" not in df.columns:
            raise KeyError("CSV must contain 'Attack Type' or 'Class'.")
        df['Attack Type'] = df['Class'].astype(str).str.strip()
    # Binary mapping
    df['Attack_Binary_Label'] = df['Attack Type'].astype(str).str.strip().str.upper().apply(lambda x: 0 if x == "NORMAL" else 1)
    exclude = {'Class', 'Attack Type', 'Attack_Binary_Label'}
    obj_cols = [c for c in df.select_dtypes(include=['object', 'category']).columns if c not in exclude]
    if obj_cols:
        df = pd.get_dummies(df, columns=obj_cols, drop_first=True)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
        df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
    drop_cols = [c for c in ['Class', 'Attack Type', 'Attack_Binary_Label'] if c in df.columns]
    X = df.drop(columns=drop_cols, errors='ignore')
    y = df['Attack_Binary_Label'].astype(int)
    return X, y


def train_svc_display(
    name,
    X_train_df,
    X_test_df,
    y_train,
    y_test,
    features,
    algo_name="SVC",
    kernel="rbf",
    C=1.0,
    gamma="scale",
    class_weight=None,
    save_outputs=False,
    outdir="outputs",
    pause_between=False,
    show_fs_info=False,
    top_k_fs=20
):
    stamp = timestamp()
    model_dir = Path(outdir) / f"{name.replace(' ', '_')}_{stamp}"
    if save_outputs:
        ensure_dir(model_dir)

    # Ensure DataFrame inputs
    if not isinstance(X_train_df, pd.DataFrame):
        X_train_df = pd.DataFrame(X_train_df, columns=features)
    if not isinstance(X_test_df, pd.DataFrame):
        X_test_df = pd.DataFrame(X_test_df, columns=features)

    # Feature selection info
    if show_fs_info and isinstance(features, (list, pd.Index, np.ndarray)):
        try:
            rf_imp = rf_feature_importances(X_train_df, y_train)
        except Exception as e:
            print("Error computing RF importances:", e)
            rf_imp = pd.Series(dtype=float)
        try:
            _, ig_scores = ig_feature_selection_with_scores(X_train_df, y_train, k=min(len(X_train_df.columns), top_k_fs))
        except Exception as e:
            print("Error computing IG scores:", e)
            ig_scores = pd.Series(dtype=float)
        print("\nSelected features used for training (first 100 shown):")
        print(list(features)[:100])
        if not rf_imp.empty and not ig_scores.empty:
            plot_feature_selection_comparison(rf_imp, ig_scores, top_k=top_k_fs, title_suffix=f"({name})")

    # Scale
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_df)
    X_test_sc = scaler.transform(X_test_df)

    # Instantiate SVC
    clf = SVC(kernel=kernel, C=C, gamma=gamma, probability=True, class_weight=class_weight, random_state=RND)

    # Fit
    clf.fit(X_train_sc, y_train)
    y_pred = clf.predict(X_test_sc)

    # Score for ROC
    score_for_roc = None
    if hasattr(clf, "predict_proba"):
        try:
            score_for_roc = clf.predict_proba(X_test_sc)[:, 1]
        except Exception:
            score_for_roc = None
    if score_for_roc is None and hasattr(clf, "decision_function"):
        try:
            score_for_roc = clf.decision_function(X_test_sc)
        except Exception:
            score_for_roc = None
    if score_for_roc is None:
        score_for_roc = y_pred

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1v = f1_score(y_test, y_pred, zero_division=0)
    report = classification_report(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    # ROC/AUC
    try:
        fpr, tpr, _ = roc_curve(y_test, score_for_roc)
        roc_auc = auc(fpr, tpr)
    except Exception:
        fpr, tpr, roc_auc = None, None, None

    # Print
    print(f"\n=== {name} ({algo_name}) ===")
    print(f"Accuracy: {acc:.4f}  Precision: {prec:.4f}  Recall: {rec:.4f}  F1: {f1v:.4f}  AUC: {roc_auc if roc_auc is not None else 'N/A'}")
    print("\nClassification report:\n", report)

    # Plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=False, xticklabels=[0, 1], yticklabels=[0, 1])
    axes[0].set_title(f"Confusion Matrix - {name} ({algo_name})")
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("True")

    axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.6)
    if fpr is not None and tpr is not None:
        axes[1].plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.3f}")
    else:
        axes[1].text(0.5, 0.5, 'ROC N/A', ha='center')
    axes[1].set_title(f"ROC Curve - {name} ({algo_name})")
    axes[1].set_xlabel("False Positive Rate"); axes[1].set_ylabel("True Positive Rate"); axes[1].legend(loc='lower right')
    plt.tight_layout(); plt.show()

    # Learning curve
    try:
        plot_learning_curve_for_model(SVC(kernel=kernel), X_train_df, y_train, title=f"Learning Curve ({name} - {algo_name})", cv=5, n_jobs=-1)
    except Exception as e:
        print("Unable to plot learning curve:", e)

    # Save artifacts
    if save_outputs:
        try:
            joblib.dump(clf, model_dir / f"{algo_name}_model_{timestamp()}.joblib")
            joblib.dump(scaler, model_dir / f"scaler_{timestamp()}.joblib")
        except Exception as e:
            print("Unable to save artifacts:", e)
        if features is not None:
            pd.Series(list(features), name="feature").to_csv(model_dir / f"features_{timestamp()}.csv", index=False)
        metrics = {"algorithm": algo_name, "kernel": kernel, "C": float(C), "gamma": str(gamma), "accuracy": float(acc), "precision": float(prec), "recall": float(rec), "f1": float(f1v), "auc": float(roc_auc) if roc_auc is not None else None}
        with open(model_dir / f"metrics_{timestamp()}.json", "w", encoding="utf8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        with open(model_dir / f"classification_report_{timestamp()}.txt", "w", encoding="utf8") as f:
            f.write(report)
        try:
            fig.savefig(model_dir / f"confusion_roc_{timestamp()}.png", dpi=150)
            plt.close(fig)
        except Exception:
            pass

    if pause_between:
        try:
            input("Press Enter to continue to next model...")
        except Exception:
            time.sleep(2)

    return {"config": name, "algorithm": algo_name, "kernel": kernel, "C": C, "gamma": gamma, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1v, "auc": roc_auc}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="../SDN-Net.csv", help="Path to SDN-Net CSV")
    parser.add_argument("--outdir", default="outputs", help="Output dir (only used if --save)")
    parser.add_argument("--k", type=int, default=30, help="Number of features for FS")
    parser.add_argument("--save", action="store_true", help="If set, save artifacts to outdir (default: False)")
    parser.add_argument("--pause", action="store_true", help="Pause between models (press Enter). Default: False")
    parser.add_argument("--kernel", default="rbf", help="Kernel for SVC (rbf, linear, poly, sigmoid)")
    parser.add_argument("--C", type=float, default=1.0, help="Regularization parameter")
    parser.add_argument("--gamma", default="scale", help="Kernel coefficient for rbf/poly/sigmoid")
    parser.add_argument("--class_weight", action="store_true", help="Use class_weight='balanced' in SVC")
    args = parser.parse_known_args(argv)[0]

    X_full, y_full = load_and_prepare(args.csv)
    print("Dataset shape:", X_full.shape)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_full, y_full, test_size=0.33, random_state=RND, stratify=y_full)
    print("Train / Test shapes:", X_train_raw.shape, X_test_raw.shape)

    results = []

    cw = "balanced" if args.class_weight else None

    results.append(train_svc_display("Model1_Unbalanced_Full", X_train_raw, X_test_raw, y_train, y_test, X_full.columns, algo_name=args.kernel.upper()+"-SVC", kernel=args.kernel, C=args.C, gamma=args.gamma, class_weight=cw, save_outputs=args.save, outdir=args.outdir, pause_between=args.pause))

    sel2 = rf_feature_selection(X_train_raw, y_train, args.k)
    results.append(train_svc_display("Model2_Unbalanced_RF-FS", X_train_raw[sel2], X_test_raw[sel2], y_train, y_test, sel2, algo_name=args.kernel.upper()+"-SVC", kernel=args.kernel, C=args.C, gamma=args.gamma, class_weight=cw, save_outputs=args.save, outdir=args.outdir, pause_between=args.pause, show_fs_info=True, top_k_fs=args.k))

    sel3, ig_scores_full = ig_feature_selection_with_scores(X_train_raw, y_train, k=args.k)
    results.append(train_svc_display("Model3_Unbalanced_IG-FS", X_train_raw[sel3], X_test_raw[sel3], y_train, y_test, sel3, algo_name=args.kernel.upper()+"-SVC", kernel=args.kernel, C=args.C, gamma=args.gamma, class_weight=cw, save_outputs=args.save, outdir=args.outdir, pause_between=args.pause, show_fs_info=True, top_k_fs=args.k))

    smote = SMOTE(random_state=RND)
    X_train_m4_arr, y_train_m4 = smote.fit_resample(X_train_raw, y_train)
    X_train_m4 = pd.DataFrame(X_train_m4_arr, columns=X_train_raw.columns)
    results.append(train_svc_display("Model4_SMOTE_Full", X_train_m4, X_test_raw, y_train_m4, y_test, X_full.columns, algo_name=args.kernel.upper()+"-SVC", kernel=args.kernel, C=args.C, gamma=args.gamma, class_weight=None, save_outputs=args.save, outdir=args.outdir, pause_between=args.pause))

    sel5 = rf_feature_selection(X_train_m4, y_train_m4, args.k)
    results.append(train_svc_display("Model5_SMOTE_RF-FS", X_train_m4[sel5], X_test_raw[sel5], y_train_m4, y_test, sel5, algo_name=args.kernel.upper()+"-SVC", kernel=args.kernel, C=args.C, gamma=args.gamma, class_weight=None, save_outputs=args.save, outdir=args.outdir, pause_between=args.pause, show_fs_info=True, top_k_fs=args.k))

    sel6, ig_scores_m4 = ig_feature_selection_with_scores(X_train_m4, y_train_m4, k=args.k)
    results.append(train_svc_display("Model6_SMOTE_IG-FS", X_train_m4[sel6], X_test_raw[sel6], y_train_m4, y_test, sel6, algo_name=args.kernel.upper()+"-SVC", kernel=args.kernel, C=args.C, gamma=args.gamma, class_weight=None, save_outputs=args.save, outdir=args.outdir, pause_between=args.pause, show_fs_info=True, top_k_fs=args.k))

    smt = SMOTETomek(random_state=RND)
    X_train_m7_arr, y_train_m7 = smt.fit_resample(X_train_raw, y_train)
    X_train_m7 = pd.DataFrame(X_train_m7_arr, columns=X_train_raw.columns)
    results.append(train_svc_display("Model7_SMOTETomek_Full", X_train_m7, X_test_raw, y_train_m7, y_test, X_full.columns, algo_name=args.kernel.upper()+"-SVC", kernel=args.kernel, C=args.C, gamma=args.gamma, class_weight=None, save_outputs=args.save, outdir=args.outdir, pause_between=args.pause))

    results_df = pd.DataFrame(results)
    print("\n=== Summary of all 7 models ===")
    print(results_df)

    if args.save:
        ensure_dir(args.outdir)
        summary_path = Path(args.outdir) / f"summary_7models_{args.algo}_{timestamp()}.csv"
        results_df.to_csv(summary_path, index=False)
        print("Saved summary to:", summary_path)


if __name__ == "__main__":
    main()
