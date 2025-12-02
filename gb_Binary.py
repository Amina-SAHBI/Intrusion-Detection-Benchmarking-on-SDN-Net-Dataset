#!/usr/bin/env python3
"""
gb_binary.py

Gradient Boosting-based 7-variant benchmarking script for binary intrusion
detection on the SDN-Net dataset.

This script runs the same 7 experimental pipeline variants:
  - Unbalanced
  - Unbalanced + RF Feature Selection (RF-FS)
  - Unbalanced + Information Gain / Mutual Information FS (IG-FS)
  - SMOTE
  - SMOTE + RF-FS
  - SMOTE + IG-FS
  - SMOTE-Tomek

All variants use a Gradient Boosting classifier as the core model for
binary classification. The `algo_name` label is included in printouts,
plot titles and in the final summary so that results remain traceable
when comparing multiple scripts.

Features:
 - 7 experiment variants per run
 - Inline display of metrics, confusion matrix and ROC for each variant
 - For FS variants, prints selected features and compares RF importances vs IG scores
 - Learning curve (accuracy) per model as a proxy for convergence / capacity
 - "algorithm" field saved in metrics and summary (default: "GradientBoosting")
 - Optional saving (--save) and pausing between models (--pause)

Usage:
  python gb_binary.py --csv ../SDN-Net.csv --k 30
  python gb_binary.py --csv ../SDN-Net.csv --k 30 --save --outdir outputs --pause

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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
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
    imp = rf_feature_importances(X_train, y_train)
    return list(imp.index[:k])

def ig_feature_selection_with_scores(X_train, y_train, k):
    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    selector.fit(X_train, y_train)
    scores = pd.Series(selector.scores_, index=X_train.columns).sort_values(ascending=False)
    return list(scores.index[:k]), scores

def plot_feature_selection_comparison(rf_imp, ig_scores, top_k=20, title_suffix=""):
    rf_imp = rf_imp.sort_values(ascending=False)
    ig_scores = ig_scores.sort_values(ascending=False)
    rf_top = rf_imp.head(top_k); ig_top = ig_scores.head(top_k)
    fig, axes = plt.subplots(1,2,figsize=(16,7))
    rf_top.sort_values().plot.barh(ax=axes[0], color='tab:blue'); axes[0].set_title("RF importances " + title_suffix)
    ig_top.sort_values().plot.barh(ax=axes[1], color='tab:green'); axes[1].set_title("IG scores " + title_suffix)
    plt.tight_layout(); plt.show()
    overlap = set(rf_top.index) & set(ig_top.index)
    print(f"Overlap ({len(overlap)}):", sorted(list(overlap)))

def plot_learning_curve_for_model(estimator, X, y, title="Learning Curve", cv=5, n_jobs=1, train_sizes=np.linspace(0.1,1.0,5)):
    plt.figure(figsize=(8,6))
    train_sizes, train_scores, val_scores = learning_curve(estimator, X, y, cv=cv, scoring='accuracy', train_sizes=train_sizes, n_jobs=n_jobs)
    train_mean = np.mean(train_scores, axis=1); train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1); val_std = np.std(val_scores, axis=1)
    plt.plot(train_sizes, train_mean, 'o-', color='r', label='Training score')
    plt.plot(train_sizes, val_mean, 'o-', color='g', label='CV score')
    plt.fill_between(train_sizes, train_mean-train_std, train_mean+train_std, alpha=0.1, color='r')
    plt.fill_between(train_sizes, val_mean-val_std, val_mean+val_std, alpha=0.1, color='g')
    plt.title(title); plt.xlabel("Training examples"); plt.ylabel("Accuracy"); plt.legend(); plt.tight_layout(); plt.show()

def train_gb_display(
    name,
    X_train_df,
    X_test_df,
    y_train,
    y_test,
    features,
    algo_name="GradientBoosting",
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    subsample=1.0,
    use_sample_weight=False,
    save_outputs=False,
    outdir="outputs",
    pause_between=False,
    show_fs_info=False,
    top_k_fs=20
):
    stamp = timestamp()
    model_dir = Path(outdir) / f"{name.replace(' ', '_')}_{stamp}"
    if save_outputs: ensure_dir(model_dir)

    if not isinstance(X_train_df, pd.DataFrame): X_train_df = pd.DataFrame(X_train_df, columns=features)
    if not isinstance(X_test_df, pd.DataFrame): X_test_df = pd.DataFrame(X_test_df, columns=features)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_df)
    X_test_sc = scaler.transform(X_test_df)

    clf = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, subsample=subsample, random_state=RND)

    sample_weight = None
    if use_sample_weight:
        try:
            classes = np.unique(y_train)
            cw = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
            class_weight_map = {int(c): float(w) for c,w in zip(classes, cw)}
            sample_weight = np.array([class_weight_map[int(lbl)] for lbl in y_train])
        except Exception:
            sample_weight = None

    if sample_weight is not None:
        clf.fit(X_train_sc, y_train, sample_weight=sample_weight)
    else:
        clf.fit(X_train_sc, y_train)

    y_pred = clf.predict(X_test_sc)

    score_for_roc = None
    if hasattr(clf, "predict_proba"):
        try: score_for_roc = clf.predict_proba(X_test_sc)[:,1]
        except Exception: score_for_roc = None
    if score_for_roc is None and hasattr(clf, "decision_function"):
        try: score_for_roc = clf.decision_function(X_test_sc)
        except Exception: score_for_roc = None
    if score_for_roc is None: score_for_roc = y_pred

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1v = f1_score(y_test, y_pred, zero_division=0)
    report = classification_report(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    try:
        fpr, tpr, _ = roc_curve(y_test, score_for_roc)
        roc_auc = auc(fpr, tpr)
    except Exception:
        fpr, tpr, roc_auc = None, None, None

    print(f"\n=== {name} ({algo_name}) ===")
    print(f"Accuracy: {acc:.4f}  Precision: {prec:.4f}  Recall: {rec:.4f}  F1: {f1v:.4f}  AUC: {roc_auc if roc_auc is not None else 'N/A'}")
    print("\nClassification report:\n", report)

    if show_fs_info:
        try: rf_imp = rf_feature_importances(X_train_df, y_train)
        except Exception: rf_imp = pd.Series(dtype=float)
        try: _, ig_scores = ig_feature_selection_with_scores(X_train_df, y_train, k=min(len(X_train_df.columns), top_k_fs))
        except Exception: ig_scores = pd.Series(dtype=float)
        print("\nSelected features used for training (first 100 shown):"); print(list(features)[:100])
        if not rf_imp.empty and not ig_scores.empty: plot_feature_selection_comparison(rf_imp, ig_scores, top_k=top_k_fs, title_suffix=f"({name})")

    fig, axes = plt.subplots(1,2,figsize=(12,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=False, xticklabels=[0,1], yticklabels=[0,1])
    axes[0].set_title(f"Confusion Matrix - {name} ({algo_name})"); axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("True")

    axes[1].plot([0,1],[0,1],'k--', alpha=0.6)
    if fpr is not None and tpr is not None:
        axes[1].plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.3f}")
    else:
        axes[1].text(0.5,0.5,'ROC N/A',ha='center')
    axes[1].set_title(f"ROC Curve - {name} ({algo_name})"); axes[1].set_xlabel("False Positive Rate"); axes[1].set_ylabel("True Positive Rate"); axes[1].legend(loc='lower right')
    plt.tight_layout(); plt.show()

    try:
        plot_learning_curve_for_model(GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, subsample=subsample, random_state=RND), X_train_df, y_train, title=f"Learning Curve ({name} - {algo_name})", cv=5, n_jobs=1)
    except Exception:
        pass

    if save_outputs:
        try: joblib.dump(clf, model_dir / f"{algo_name}_model_{stamp}.joblib")
        except Exception: pass
        try: joblib.dump(scaler, model_dir / f"scaler_{stamp}.joblib")
        except Exception: pass
        if features is not None:
            pd.Series(list(features), name="feature").to_csv(model_dir / f"features_{stamp}.csv", index=False)
        auc_value = None
        try:
            if roc_auc is not None and not (isinstance(roc_auc, float) and np.isnan(roc_auc)):
                auc_value = float(roc_auc)
        except Exception:
            auc_value = None
        metrics = {"algorithm": algo_name, "n_estimators": int(n_estimators), "learning_rate": float(learning_rate), "max_depth": int(max_depth), "subsample": float(subsample), "accuracy": float(acc), "precision": float(prec), "recall": float(rec), "f1": float(f1v), "auc": auc_value}
        with open(model_dir / f"metrics_{stamp}.json","w",encoding="utf8") as f: json.dump(metrics,f,indent=2,ensure_ascii=False)
        with open(model_dir / f"classification_report_{stamp}.txt","w",encoding="utf8") as f: f.write(report)
        try: fig.savefig(model_dir / f"confusion_roc_{stamp}.png", dpi=150); plt.close(fig)
        except Exception: pass

    if pause_between:
        try: input("Press Enter to continue to next model...")
        except Exception: time.sleep(2)

    return {"config": name, "algorithm": algo_name, "n_estimators": n_estimators, "learning_rate": learning_rate, "max_depth": max_depth, "subsample": subsample, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1v, "auc": roc_auc}
