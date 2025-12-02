#!/usr/bin/env python3
"""
gb_multiclass.py

Gradient Boosting-based 7-variant benchmarking script for multiclass intrusion detection.

Notes:
 - uses GradientBoostingClassifier (supports multiclass with 'deviance')
 - computes weighted precision/recall/f1 and macro OVR AUC when possible
 - supports sample weighting to balance classes (passed via sample_weight)
 - saves class mapping when --save
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
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.utils.class_weight import compute_class_weight

from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek

sns.set(style="whitegrid")
RND = 42

def timestamp(): return datetime.now().strftime("%Y%m%d_%H%M%S")
def ensure_dir(d): Path(d).mkdir(parents=True, exist_ok=True)

def rf_feature_importances(X_train, y_train):
    rf = RandomForestClassifier(n_estimators=200, random_state=RND, n_jobs=-1)
    rf.fit(X_train, y_train)
    return pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)

def rf_feature_selection(X_train, y_train, k):
    imp = rf_feature_importances(X_train, y_train)
    return list(imp.index[:k])

def ig_feature_selection_with_scores(X_train, y_train, k):
    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    selector.fit(X_train, y_train)
    scores = pd.Series(selector.scores_, index=X_train.columns).sort_values(ascending=False)
    return list(scores.index[:k]), scores

def plot_feature_selection_comparison(rf_imp, ig_scores, top_k=20, title_suffix=""):
    rf_imp = rf_imp.sort_values(ascending=False); ig_scores = ig_scores.sort_values(ascending=False)
    rf_top = rf_imp.head(top_k); ig_top = ig_scores.head(top_k)
    fig, axes = plt.subplots(1,2,figsize=(16,7))
    rf_top.sort_values().plot.barh(ax=axes[0], color='tab:blue'); axes[0].set_title("RF importances " + title_suffix)
    ig_top.sort_values().plot.barh(ax=axes[1], color='tab:green'); axes[1].set_title("IG scores " + title_suffix)
    plt.tight_layout(); plt.show()
    overlap = set(rf_top.index) & set(ig_top.index); print(f"Overlap ({len(overlap)}):", sorted(list(overlap)))

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

def prepare_multiclass(csv_path):
    csv_path = Path(csv_path)
    if not csv_path.exists(): raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path, skipinitialspace=True, low_memory=False)
    unnamed = [c for c in df.columns if c.startswith("Unnamed")]
    if unnamed: df.drop(columns=unnamed, inplace=True)
    if "Attack Type" not in df.columns:
        if "Class" not in df.columns: raise KeyError("CSV must contain 'Attack Type' or 'Class'.")
        df['Attack Type'] = df['Class'].astype(str).str.strip()
    df['Attack Type'] = df['Attack Type'].astype(str).str.strip().fillna("UNKNOWN")
    cat = pd.Categorical(df['Attack Type']); class_names = list(cat.categories)
    df['Attack_Label_Code'] = cat.codes; y = df['Attack_Label_Code'].astype(int)
    exclude = {'Attack Type','Attack_Label_Code'}
    obj_cols = [c for c in df.select_dtypes(include=['object','category']).columns if c not in exclude]
    if obj_cols: df = pd.get_dummies(df, columns=obj_cols, drop_first=True)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist(); feature_cols = [c for c in num_cols if c not in ['Attack_Label_Code']]
    if not feature_cols: raise ValueError("No feature columns found after preprocessing.")
    X = df[feature_cols].replace([np.inf,-np.inf], np.nan).fillna(df[feature_cols].mean())
    return X, y, class_names

def train_gb_display_multiclass(
    name, X_train_df, X_test_df, y_train, y_test, features, class_names,
    algo_name="GradientBoosting-Multiclass", n_estimators=100, learning_rate=0.1, max_depth=3, subsample=1.0,
    use_sample_weight=False, save_outputs=False, outdir="outputs", pause_between=False, show_fs_info=False, top_k_fs=20
):
    stamp = timestamp(); model_dir = Path(outdir)/f"{name.replace(' ','_')}_{stamp}"
    if save_outputs: ensure_dir(model_dir)
    if not isinstance(X_train_df, pd.DataFrame): X_train_df = pd.DataFrame(X_train_df, columns=features)
    if not isinstance(X_test_df, pd.DataFrame): X_test_df = pd.DataFrame(X_test_df, columns=features)
    scaler = StandardScaler(); X_train_sc = scaler.fit_transform(X_train_df); X_test_sc = scaler.transform(X_test_df)

    clf = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, subsample=subsample, random_state=RND)

    sample_weight = None
    if use_sample_weight:
        try:
            classes = np.unique(y_train); cw = compute_class_weight('balanced', classes=classes, y=y_train)
            cw_map = {int(c): float(w) for c,w in zip(classes,cw)}
            sample_weight = np.array([cw_map[int(lbl)] for lbl in y_train])
        except Exception:
            sample_weight = None

    if sample_weight is not None:
        clf.fit(X_train_sc, y_train, sample_weight=sample_weight)
    else:
        clf.fit(X_train_sc, y_train)

    y_pred = clf.predict(X_test_sc)
    try:
        y_prob = clf.predict_proba(X_test_sc)
    except Exception:
        n_classes = len(class_names); y_prob = np.zeros((len(y_pred), n_classes))
        for i,p in enumerate(y_pred): y_prob[i,int(p)] = 1.0

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1v = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    report = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    try:
        y_test_bin = label_binarize(y_test, classes=list(range(len(class_names))))
        roc_auc = float(roc_auc_score(y_test_bin, y_prob, average='macro', multi_class='ovr'))
    except Exception:
        roc_auc = None

    print(f"\n=== {name} ({algo_name}) ===")
    print(f"Classes: {class_names}")
    print(f"Accuracy: {acc:.4f}  Precision (weighted): {prec:.4f}  Recall (weighted): {rec:.4f}  F1 (weighted): {f1v:.4f}  AUC (macro OVR): {roc_auc if roc_auc is not None else 'N/A'}")
    print("\nClassification report:\n", report)

    if show_fs_info:
        try: rf_imp = rf_feature_importances(X_train_df, y_train)
        except Exception: rf_imp = pd.Series(dtype=float)
        try: _, ig_scores = ig_feature_selection_with_scores(X_train_df, y_train, k=min(len(X_train_df.columns), top_k_fs))
        except Exception: ig_scores = pd.Series(dtype=float)
        print("\nSelected features used for training (first 100 shown):"); print(list(features)[:100])
        if not rf_imp.empty and not ig_scores.empty: plot_feature_selection_comparison(rf_imp, ig_scores, top_k=top_k_fs, title_suffix=f"({name})")

    fig, axes = plt.subplots(1,2,figsize=(14,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=False, xticklabels=class_names, yticklabels=class_names)
    axes[0].set_title(f"Confusion Matrix - {name} ({algo_name})"); axes[0].tick_params(axis='x', rotation=45)

    axes[1].plot([0,1],[0,1],'k--',alpha=0.6)
    if roc_auc is not None:
        try:
            fpr={}; tpr={}; n_classes=len(class_names)
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_test == i, y_prob[:, i])
            for i in range(n_classes):
                axes[1].plot(fpr[i], tpr[i], lw=1.5, label=f"{class_names[i]} (AUC {auc(fpr[i], tpr[i]):.2f})")
            axes[1].legend(loc='lower right', fontsize='small')
        except Exception:
            axes[1].text(0.5,0.5,'ROC N/A',ha='center')
    else:
        axes[1].text(0.5,0.5,'ROC N/A',ha='center')
    axes[1].set_title(f"ROC Curves - {name} ({algo_name})"); axes[1].set_xlabel("False Positive Rate"); axes[1].set_ylabel("True Positive Rate")
    plt.tight_layout(); plt.show()

    try:
        plot_learning_curve_for_model(GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, subsample=subsample, random_state=RND), X_train_df, y_train, title=f"Learning Curve ({name})", cv=5, n_jobs=1)
    except Exception:
        pass

    if save_outputs:
        try: joblib.dump(clf, Path(outdir)/f"{algo_name}_model_{timestamp()}.joblib")
        except Exception: pass
        try: joblib.dump(scaler, Path(outdir)/f"scaler_{timestamp()}.joblib")
        except Exception: pass
        if features is not None:
            pd.Series(list(features), name="feature").to_csv(Path(outdir)/f"features_{timestamp()}.csv", index=False)
        try:
            with open(Path(outdir)/f"classes_{timestamp()}.json","w",encoding="utf8") as f:
                json.dump({"classes": class_names}, f, indent=2, ensure_ascii=False)
        except Exception:
            pass
        metrics = {"algorithm": algo_name, "n_estimators": int(n_estimators), "learning_rate": float(learning_rate), "max_depth": int(max_depth), "subsample": float(subsample), "n_classes": int(len(class_names)), "accuracy": float(acc), "precision_weighted": float(prec), "recall_weighted": float(rec), "f1_weighted": float(f1v), "auc_macro_ovr": float(roc_auc) if roc_auc is not None else None}
        with open(Path(outdir)/f"metrics_{timestamp()}.json","w",encoding="utf8") as f: json.dump(metrics,f,indent=2,ensure_ascii=False)
        with open(Path(outdir)/f"classification_report_{timestamp()}.txt","w",encoding="utf8") as f: f.write(report)
        try: fig.savefig(Path(outdir)/f"confusion_roc_{timestamp()}.png", dpi=150); plt.close(fig)
        except Exception: pass

    if pause_between:
        try: input("Press Enter to continue...")
        except Exception: time.sleep(2)

    return {"config": name, "algorithm": algo_name, "n_estimators": n_estimators, "learning_rate": learning_rate, "max_depth": max_depth, "subsample": subsample, "n_classes": int(len(class_names)), "accuracy": acc, "precision_weighted": prec, "recall_weighted": rec, "f1_weighted": f1v, "auc_macro_ovr": roc_auc}
