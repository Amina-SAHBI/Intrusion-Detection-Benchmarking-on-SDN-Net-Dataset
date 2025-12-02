#!/usr/bin/env python3
"""
lr_multiclass.py

Logistic Regression-based 7-variant benchmarking script for multiclass intrusion detection.

Notes:
 - classifier head uses LogisticRegression(multi_class='multinomial', solver=solver)
 - metrics: accuracy, precision/recall/f1 (weighted), AUC macro OVR when possible
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
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
    rf_top = rf_imp.head(top_k)
    ig_top = ig_scores.head(top_k)
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

def load_and_prepare_multiclass(csv_path):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path, skipinitialspace=True, low_memory=False)
    unnamed = [c for c in df.columns if c.startswith("Unnamed")]
    if unnamed: df.drop(columns=unnamed, inplace=True)
    if "Attack Type" not in df.columns:
        if "Class" not in df.columns:
            raise KeyError("CSV must contain 'Attack Type' or 'Class'.")
        df['Attack Type'] = df['Class'].astype(str).str.strip()
    df['Attack Type'] = df['Attack Type'].astype(str).str.strip().fillna("UNKNOWN")
    cat = pd.Categorical(df['Attack Type'])
    class_names = list(cat.categories)
    df['Attack_Label_Code'] = cat.codes
    y = df['Attack_Label_Code'].astype(int)
    exclude = {'Attack Type', 'Attack_Label_Code'}
    obj_cols = [c for c in df.select_dtypes(include=['object','category']).columns if c not in exclude]
    if obj_cols: df = pd.get_dummies(df, columns=obj_cols, drop_first=True)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    drop_cols = ['Attack_Label_Code']
    feature_cols = [c for c in num_cols if c not in drop_cols]
    if not feature_cols: raise ValueError("No feature columns found after preprocessing.")
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(df[feature_cols].mean())
    return X, y, class_names

def train_lr_display_multiclass(
    name, X_train_df, X_test_df, y_train, y_test, features, class_names,
    algo_name="LogisticRegression-Multiclass", C=1.0, penalty='l2', solver='lbfgs',
    class_weight_option=False, save_outputs=False, outdir="outputs", pause_between=False,
    show_fs_info=False, top_k_fs=20
):
    stamp = timestamp()
    model_dir = Path(outdir) / f"{name.replace(' ', '_')}_{stamp}"
    if save_outputs: ensure_dir(model_dir)

    if not isinstance(X_train_df, pd.DataFrame): X_train_df = pd.DataFrame(X_train_df, columns=features)
    if not isinstance(X_test_df, pd.DataFrame): X_test_df = pd.DataFrame(X_test_df, columns=features)

    scaler = StandardScaler(); X_train_sc = scaler.fit_transform(X_train_df); X_test_sc = scaler.transform(X_test_df)

    class_weight = None
    if class_weight_option:
        try:
            classes = np.unique(y_train); cw = compute_class_weight('balanced', classes=classes, y=y_train)
            class_weight = {int(c): float(w) for c,w in zip(classes,cw)}
        except Exception:
            class_weight = 'balanced'

    lr_kwargs = {"C": C, "penalty": penalty, "solver": solver, "multi_class": "multinomial", "max_iter": 1000}
    if class_weight is not None:
        lr_kwargs["class_weight"] = class_weight

    clf = LogisticRegression(**lr_kwargs)
    clf.fit(X_train_sc, y_train)
    y_pred = clf.predict(X_test_sc)

    # probabilities
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
        plot_learning_curve_for_model(LogisticRegression(C=C, penalty=penalty, solver=solver, multi_class='multinomial', max_iter=1000), X_train_df, y_train, title=f"Learning Curve ({name} - {algo_name})", cv=5, n_jobs=1)
    except Exception:
        pass

    if save_outputs:
        try: joblib.dump(clf, Path(outdir) / f"{algo_name}_model_{timestamp()}.joblib")
        except Exception: pass
        try: joblib.dump(scaler, Path(outdir) / f"scaler_{timestamp()}.joblib")
        except Exception: pass
        if features is not None:
            pd.Series(list(features), name="feature").to_csv(Path(outdir) / f"features_{timestamp()}.csv", index=False)
        try:
            with open(Path(outdir)/f"classes_{timestamp()}.json","w",encoding="utf8") as f:
                json.dump({"classes": class_names}, f, indent=2, ensure_ascii=False)
        except Exception:
            pass
        metrics = {"algorithm": algo_name, "C": float(C), "penalty": penalty, "solver": solver, "n_classes": int(len(class_names)), "accuracy": float(acc), "precision_weighted": float(prec), "recall_weighted": float(rec), "f1_weighted": float(f1v), "auc_macro_ovr": float(roc_auc) if roc_auc is not None else None}
        with open(Path(outdir)/f"metrics_{timestamp()}.json","w",encoding="utf8") as f:
            json.dump(metrics,f,indent=2,ensure_ascii=False)
        with open(Path(outdir)/f"classification_report_{timestamp()}.txt","w",encoding="utf8") as f:
            f.write(report)
        try:
            fig.savefig(Path(outdir)/f"confusion_roc_{timestamp()}.png", dpi=150); plt.close(fig)
        except Exception:
            pass

    if pause_between:
        try: input("Press Enter to continue...")
        except Exception: time.sleep(2)

    return {"config": name, "algorithm": algo_name, "C": C, "penalty": penalty, "solver": solver, "n_classes": int(len(class_names)), "accuracy": acc, "precision_weighted": prec, "recall_weighted": rec, "f1_weighted": f1v, "auc_macro_ovr": roc_auc}

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="../SDN-Net.csv")
    parser.add_argument("--outdir", default="outputs")
    parser.add_argument("--k", type=int, default=30)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--pause", action="store_true")
    parser.add_argument("--algo", default="LogisticRegression-Multiclass")
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--penalty", type=str, default="l2", choices=["l1","l2","elasticnet","none"])
    parser.add_argument("--solver", type=str, default="lbfgs")
    parser.add_argument("--class_weight", action="store_true")
    args = parser.parse_args(argv)

    X_full, y_full, class_names = load_and_prepare_multiclass(args.csv)
    print("Dataset shape (features):", X_full.shape, "n_classes:", len(class_names))
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_full, y_full, test_size=0.33, random_state=RND, stratify=y_full)

    results = []
    results.append(train_lr_display_multiclass("Model1_Unbalanced_Full", X_train_raw, X_test_raw, y_train, y_test, X_full.columns, class_names, algo_name=args.algo, C=args.C, penalty=args.penalty, solver=args.solver, class_weight_option=args.class_weight, save_outputs=args.save, outdir=args.outdir, pause_between=args.pause))
    sel2 = rf_feature_selection(X_train_raw, y_train, args.k)
    results.append(train_lr_display_multiclass("Model2_Unbalanced_RF-FS", X_train_raw[sel2], X_test_raw[sel2], y_train, y_test, sel2, class_names, algo_name=args.algo, C=args.C, penalty=args.penalty, solver=args.solver, class_weight_option=args.class_weight, save_outputs=args.save, outdir=args.outdir, pause_between=args.pause, show_fs_info=True, top_k_fs=args.k))
    sel3, ig_scores = ig_feature_selection_with_scores(X_train_raw, y_train, k=args.k)
    results.append(train_lr_display_multiclass("Model3_Unbalanced_IG-FS", X_train_raw[sel3], X_test_raw[sel3], y_train, y_test, sel3, class_names, algo_name=args.algo, C=args.C, penalty=args.penalty, solver=args.solver, class_weight_option=args.class_weight, save_outputs=args.save, outdir=args.outdir, pause_between=args.pause, show_fs_info=True, top_k_fs=args.k))
    smote = SMOTE(random_state=RND); X_train_m4_arr, y_train_m4 = smote.fit_resample(X_train_raw, y_train); X_train_m4 = pd.DataFrame(X_train_m4_arr, columns=X_train_raw.columns)
    results.append(train_lr_display_multiclass("Model4_SMOTE_Full", X_train_m4, X_test_raw, y_train_m4, y_test, X_full.columns, class_names, algo_name=args.algo, C=args.C, penalty=args.penalty, solver=args.solver, class_weight_option=args.class_weight, save_outputs=args.save, outdir=args.outdir, pause_between=args.pause))
    sel5 = rf_feature_selection(X_train_m4, y_train_m4, args.k)
    results.append(train_lr_display_multiclass("Model5_SMOTE_RF-FS", X_train_m4[sel5], X_test_raw[sel5], y_train_m4, y_test, sel5, class_names, algo_name=args.algo, C=args.C, penalty=args.penalty, solver=args.solver, class_weight_option=args.class_weight, save_outputs=args.save, outdir=args.outdir, pause_between=args.pause, show_fs_info=True, top_k_fs=args.k))
    sel6, ig_scores_m4 = ig_feature_selection_with_scores(X_train_m4, y_train_m4, k=args.k)
    results.append(train_lr_display_multiclass("Model6_SMOTE_IG-FS", X_train_m4[sel6], X_test_raw[sel6], y_train_m4, y_test, sel6, class_names, algo_name=args.algo, C=args.C, penalty=args.penalty, solver=args.solver, class_weight_option=args.class_weight, save_outputs=args.save, outdir=args.outdir, pause_between=args.pause, show_fs_info=True, top_k_fs=args.k))
    smt = SMOTETomek(random_state=RND); X_train_m7_arr, y_train_m7 = smt.fit_resample(X_train_raw, y_train); X_train_m7 = pd.DataFrame(X_train_m7_arr, columns=X_train_raw.columns)
    results.append(train_lr_display_multiclass("Model7_SMOTETomek_Full", X_train_m7, X_test_raw, y_train_m7, y_test, X_full.columns, class_names, algo_name=args.algo, C=args.C, penalty=args.penalty, solver=args.solver, class_weight_option=args.class_weight, save_outputs=args.save, outdir=args.outdir, pause_between=args.pause))

    results_df = pd.DataFrame(results); print("\n=== Summary ==="); print(results_df)
    if args.save:
        ensure_dir(args.outdir); summary_path = Path(args.outdir)/f"summary_7models_{args.algo}_{timestamp()}.csv"; results_df.to_csv(summary_path, index=False); print("Saved summary to:", summary_path)

if __name__ == "__main__":
    main()
