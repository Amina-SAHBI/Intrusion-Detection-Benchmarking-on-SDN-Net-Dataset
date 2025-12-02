#!/usr/bin/env python3
"""
sgd_multiclass.py

SGD linear classifier 7-variant benchmarking script for multiclass intrusion detection.

Notes / features:
 - prepares multiclass labels (categorical Attack Type -> integer codes)
 - runs same 7 variants (Unbalanced, RF-FS, IG-FS, SMOTE, SMOTE+RF-FS, SMOTE+IG-FS, SMOTE-Tomek)
 - uses SGDClassifier(loss='log_loss') for multiclass (supports predict_proba in recent sklearn)
 - computes accuracy and weighted precision/recall/f1, and macro OVR ROC-AUC if possible
 - optional class_weight balancing (class_weight='balanced' or sample_weight for SMOTE variants)
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
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
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
    rf = RandomForestClassifier(n_estimators=200, random_state=RND, n_jobs=-1)
    rf.fit(X_train, y_train)
    return pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)


def rf_feature_selection(X_train, y_train, k):
    feats = rf_feature_importances(X_train, y_train)
    return list(feats.index[:k]), feats


def ig_feature_selection_with_scores(X_train, y_train, k):
    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    selector.fit(X_train, y_train)
    ig = pd.Series(selector.scores_, index=X_train.columns).sort_values(ascending=False)
    return list(ig.index[:k]), ig


def plot_fs_comparison(rf_scores, ig_scores, top_k=20, suffix=""):
    rf_top = rf_scores.head(top_k)
    ig_top = ig_scores.head(top_k)
    shared = set(rf_top.index) & set(ig_top.index)
    plt.figure(figsize=(10, 5)); rf_top.sort_values().plot.barh(); plt.title(f"RF {suffix}"); plt.tight_layout(); plt.show()
    plt.figure(figsize=(10, 5)); ig_top.sort_values().plot.barh(); plt.title(f"IG {suffix}"); plt.tight_layout(); plt.show()
    print(f"Top {top_k} overlap (RF ∩ IG): {len(shared)}")
    if shared: print(sorted(shared))


def plot_learning_curve(clf, X, y, suffix=""):
    try:
        sizes, train_scores, val_scores = learning_curve(clf, X, y, cv=5, scoring='accuracy', n_jobs=-1, train_sizes=[0.1,0.3,0.5,0.7,1.0])
        plt.figure(figsize=(7, 5))
        plt.plot(sizes, np.mean(train_scores, 1), 'o-', label="Train")
        plt.plot(sizes, np.mean(val_scores, 1), 'o-', label="Validation")
        plt.title(f"Learning Curve – SGD {suffix}"); plt.xlabel("Training examples"); plt.ylabel("Accuracy"); plt.legend(); plt.tight_layout(); plt.show()
    except Exception as e:
        print("Unable to plot learning curve:", e)


def load_and_prepare_multiclass(csv_path):
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
        df["Attack Type"] = df["Class"].astype(str).str.strip()
    df["Attack Type"] = df["Attack Type"].astype(str).str.strip().fillna("UNKNOWN")
    cat = pd.Categorical(df["Attack Type"])
    class_names = list(cat.categories)
    df["Attack_Label_Code"] = cat.codes
    y = df["Attack_Label_Code"].astype(int)
    exclude = {"Attack Type", "Attack_Label_Code"}
    obj_cols = [c for c in df.select_dtypes(include=["object", "category"]).columns if c not in exclude]
    if obj_cols:
        df = pd.get_dummies(df, columns=obj_cols, drop_first=True)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in num_cols if c not in ["Attack_Label_Code"]]
    if not feature_cols:
        raise ValueError("No feature columns found after preprocessing.")
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(df[feature_cols].mean())
    return X, y, class_names


def run_sgd_multiclass(name, X_train, X_test, y_train, y_test, class_names, *,
                       loss='log_loss', penalty='l2', alpha=1e-4, max_iter=1000,
                       class_weight=None, use_sample_weight=False, save=False, outdir="outputs"):
    stamp = timestamp()
    model_dir = Path(outdir) / f"{name.replace(' ', '_')}_{stamp}"
    if save:
        ensure_dir(model_dir)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    clf = SGDClassifier(loss=loss, penalty=penalty, alpha=alpha, max_iter=max_iter, random_state=RND, class_weight=class_weight)
    # sample_weight based on class frequencies if requested
    sample_weight = None
    if use_sample_weight and class_weight is None:
        try:
            classes = np.unique(y_train)
            cw = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
            cw_map = {int(c): float(w) for c, w in zip(classes, cw)}
            sample_weight = np.array([cw_map[int(lbl)] for lbl in y_train])
        except Exception:
            sample_weight = None

    if sample_weight is not None:
        clf.fit(X_train_sc, y_train, sample_weight=sample_weight)
    else:
        clf.fit(X_train_sc, y_train)

    y_pred = clf.predict(X_test_sc)
    # probabilities
    try:
        y_prob = clf.predict_proba(X_test_sc)
    except Exception:
        n_classes = len(class_names)
        y_prob = np.zeros((len(y_pred), n_classes))
        for i, p in enumerate(y_pred):
            y_prob[i, int(p)] = 1.0

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1v = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    report = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    try:
        y_test_bin = label_binarize(y_test, classes=list(range(len(class_names))))
        roc_auc = float(roc_auc_score(y_test_bin, y_prob, average="macro", multi_class="ovr"))
    except Exception:
        roc_auc = None

    print(f"\n=== {name} (SGD) ===")
    print(f"Classes: {class_names}")
    print(f"Accuracy: {acc:.4f}  Precision (weighted): {prec:.4f}  Recall (weighted): {rec:.4f}  F1 (weighted): {f1v:.4f}  AUC (macro OVR): {roc_auc if roc_auc is not None else 'N/A'}")
    print("\nClassification report:\n", report)

    # plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0], cbar=False, xticklabels=class_names, yticklabels=class_names)
    axes[0].set_title(f"Confusion Matrix - {name} (SGD)"); axes[0].tick_params(axis="x", rotation=45)

    axes[1].plot([0, 1], [0, 1], "k--", alpha=0.6)
    if roc_auc is not None:
        try:
            fpr = dict(); tpr = dict(); n_classes = len(class_names)
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_test == i, y_prob[:, i])
            for i in range(n_classes):
                axes[1].plot(fpr[i], tpr[i], lw=1.5, label=f"{class_names[i]} (AUC {auc(fpr[i], tpr[i]):.2f})")
            axes[1].legend(loc="lower right", fontsize="small")
        except Exception:
            axes[1].text(0.5, 0.5, "ROC N/A", ha="center")
    else:
        axes[1].text(0.5, 0.5, "ROC N/A", ha="center")
    axes[1].set_title(f"ROC Curves - {name} (SGD)"); axes[1].set_xlabel("False Positive Rate"); axes[1].set_ylabel("True Positive Rate")
    plt.tight_layout(); plt.show()

    try:
        plot_learning_curve(SGDClassifier(loss=loss, penalty=penalty, alpha=alpha, random_state=RND), X_train, y_train, suffix=f"({name})")
    except Exception:
        pass

    if save:
        try: joblib.dump(clf, Path(outdir)/f"sgd_model_{timestamp()}.joblib")
        except Exception: pass
        try: joblib.dump(scaler, Path(outdir)/f"scaler_{timestamp()}.joblib")
        except Exception: pass

    return {"config": name, "algorithm": "SGD", "loss": loss, "penalty": penalty, "alpha": alpha, "max_iter": int(max_iter), "n_classes": int(len(class_names)), "accuracy": acc, "precision_weighted": prec, "recall_weighted": rec, "f1_weighted": f1v, "auc_macro_ovr": roc_auc}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="../SDN-Net.csv")
    parser.add_argument("--outdir", default="outputs")
    parser.add_argument("--k", type=int, default=30)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--loss", default="log_loss")
    parser.add_argument("--penalty", default="l2")
    parser.add_argument("--alpha", type=float, default=1e-4)
    parser.add_argument("--max_iter", type=int, default=1000)
    parser.add_argument("--class_weight", action="store_true")
    args = parser.parse_args(argv)

    X_full, y_full, class_names = load_and_prepare_multiclass(args.csv)
    print("Dataset shape (features):", X_full.shape)
    print("Number of classes:", len(class_names))

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_full, y_full, test_size=0.33, random_state=RND, stratify=y_full)

    results = []

    cw = "balanced" if args.class_weight else None

    # Model 1
    results.append(run_sgd_multiclass("Model1_Unbalanced_Full", X_train_raw, X_test_raw, y_train, y_test, class_names,
                                      loss=args.loss, penalty=args.penalty, alpha=args.alpha, max_iter=args.max_iter, class_weight=cw, use_sample_weight=False, save=args.save, outdir=args.outdir))

    # Model 2
    sel2, rf_scores2 = rf_feature_selection(X_train_raw, y_train, args.k)
    if args.save:
        ensure_dir(args.outdir)
        pd.Series(sel2, name="feature").to_csv(Path(args.outdir)/f"features_model2_{timestamp()}.csv", index=False)
    results.append(run_sgd_multiclass("Model2_Unbalanced_RF-FS", X_train_raw[sel2], X_test_raw[sel2], y_train, y_test, class_names,
                                      loss=args.loss, penalty=args.penalty, alpha=args.alpha, max_iter=args.max_iter, class_weight=cw, save=args.save, outdir=args.outdir))

    # Model 3
    sel3, ig_scores3 = ig_feature_selection_with_scores(X_train_raw, y_train, args.k)
    if args.save:
        pd.Series(sel3, name="feature").to_csv(Path(args.outdir)/f"features_model3_{timestamp()}.csv", index=False)
    results.append(run_sgd_multiclass("Model3_Unbalanced_IG-FS", X_train_raw[sel3], X_test_raw[sel3], y_train, y_test, class_names,
                                      loss=args.loss, penalty=args.penalty, alpha=args.alpha, max_iter=args.max_iter, class_weight=cw, save=args.save, outdir=args.outdir))

    # Model 4
    smote = SMOTE(random_state=RND)
    X4_arr, y4 = smote.fit_resample(X_train_raw, y_train)
    X_train_m4 = pd.DataFrame(X4_arr, columns=X_train_raw.columns)
    results.append(run_sgd_multiclass("Model4_SMOTE_Full", X_train_m4, X_test_raw, y4, y_test, class_names,
                                      loss=args.loss, penalty=args.penalty, alpha=args.alpha, max_iter=args.max_iter, class_weight=None, use_sample_weight=False, save=args.save, outdir=args.outdir))

    # Model 5
    sel5, rf_scores5 = rf_feature_selection(X_train_m4, y4, args.k)
    if args.save:
        pd.Series(sel5, name="feature").to_csv(Path(args.outdir)/f"features_model5_{timestamp()}.csv", index=False)
    results.append(run_sgd_multiclass("Model5_SMOTE_RF-FS", X_train_m4[sel5], X_test_raw[sel5], y4, y_test, class_names,
                                      loss=args.loss, penalty=args.penalty, alpha=args.alpha, max_iter=args.max_iter, class_weight=None, save=args.save, outdir=args.outdir))

    # Model 6
    sel6, ig_scores6 = ig_feature_selection_with_scores(X_train_m4, y4, args.k)
    if args.save:
        pd.Series(sel6, name="feature").to_csv(Path(args.outdir)/f"features_model6_{timestamp()}.csv", index=False)
    results.append(run_sgd_multiclass("Model6_SMOTE_IG-FS", X_train_m4[sel6], X_test_raw[sel6], y4, y_test, class_names,
                                      loss=args.loss, penalty=args.penalty, alpha=args.alpha, max_iter=args.max_iter, class_weight=None, save=args.save, outdir=args.outdir))

    # Model 7
    smt = SMOTETomek(random_state=RND)
    X7_arr, y7 = smt.fit_resample(X_train_raw, y_train)
    X_train_m7 = pd.DataFrame(X7_arr, columns=X_train_raw.columns)
    results.append(run_sgd_multiclass("Model7_SMOTETomek_Full", X_train_m7, X_test_raw, y7, y_test, class_names,
                                      loss=args.loss, penalty=args.penalty, alpha=args.alpha, max_iter=args.max_iter, class_weight=None, save=args.save, outdir=args.outdir))

    summary = pd.DataFrame(results)
    print("\n==== FINAL SUMMARY ====\n", summary)
    if args.save:
        ensure_dir(args.outdir)
        summary.to_csv(Path(args.outdir)/f"summary_{timestamp()}.csv", index=False)


if __name__ == "__main__":
    main()
