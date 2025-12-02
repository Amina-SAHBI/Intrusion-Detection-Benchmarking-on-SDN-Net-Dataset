#!/usr/bin/env python3
"""
knn_multiclass.py

k-Nearest Neighbors-based 7-variant benchmarking script for multiclass intrusion
detection on the SDN-Net dataset.

This script runs the same 7 experimental pipeline variants:
  - Unbalanced
  - Unbalanced + RF Feature Selection (RF-FS)
  - Unbalanced + Information Gain / Mutual Information FS (IG-FS)
  - SMOTE
  - SMOTE + RF-FS
  - SMOTE + IG-FS
  - SMOTE-Tomek

All variants use a k-Nearest Neighbors (KNN) classifier as the core model
for multiclass classification. The `algo_name` label is included in printouts,
plot titles and in the final summary so that results remain traceable.

Features:
 - 7 experiment variants per run
 - Inline display of metrics, confusion matrix and per-class ROC for each variant
 - For FS variants, prints selected features and compares RF importances vs IG scores
 - Learning curve (accuracy) per model as a proxy for convergence / capacity
 - "algorithm" field saved in metrics and summary (default: "KNN-Multiclass")
 - Optional saving (--save) and pausing between models (--pause)

Usage:
  python knn_multiclass.py --csv ../SDN-Net.csv --k 30
  python knn_multiclass.py --csv ../SDN-Net.csv --k 30 --save --outdir outputs --pause

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
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight

from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek

sns.set(style="whitegrid")
RND = 42


def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(d):
    Path(d).mkdir(parents=True, exist_ok=True)


def safe_map_attack_type(series):
    attacks_types = {
        'NORMAL': 'normal',
        'DOS': 'attack',
        'DDOS': 'attack',
        'WEB ATTACK � BRUTE FORCE': 'attack',
        'WEB ATTACK � XSS': 'attack',
        'WEB ATTACK � SQL INJECTION': 'attack',
        'WEB-ATTACK': 'attack',
        'U2R': 'attack',
        'PROBE': 'attack',
        'BFA': 'attack',
        'BOTNET': 'attack',
    }
    attacks_map = {str(k).strip().upper(): v for k, v in attacks_types.items()}
    normalized = series.astype(str).str.strip().str.upper()
    mapped = normalized.map(attacks_map).fillna('attack')
    return mapped


def rf_feature_importances(X_train, y_train):
    """Random Forest-based feature importance used for RF-FS variants."""
    rf = RandomForestClassifier(n_estimators=200, random_state=RND, n_jobs=-1)
    rf.fit(X_train, y_train)
    importances = pd.Series(rf.feature_importances_, index=X_train.columns)
    importances = importances.sort_values(ascending=False)
    return importances


def rf_feature_selection(X_train, y_train, k):
    importances = rf_feature_importances(X_train, y_train)
    selected = list(importances.index[:k])
    return selected


def ig_feature_selection_with_scores(X_train, y_train, k):
    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    selector.fit(X_train, y_train)
    scores = pd.Series(selector.scores_, index=X_train.columns).sort_values(ascending=False)
    selected = list(scores.index[:k])
    return selected, scores


def plot_feature_selection_comparison(rf_imp, ig_scores, top_k=20, title_suffix=""):
    rf_imp = rf_imp.sort_values(ascending=False)
    ig_scores = ig_scores.sort_values(ascending=False)

    rf_top = rf_imp.head(top_k)
    ig_top = ig_scores.head(top_k)

    rf_set = set(rf_top.index)
    ig_set = set(ig_top.index)
    overlap = rf_set & ig_set

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    rf_top.sort_values().plot.barh(ax=axes[0], color='tab:blue')
    axes[0].set_title("Feature Importances (Random Forest) " + title_suffix)
    axes[0].set_xlabel("Importance")

    ig_top.sort_values().plot.barh(ax=axes[1], color='tab:green')
    axes[1].set_title("Feature Scores (Information Gain) " + title_suffix)
    axes[1].set_xlabel("Mutual Information Score")

    plt.tight_layout()
    plt.show()

    print(f"Top {top_k} RF features count: {len(rf_top)}; Top {top_k} IG features count: {len(ig_top)}")
    print(f"Overlap count: {len(overlap)}")
    if overlap:
        print("Overlapping features (RF ∩ IG):")
        print(sorted(list(overlap)))
    else:
        print("No overlap in top features.")


def plot_learning_curve_for_model(estimator, X, y, title="Learning Curve", cv=5, n_jobs=1,
                                  train_sizes=np.linspace(0.1, 1.0, 5)):
    """
    Plot learning curve (train and cross-validation score) for estimator on data X, y.
    Uses accuracy as scoring.
    """
    plt.figure(figsize=(8, 6))
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=cv, scoring='accuracy',
        train_sizes=train_sizes, n_jobs=n_jobs
    )
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    plt.plot(train_sizes, train_mean, 'o-', color='r', label='Training score')
    plt.plot(train_sizes, val_mean, 'o-', color='g', label='Cross-validation score')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='r')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='g')
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score (accuracy)")
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def train_knn_display_multiclass(
    name,
    X_train_df,
    X_test_df,
    y_train,
    y_test,
    features,
    class_names,
    algo_name="KNN-Multiclass",
    n_neighbors=5,
    save_outputs=False,
    outdir="outputs",
    pause_between=False,
    show_fs_info=False,
    top_k_fs=20
):
    """
    Train classifier (k-Nearest Neighbors), display classification
    report, confusion matrix and per-class ROC inline for multiclass.

    class_names: list mapping label index -> label string
    """
    stamp = timestamp()
    model_dir = Path(outdir) / f"{name.replace(' ', '_')}_{stamp}"
    if save_outputs:
        ensure_dir(model_dir)

    # Ensure DataFrame inputs (SMOTE resampling may have produced arrays)
    if not isinstance(X_train_df, pd.DataFrame):
        X_train_df = pd.DataFrame(X_train_df, columns=features)
    if not isinstance(X_test_df, pd.DataFrame):
        X_test_df = pd.DataFrame(X_test_df, columns=features)

    # Build pipeline: scaler + KNN to ensure consistent preprocessing for training, prediction and learning curves
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1))
    ])

    # Train
    pipeline.fit(X_train_df, y_train)
    y_pred = pipeline.predict(X_test_df)

    # Probabilities for multiclass
    y_prob = None
    try:
        if hasattr(pipeline, "predict_proba"):
            y_prob = pipeline.predict_proba(X_test_df)  # shape (n_samples, n_classes)
    except Exception:
        y_prob = None
    if y_prob is None:
        # fallback: one-hot encode predictions (not ideal for ROC)
        n_classes = len(class_names)
        y_prob = np.zeros((len(y_pred), n_classes))
        for i, p in enumerate(y_pred):
            y_prob[i, int(p)] = 1.0

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1v = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    report = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    # ROC/AUC (macro OVR)
    try:
        y_test_bin = label_binarize(y_test, classes=list(range(len(class_names))))
        roc_auc = float(roc_auc_score(y_test_bin, y_prob, average='macro', multi_class='ovr'))
    except Exception:
        roc_auc = None

    # Print and display
    print(f"\n=== {name} ({algo_name}) ===")
    print(f"Classes: {class_names}")
    print(f"Accuracy: {acc:.4f}  Precision (weighted): {prec:.4f}  Recall (weighted): {rec:.4f}  F1 (weighted): {f1v:.4f}  AUC (macro OVR): {roc_auc if roc_auc is not None else 'N/A'}")
    print("\nClassification report:\n", report)

    # Feature selection info for FS variants
    if show_fs_info and isinstance(features, (list, pd.Index, np.ndarray)):
        Xtr_fs = X_train_df.copy()
        try:
            rf_imp = rf_feature_importances(Xtr_fs, y_train)
        except Exception as e:
            print("Error computing RF importances:", e)
            rf_imp = pd.Series(dtype=float)
        try:
            _, ig_scores = ig_feature_selection_with_scores(
                Xtr_fs, y_train, k=min(len(Xtr_fs.columns), top_k_fs)
            )
        except Exception as e:
            print("Error computing IG scores:", e)
            ig_scores = pd.Series(dtype=float)

        # Print selected features (the ones used for training)
        print("\nSelected features used for training (first 100 shown):")
        print(list(features)[:100])

        # Plot comparison
        if not rf_imp.empty and not ig_scores.empty:
            plot_feature_selection_comparison(
                rf_imp, ig_scores, top_k=top_k_fs, title_suffix=f"({name})"
            )
        else:
            if not rf_imp.empty:
                plt.figure(figsize=(8, 6))
                rf_imp.head(top_k_fs).sort_values().plot.barh(color='tab:blue')
                plt.title(f"RF Feature Importances ({name})")
                plt.tight_layout()
                plt.show()
            if not ig_scores.empty:
                plt.figure(figsize=(8, 6))
                ig_scores.head(top_k_fs).sort_values().plot.barh(color='tab:green')
                plt.title(f"IG Scores ({name})")
                plt.tight_layout()
                plt.show()

    # Side-by-side plot: Confusion Matrix and ROC
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        ax=axes[0],
        cbar=False,
        xticklabels=class_names,
        yticklabels=class_names
    )
    axes[0].set_title(f"Confusion Matrix - {name} ({algo_name})")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    axes[0].tick_params(axis='x', rotation=45)

    # ROC: plot per-class ROC curves if available
    axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.6)
    if roc_auc is not None:
        try:
            fpr = dict()
            tpr = dict()
            n_classes = len(class_names)
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_test == i, y_prob[:, i])
            for i in range(n_classes):
                axes[1].plot(fpr[i], tpr[i], lw=1.5, label=f"{class_names[i]} (AUC {auc(fpr[i], tpr[i]):.2f})")
            axes[1].legend(loc='lower right', fontsize='small')
        except Exception:
            axes[1].text(0.5, 0.5, 'ROC N/A', ha='center')
    else:
        axes[1].text(0.5, 0.5, 'ROC N/A', ha='center')
    axes[1].set_title(f"ROC Curves - {name} ({algo_name})")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    plt.tight_layout()
    plt.show()

    # Learning curve (accuracy) as a proxy for loss curve
    try:
        pipeline_for_lc = Pipeline([
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier(n_neighbors=n_neighbors))
        ])
        plot_learning_curve_for_model(
            pipeline_for_lc,
            X_train_df,
            y_train,
            title=f"Learning Curve ({name} - {algo_name})",
            cv=5,
            n_jobs=-1
        )
    except Exception as e:
        print("Unable to plot learning curve:", e)

    # Optionally save artifacts
    if save_outputs:
        joblib.dump(pipeline, model_dir / f"{algo_name}_pipeline_{stamp}.joblib")
        try:
            scaler = pipeline.named_steps.get('scaler', None)
            if scaler is not None:
                joblib.dump(scaler, model_dir / f"scaler_{stamp}.joblib")
        except Exception:
            pass

        if features is not None:
            pd.Series(list(features), name="feature").to_csv(
                model_dir / f"features_{stamp}.csv", index=False
            )

        # Save class mapping
        classes_path = model_dir / f"classes_{stamp}.json"
        try:
            with open(classes_path, "w", encoding="utf8") as f:
                json.dump({"classes": class_names}, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

        # safe guard for auc
        auc_value = None
        try:
            if roc_auc is not None and not (isinstance(roc_auc, float) and np.isnan(roc_auc)):
                auc_value = float(roc_auc)
        except Exception:
            auc_value = None

        metrics = {
            "algorithm": algo_name,
            "n_neighbors": int(n_neighbors),
            "n_classes": int(len(class_names)),
            "accuracy": float(acc),
            "precision_weighted": float(prec),
            "recall_weighted": float(rec),
            "f1_weighted": float(f1v),
            "auc_macro_ovr": auc_value
        }
        with open(model_dir / f"metrics_{stamp}.json", "w", encoding="utf8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        with open(model_dir / f"classification_report_{stamp}.txt", "w", encoding="utf8") as f:
            f.write(report)
        try:
            fig.savefig(model_dir / f"confusion_roc_{stamp}.png", dpi=150)
            plt.close(fig)
        except Exception:
            pass

    # Pause option useful in notebooks
    if pause_between:
        try:
            input("Press Enter to continue to next model...")
        except Exception:
            time.sleep(2)

    return {
        "config": name,
        "algorithm": algo_name,
        "n_neighbors": int(n_neighbors),
        "n_classes": int(len(class_names)),
        "accuracy": acc,
        "precision_weighted": prec,
        "recall_weighted": rec,
        "f1_weighted": f1v,
        "auc_macro_ovr": roc_auc
    }


def load_and_prepare_multiclass(csv_path):
    """
    Loads CSV and returns (X, y, class_names).

    - X: DataFrame of features (object columns one-hot encoded except the label column)
    - y: integer labels (0..n_classes-1)
    - class_names: list mapping label index -> original Attack Type string
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path, skipinitialspace=True, low_memory=False)
    # Drop unnamed index columns
    unnamed = [c for c in df.columns if c.startswith("Unnamed")]
    if unnamed:
        df.drop(columns=unnamed, inplace=True)

    # Prefer 'Attack Type' column or 'Class'
    if "Attack Type" not in df.columns:
        if "Class" not in df.columns:
            raise KeyError("CSV must contain 'Attack Type' or 'Class'.")
        df['Attack Type'] = df['Class'].astype(str).str.strip()
    else:
        df['Attack Type'] = df['Attack Type'].astype(str).str.strip()

    # Fill missing labels with 'UNKNOWN'
    df['Attack Type'] = df['Attack Type'].fillna("UNKNOWN")

    # Preserve original class names (ordered categories)
    cat = pd.Categorical(df['Attack Type'])
    class_names = list(cat.categories)
    df['Attack_Label_Code'] = cat.codes
    y = df['Attack_Label_Code'].astype(int)

    # One-hot encode object cols except the label 'Attack Type' and the code column
    exclude = {'Attack Type', 'Attack_Label_Code'}
    obj_cols = [c for c in df.select_dtypes(include=['object', 'category']).columns if c not in exclude]
    if obj_cols:
        df = pd.get_dummies(df, columns=obj_cols, drop_first=True)

    # Numeric cleanup and select features (exclude label columns)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    drop_cols = ['Attack_Label_Code']
    feature_cols = [c for c in num_cols if c not in drop_cols]
    if not feature_cols:
        raise ValueError("No numeric or encoded features found after preprocessing.")
    X = df[feature_cols]
    # Replace infinities and fill NaNs
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.mean())

    return X, y, class_names


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="../SDN-Net.csv", help="Path to SDN-Net CSV")
    parser.add_argument("--outdir", default="outputs", help="Output dir (only used if --save)")
    parser.add_argument("--k", type=int, default=30, help="Number of features for FS")
    parser.add_argument("--save", action="store_true", help="If set, save artifacts to outdir (default: False)")
    parser.add_argument("--pause", action="store_true", help="Pause between models (press Enter). Default: False")
    parser.add_argument("--algo", default="KNN-Multiclass", help="Algorithm name label to include in results (default: KNN-Multiclass)")
    parser.add_argument("--no-gpu", action="store_true", help="If set, disable GPU usage (use CPU only) - kept for API parity")
    parser.add_argument("--n_neighbors", type=int, default=5, help="Number of neighbours (k) for KNN (default: 5)")
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        print("Ignored unknown args (likely from Jupyter):", unknown)

    X_full, y_full, class_names = load_and_prepare_multiclass(args.csv)
    print("Dataset shape (features):", X_full.shape)
    print("Number of classes:", len(class_names))
    print("Classes:", class_names)

    # Train/test split (stratified)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_full, y_full, test_size=0.33, random_state=RND, stratify=y_full
    )
    print("Train / Test shapes:", X_train_raw.shape, X_test_raw.shape)

    results = []

    # Model 1: Unbalanced full
    results.append(
        train_knn_display_multiclass(
            "Model1_Unbalanced_Full",
            X_train_raw,
            X_test_raw,
            y_train,
            y_test,
            X_full.columns,
            class_names,
            algo_name=args.algo,
            n_neighbors=args.n_neighbors,
            save_outputs=args.save,
            outdir=args.outdir,
            pause_between=args.pause
        )
    )

    # Model 2: Unbalanced + RF-FS
    sel2 = rf_feature_selection(X_train_raw, y_train, args.k)
    results.append(
        train_knn_display_multiclass(
            "Model2_Unbalanced_RF-FS",
            X_train_raw[sel2],
            X_test_raw[sel2],
            y_train,
            y_test,
            sel2,
            class_names,
            algo_name=args.algo,
            n_neighbors=args.n_neighbors,
            save_outputs=args.save,
            outdir=args.outdir,
            pause_between=args.pause,
            show_fs_info=True,
            top_k_fs=args.k
        )
    )

    # Model 3: Unbalanced + IG-FS
    sel3, ig_scores_full = ig_feature_selection_with_scores(X_train_raw, y_train, k=args.k)
    results.append(
        train_knn_display_multiclass(
            "Model3_Unbalanced_IG-FS",
            X_train_raw[sel3],
            X_test_raw[sel3],
            y_train,
            y_test,
            sel3,
            class_names,
            algo_name=args.algo,
            n_neighbors=args.n_neighbors,
            save_outputs=args.save,
            outdir=args.outdir,
            pause_between=args.pause,
            show_fs_info=True,
            top_k_fs=args.k
        )
    )

    # Model 4: SMOTE balanced (full)
    smote = SMOTE(random_state=RND)
    X_train_m4_arr, y_train_m4 = smote.fit_resample(X_train_raw, y_train)
    X_train_m4 = pd.DataFrame(X_train_m4_arr, columns=X_train_raw.columns)
    results.append(
        train_knn_display_multiclass(
            "Model4_SMOTE_Full",
            X_train_m4,
            X_test_raw,
            y_train_m4,
            y_test,
            X_full.columns,
            class_names,
            algo_name=args.algo,
            n_neighbors=args.n_neighbors,
            save_outputs=args.save,
            outdir=args.outdir,
            pause_between=args.pause
        )
    )

    # Model 5: SMOTE + RF-FS
    sel5 = rf_feature_selection(X_train_m4, y_train_m4, args.k)
    results.append(
        train_knn_display_multiclass(
            "Model5_SMOTE_RF-FS",
            X_train_m4[sel5],
            X_test_raw[sel5],
            y_train_m4,
            y_test,
            sel5,
            class_names,
            algo_name=args.algo,
            n_neighbors=args.n_neighbors,
            save_outputs=args.save,
            outdir=args.outdir,
            pause_between=args.pause,
            show_fs_info=True,
            top_k_fs=args.k
        )
    )

    # Model 6: SMOTE + IG-FS
    sel6, ig_scores_m4 = ig_feature_selection_with_scores(X_train_m4, y_train_m4, k=args.k)
    results.append(
        train_knn_display_multiclass(
            "Model6_SMOTE_IG-FS",
            X_train_m4[sel6],
            X_test_raw[sel6],
            y_train_m4,
            y_test,
            sel6,
            class_names,
            algo_name=args.algo,
            n_neighbors=args.n_neighbors,
            save_outputs=args.save,
            outdir=args.outdir,
            pause_between=args.pause,
            show_fs_info=True,
            top_k_fs=args.k
        )
    )

    # Model 7: SMOTE-Tomek balanced (full)
    smt = SMOTETomek(random_state=RND)
    X_train_m7_arr, y_train_m7 = smt.fit_resample(X_train_raw, y_train)
    X_train_m7 = pd.DataFrame(X_train_m7_arr, columns=X_train_raw.columns)
    results.append(
        train_knn_display_multiclass(
            "Model7_SMOTETomek_Full",
            X_train_m7,
            X_test_raw,
            y_train_m7,
            y_test,
            X_full.columns,
            class_names,
            algo_name=args.algo,
            n_neighbors=args.n_neighbors,
            save_outputs=args.save,
            outdir=args.outdir,
            pause_between=args.pause
        )
    )

    # Summary of all 7 models (includes algorithm column)
    results_df = pd.DataFrame(results)
    print("\n=== Summary of all 7 models ===")
    print(results_df)

    if args.save:
        ensure_dir(args.outdir)
        summary_path = Path(args.outdir) / f"summary_7models_{args.algo}_{timestamp()}.csv"
        results_df.to_csv(summary_path, index=False)
        # Save class mapping to outdir root too
        classes_root = Path(args.outdir) / f"classes_{timestamp()}.json"
        try:
            with open(classes_root, "w", encoding="utf8") as f:
                json.dump({"classes": class_names}, f, indent=2, ensure_ascii=False)
            print("Saved class mapping to:", classes_root)
        except Exception:
            pass
        print("Saved summary to:", summary_path)


if __name__ == "__main__":
    main()
