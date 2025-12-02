#!/usr/bin/env python3
"""
dt_binary.py

Notebook -> script conversion.

What this script does:
 - Ensures a CSV dataset is available locally (use local path or download from Zenodo record).
 - If Zenodo returns an archive (zip/tar), it extracts and finds the first CSV.
 - Runs 7 DecisionTree binary experiments:
     1) Unbalanced full
     2) Unbalanced + RF-FS
     3) Unbalanced + IG-FS
     4) SMOTE balanced (full)
     5) SMOTE + RF-FS
     6) SMOTE + IG-FS
     7) SMOTE-Tomek balanced
 - For each variant displays classification report, confusion matrix, ROC, and learning curve.
 - For FS variants prints selected features and compares RF importances vs IG scores.
 - Optionally saves models/metrics/plots (--save) and optionally pauses between variants (--pause).

Usage examples:
  python dt_binary.py --csv /path/to/SDN-Net.csv --k 30
  python dt_binary.py --zenodo-record 17761467 --k 30 --save --outdir outputs

Dependencies:
  pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn joblib requests zenodo-get

Notes:
 - If you install zenodo-get during a Jupyter session, restart kernel/terminal so the CLI becomes available.
 - On headless servers, use --save so plots are written to disk rather than only shown interactively.
"""
import argparse
import json
import subprocess
import zipfile
import tarfile
import time
from pathlib import Path
from datetime import datetime

import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             classification_report, confusion_matrix, roc_curve, auc)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif

from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek

sns.set(style="whitegrid")
RND = 42
ZENODO_API = "https://zenodo.org/api/records/{}"


# ---------------------------
# Utilities: I/O / Zenodo
# ---------------------------
def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


def is_command_available(cmd):
    try:
        subprocess.run([cmd, "--help"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False


def download_with_zenodo_get(record_id, outdir="data"):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(["zenodo_get", str(record_id), "-o", str(outdir)], check=True)
        csvs = list(outdir.rglob("*.csv"))
        if csvs:
            return str(csvs[0])
        return None
    except Exception as e:
        print("zenodo_get failed:", e)
        return None


def download_via_zenodo_api(record_id, outdir="data"):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    api_url = ZENODO_API.format(record_id)
    resp = requests.get(api_url, timeout=60)
    resp.raise_for_status()
    meta = resp.json()
    files = meta.get("files", [])
    # Prefer CSV files
    for f in files:
        key = f.get("key", "")
        if key and key.lower().endswith(".csv"):
            link = f.get("links", {}).get("self") or f.get("links", {}).get("download")
            if link:
                outpath = outdir / key
                with requests.get(link, stream=True, timeout=120) as r:
                    r.raise_for_status()
                    with open(outpath, "wb") as fh:
                        for chunk in r.iter_content(chunk_size=32768):
                            if chunk:
                                fh.write(chunk)
                return str(outpath)
    # If no CSV, download first file (maybe an archive)
    if files:
        f = files[0]
        key = f.get("key")
        link = f.get("links", {}).get("self") or f.get("links", {}).get("download")
        if link and key:
            outpath = outdir / key
            with requests.get(link, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(outpath, "wb") as fh:
                    for chunk in r.iter_content(chunk_size=32768):
                        if chunk:
                            fh.write(chunk)
            return str(outpath)
    return None


def extract_archive(path, outdir=None):
    p = Path(path)
    dest = Path(outdir) if outdir else p.parent
    dest.mkdir(parents=True, exist_ok=True)
    extracted = []
    try:
        if zipfile.is_zipfile(p):
            with zipfile.ZipFile(p, "r") as zf:
                zf.extractall(dest)
                extracted = [str(dest / f) for f in zf.namelist()]
        elif tarfile.is_tarfile(p):
            with tarfile.open(p, "r:*") as tf:
                tf.extractall(dest)
                extracted = [str(dest / m.name) for m in tf.getmembers() if m.isfile()]
        else:
            return []
    except Exception as e:
        print("Extraction failed:", e)
        return []
    return extracted


def find_first_csv(path_or_dir):
    p = Path(path_or_dir)
    if p.is_file() and p.suffix.lower() == ".csv":
        return str(p)
    csvs = list(p.rglob("*.csv"))
    return str(csvs[0]) if csvs else None


def ensure_csv_available(csv_arg=None, zenodo_record=None, outdir="data"):
    """
    Ensure CSV is available locally, otherwise download from Zenodo and extract if needed.
    Returns path to CSV file (string).
    """
    # Local first
    if csv_arg:
        p = Path(csv_arg)
        if p.exists() and p.is_file():
            print("Using local CSV:", p)
            return str(p)

    # Determine record id
    record_id = zenodo_record or None
    if not record_id and csv_arg:
        s = str(csv_arg).strip()
        if s.isdigit():
            record_id = s
        elif s.lower().startswith("zenodo:"):
            record_id = s.split(":", 1)[1]

    if not record_id:
        raise FileNotFoundError("No local CSV and no Zenodo record provided.")

    print(f"Attempting to retrieve Zenodo record: {record_id}")

    csv_path = None
    # Try zenodo_get CLI first if available
    if is_command_available("zenodo_get"):
        try:
            csv_path = download_with_zenodo_get(record_id, outdir=outdir)
        except Exception as e:
            print("zenodo_get error:", e)
            csv_path = None

    # Fallback: REST API
    if not csv_path:
        csv_or_file = download_via_zenodo_api(record_id, outdir=outdir)
        if not csv_or_file:
            raise FileNotFoundError("No files available in Zenodo record")
        csv_path = csv_or_file

    p = Path(csv_path)
    # If it's an archive, extract and find CSV
    if p.suffix.lower() in (".zip", ".tar", ".gz", ".tgz", ".bz2") or zipfile.is_zipfile(p) or tarfile.is_tarfile(p):
        print("Downloaded file is an archive; extracting...")
        extracted = extract_archive(p, outdir=Path(outdir) / p.stem)
        found = find_first_csv(Path(outdir) / p.stem)
        if found:
            return found
        fallback = find_first_csv(outdir)
        if fallback:
            return fallback
        raise FileNotFoundError("No CSV found after extracting archive.")
    # If CSV file
    found = find_first_csv(p)
    if found:
        return found
    raise FileNotFoundError("CSV not found.")


# ---------------------------
# Data prep / Feature selection / plotting
# ---------------------------
def load_and_prepare(csv_path):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path, skipinitialspace=True, low_memory=False)
    # Drop unnamed index columns
    unnamed = [c for c in df.columns if c.startswith("Unnamed")]
    if unnamed:
        df.drop(columns=unnamed, inplace=True)
    # Ensure Attack Type
    if "Attack Type" not in df.columns:
        if "Class" not in df.columns:
            raise KeyError("CSV must contain 'Attack Type' or 'Class'.")
        df["Attack Type"] = df["Class"].astype(str).str.strip()
    # Binary label: NORMAL -> 0 else 1
    df["Attack_Binary_Label"] = df["Attack Type"].astype(str).str.strip().str.upper().apply(lambda x: 0 if x == "NORMAL" else 1)
    # One-hot encode non-label object columns
    exclude = {"Class", "Attack Type", "Attack_Binary_Label"}
    obj_cols = [c for c in df.select_dtypes(include=["object", "category"]).columns if c not in exclude]
    if obj_cols:
        df = pd.get_dummies(df, columns=obj_cols, drop_first=True)
    # Numeric cleanup
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan).fillna(df[num_cols].mean())
    # Features and labels
    drop_cols = [c for c in ["Class", "Attack Type", "Attack_Binary_Label"] if c in df.columns]
    X = df.drop(columns=drop_cols, errors="ignore")
    y = df["Attack_Binary_Label"].astype(int)
    return X, y


def rf_feature_importances(X_train, y_train):
    rf = RandomForestClassifier(n_estimators=200, random_state=RND, n_jobs=-1)
    rf.fit(X_train, y_train)
    importances = pd.Series(rf.feature_importances_, index=X_train.columns)
    importances = importances.sort_values(ascending=False)
    return importances


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
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    rf_top.sort_values().plot.barh(ax=axes[0], color="tab:blue")
    axes[0].set_title("Feature Importances (Random Forest) " + title_suffix)
    axes[0].set_xlabel("Importance")
    ig_top.sort_values().plot.barh(ax=axes[1], color="tab:green")
    axes[1].set_title("Feature Scores (Information Gain) " + title_suffix)
    axes[1].set_xlabel("Mutual Information Score")
    plt.tight_layout()
    plt.show()
    overlap = set(rf_top.index) & set(ig_top.index)
    print(f"Top {top_k} overlap count: {len(overlap)}")
    if overlap:
        print(sorted(list(overlap)))


def plot_learning_curve_for_model(estimator, X, y, title="Learning Curve", cv=5, n_jobs=1,
                                  train_sizes=np.linspace(0.1, 1.0, 5)):
    plt.figure(figsize=(8, 6))
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=cv, scoring="accuracy", train_sizes=train_sizes, n_jobs=n_jobs
    )
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    plt.plot(train_sizes, train_mean, "o-", color="r", label="Training score")
    plt.plot(train_sizes, val_mean, "o-", color="g", label="Cross-validation score")
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color="g")
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score (accuracy)")
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ---------------------------
# Training / evaluation / display
# ---------------------------
def train_dt_display(
    name,
    X_train_df,
    X_test_df,
    y_train,
    y_test,
    features,
    algo_name="DecisionTree",
    save_outputs=False,
    outdir="outputs",
    pause_between=False,
    show_fs_info=False,
    top_k_fs=20,
):
    stamp = timestamp()
    model_dir = Path(outdir) / f"{name.replace(' ', '_')}_{stamp}"
    if save_outputs:
        ensure_dir(model_dir)

    # Scale
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_df)
    X_test_sc = scaler.transform(X_test_df)

    # Choose classifier according to label (currently only DecisionTree implemented)
    if algo_name.lower() in ("decisiontree", "decision_tree", "dt"):
        clf = DecisionTreeClassifier(random_state=RND)
    else:
        print(f"Algo '{algo_name}' not recognized; using DecisionTree by default.")
        clf = DecisionTreeClassifier(random_state=RND)

    # Train
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

    # FS info (if requested)
    if show_fs_info and isinstance(features, (list, pd.Index, np.ndarray)):
        Xtr_fs = X_train_df.copy()
        try:
            rf_imp = rf_feature_importances(Xtr_fs, y_train)
        except Exception as e:
            print("Error computing RF importances:", e)
            rf_imp = pd.Series(dtype=float)
        try:
            _, ig_scores = ig_feature_selection_with_scores(Xtr_fs, y_train, k=min(len(Xtr_fs.columns), top_k_fs))
        except Exception as e:
            print("Error computing IG scores:", e)
            ig_scores = pd.Series(dtype=float)

        print("\nSelected features used for training (first 100 shown):")
        print(list(features)[:100])

        if not rf_imp.empty and not ig_scores.empty:
            plot_feature_selection_comparison(rf_imp, ig_scores, top_k=top_k_fs, title_suffix=f"({name})")
        else:
            if not rf_imp.empty:
                plt.figure(figsize=(8, 6))
                rf_imp.head(top_k_fs).sort_values().plot.barh(color="tab:blue")
                plt.title(f"RF Feature Importances ({name})")
                plt.tight_layout()
                plt.show()
            if not ig_scores.empty:
                plt.figure(figsize=(8, 6))
                ig_scores.head(top_k_fs).sort_values().plot.barh(color="tab:green")
                plt.title(f"IG Scores ({name})")
                plt.tight_layout()
                plt.show()

    # Confusion matrix & ROC side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0], cbar=False, xticklabels=[0, 1], yticklabels=[0, 1])
    axes[0].set_title(f"Confusion Matrix - {name} ({algo_name})")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    axes[1].plot([0, 1], [0, 1], "k--", alpha=0.6)
    if fpr is not None and tpr is not None:
        axes[1].plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.3f}")
    else:
        axes[1].text(0.5, 0.5, "ROC N/A", ha="center")
    axes[1].set_title(f"ROC Curve - {name} ({algo_name})")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].legend(loc="lower right")
    plt.tight_layout()
    plt.show()

    # Learning curve (accuracy)
    try:
        plot_learning_curve_for_model(DecisionTreeClassifier(random_state=RND), X_train_df, y_train, title=f"Learning Curve ({name} - {algo_name})", cv=5, n_jobs=-1)
    except Exception as e:
        print("Unable to plot learning curve:", e)

    # Save artifacts if requested
    if save_outputs:
        joblib.dump(clf, model_dir / f"{algo_name}_model_{stamp}.joblib")
        joblib.dump(scaler, model_dir / f"scaler_{stamp}.joblib")
        if features is not None:
            pd.Series(list(features), name="feature").to_csv(model_dir / f"features_{stamp}.csv", index=False)
        metrics = {"algorithm": algo_name, "accuracy": float(acc), "precision": float(prec), "recall": float(rec), "f1": float(f1v), "auc": float(roc_auc) if roc_auc is not None else None}
        with open(model_dir / f"metrics_{stamp}.json", "w", encoding="utf8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        with open(model_dir / f"classification_report_{stamp}.txt", "w", encoding="utf8") as f:
            f.write(report)
        fig.savefig(model_dir / f"confusion_roc_{stamp}.png", dpi=150)
        plt.close(fig)

    # Pause between variants if desired
    if pause_between:
        try:
            input("Press Enter to continue to next model...")
        except Exception:
            time.sleep(2)

    return {"config": name, "algorithm": algo_name, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1v, "auc": roc_auc}


# ---------------------------
# Runner
# ---------------------------
def run_all(csv_arg=None, zenodo_record=None, outdir="outputs", k=30, save_outputs=False, pause_between=False, algo="DecisionTree"):
    csv_path = ensure_csv_available(csv_arg, zenodo_record, outdir="data")
    print("Using CSV:", csv_path)
    X_full, y_full = load_and_prepare(csv_path)
    print("Dataset shape:", X_full.shape)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_full, y_full, test_size=0.33, random_state=RND, stratify=y_full)
    print("Train / Test shapes:", X_train_raw.shape, X_test_raw.shape)

    results = []
    # Model 1
    results.append(train_dt_display("Model1_Unbalanced_Full", X_train_raw, X_test_raw, y_train, y_test, X_full.columns, algo_name=algo, save_outputs=save_outputs, outdir=outdir, pause_between=pause_between))
    # Model 2 RF-FS
    sel2 = rf_feature_importances(X_train_raw, y_train).index[:k].tolist()
    results.append(train_dt_display("Model2_Unbalanced_RF-FS", X_train_raw[sel2], X_test_raw[sel2], y_train, y_test, sel2, algo_name=algo, save_outputs=save_outputs, outdir=outdir, pause_between=pause_between, show_fs_info=True, top_k_fs=k))
    # Model 3 IG-FS
    sel3, ig_scores_full = ig_feature_selection_with_scores(X_train_raw, y_train, k=k)
    results.append(train_dt_display("Model3_Unbalanced_IG-FS", X_train_raw[sel3], X_test_raw[sel3], y_train, y_test, sel3, algo_name=algo, save_outputs=save_outputs, outdir=outdir, pause_between=pause_between, show_fs_info=True, top_k_fs=k))
    # Model 4 SMOTE full
    smote = SMOTE(random_state=RND)
    X_train_m4_arr, y_train_m4 = smote.fit_resample(X_train_raw, y_train)
    X_train_m4 = pd.DataFrame(X_train_m4_arr, columns=X_train_raw.columns)
    results.append(train_dt_display("Model4_SMOTE_Full", X_train_m4, X_test_raw, y_train_m4, y_test, X_full.columns, algo_name=algo, save_outputs=save_outputs, outdir=outdir, pause_between=pause_between))
    # Model 5 SMOTE + RF-FS
    sel5 = rf_feature_importances(X_train_m4, y_train_m4).index[:k].tolist()
    results.append(train_dt_display("Model5_SMOTE_RF-FS", X_train_m4[sel5], X_test_raw[sel5], y_train_m4, y_test, sel5, algo_name=algo, save_outputs=save_outputs, outdir=outdir, pause_between=pause_between, show_fs_info=True, top_k_fs=k))
    # Model 6 SMOTE + IG-FS
    sel6, ig_scores_m4 = ig_feature_selection_with_scores(X_train_m4, y_train_m4, k=k)
    results.append(train_dt_display("Model6_SMOTE_IG-FS", X_train_m4[sel6], X_test_raw[sel6], y_train_m4, y_test, sel6, algo_name=algo, save_outputs=save_outputs, outdir=outdir, pause_between=pause_between, show_fs_info=True, top_k_fs=k))
    # Model 7 SMOTE-Tomek full
    smt = SMOTETomek(random_state=RND)
    X_train_m7_arr, y_train_m7 = smt.fit_resample(X_train_raw, y_train)
    X_train_m7 = pd.DataFrame(X_train_m7_arr, columns=X_train_raw.columns)
    results.append(train_dt_display("Model7_SMOTETomek_Full", X_train_m7, X_test_raw, y_train_m7, y_test, X_full.columns, algo_name=algo, save_outputs=save_outputs, outdir=outdir, pause_between=pause_between))

    results_df = pd.DataFrame(results)
    print("\n=== Summary of all 7 models ===")
    print(results_df)
    if save_outputs:
        ensure_dir(outdir)
        summary_path = Path(outdir) / f"summary_7models_{algo}_{timestamp()}.csv"
        results_df.to_csv(summary_path, index=False)
        print("Saved summary to:", summary_path)
    return results_df


# ---------------------------
# CLI entrypoint
# ---------------------------
def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=None, help="Path to CSV file or Zenodo record id")
    parser.add_argument("--zenodo-record", default=None, help="Zenodo record id (if provided use this to download)")
    parser.add_argument("--outdir", default="outputs", help="Output directory for saving artifacts")
    parser.add_argument("--k", type=int, default=30, help="Number of features for feature selection")
    parser.add_argument("--save", action="store_true", help="If set, save artifacts (models/plots/metrics)")
    parser.add_argument("--pause", action="store_true", help="If set, pause between variants (press Enter)")
    parser.add_argument("--algo", default="DecisionTree", help="Algorithm label (default DecisionTree)")
    args = parser.parse_args(argv)

    # Determine CSV source and run
    try:
        results_df = run_all(csv_arg=args.csv, zenodo_record=args.zenodo_record, outdir=args.outdir, k=args.k, save_outputs=args.save, pause_between=args.pause, algo=args.algo)
    except Exception as e:
        print("Error:", e)
        raise


if __name__ == "__main__":
    main()
