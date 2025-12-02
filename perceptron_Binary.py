#!/usr/bin/env python3
"""
perceptron_binary.py

Perceptron / MLP-based 7-variant benchmarking script for binary intrusion detection on the SDN-Net dataset.

Replaces the previous CNN/RNN with a simple perceptron / MLP (configurable).
Preserves the same 7 experimental pipeline variants:
  - Unbalanced
  - Unbalanced + RF Feature Selection (RF-FS)
  - Unbalanced + Information Gain / Mutual Information FS (IG-FS)
  - SMOTE
  - SMOTE + RF-FS
  - SMOTE + IG-FS
  - SMOTE-Tomek

Features:
 - 7 experiment variants per run (now using Perceptron / MLP)
 - Training/validation with EarlyStopping and optional ModelCheckpoint
 - Optional GPU disabling via --no-gpu
 - Feature selection comparisons (RF importances vs IG scores) for FS variants
 - Learning curves (training & validation loss/accuracy)
 - "algorithm" field saved in metrics and summary (default: "Perceptron")
 - Optional saving (--save) and pausing between models (--pause)

Usage:
  python perceptron_binary.py --csv ../SDN-Net.csv --k 30 --epochs 30 --batch_size 128 --hidden_layers 1 --hidden_units 64 --save

Dependencies:
  pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn joblib tensorflow
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.utils.class_weight import compute_class_weight

from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek

sns.set(style="whitegrid")
RND = 42

# Try to import TensorFlow
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, InputLayer, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
except Exception as e:
    raise ImportError(
        "TensorFlow is required for this script. Install with 'pip install tensorflow'.\n"
        f"Original error: {e}"
    )


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


def build_perceptron_model(input_dim, hidden_layers=1, hidden_units=64, dropout=0.3, lr=1e-3):
    """
    Build and compile a perceptron / MLP model for binary classification.
    If hidden_layers == 0 -> single-layer logistic (no hidden layers)
    Otherwise builds the specified number of hidden Dense layers with ReLU.
    """
    model = Sequential()
    model.add(InputLayer(input_shape=(input_dim,)))
    if hidden_layers <= 0:
        # Single-layer perceptron (logistic regression)
        model.add(Dense(1, activation='sigmoid'))
    else:
        for i in range(hidden_layers):
            units = int(hidden_units)
            model.add(Dense(units, activation='relu'))
            model.add(BatchNormalization())
            if dropout and dropout > 0.0:
                model.add(Dropout(dropout))
        model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def plot_history_learning_curve(history, title="Training History"):
    """Plot training & validation accuracy and loss from Keras history."""
    if history is None:
        return
    hs = history.history
    epochs = range(1, len(hs.get('loss', [])) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, hs.get('loss', []), 'r-', label='Training loss')
    if 'val_loss' in hs:
        plt.plot(epochs, hs.get('val_loss', []), 'g--', label='Validation loss')
    plt.title(f"{title} - Loss")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, hs.get('accuracy', []), 'r-', label='Training acc')
    if 'val_accuracy' in hs:
        plt.plot(epochs, hs.get('val_accuracy', []), 'g--', label='Validation acc')
    plt.title(f"{title} - Accuracy")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()


def prepare_for_perceptron(X_df, timesteps=1):
    """
    Prepare dataframe X_df for perceptron/MLP input.
    Perceptron expects flat 2D input (n_samples, n_features).
    timesteps parameter is ignored for perceptron; included for CLI compatibility.
    """
    if not isinstance(X_df, pd.DataFrame):
        X_df = pd.DataFrame(X_df)
    arr = X_df.values.astype('float32')
    return arr  # shape (n_samples, n_features)


def train_perceptron_display(
    name,
    X_train_df,
    X_test_df,
    y_train,
    y_test,
    features,
    algo_name="Perceptron",
    hidden_layers=1,
    hidden_units=64,
    dropout=0.3,
    epochs=30,
    batch_size=128,
    patience=5,
    validation_split=0.15,
    save_outputs=False,
    outdir="outputs",
    pause_between=False,
    show_fs_info=False,
    top_k_fs=20,
    use_class_weight=True,
    optimizer_lr=1e-3,
    timesteps=1
):
    """
    Train perceptron/MLP model, display classification report, confusion matrix and ROC inline.

    Returns a dict of metrics.
    """
    stamp = timestamp()
    model_dir = Path(outdir) / f"{name.replace(' ', '_')}_{stamp}"
    if save_outputs:
        ensure_dir(model_dir)

    # Ensure DataFrame inputs and consistent columns
    if not isinstance(X_train_df, pd.DataFrame):
        X_train_df = pd.DataFrame(X_train_df, columns=features)
    if not isinstance(X_test_df, pd.DataFrame):
        X_test_df = pd.DataFrame(X_test_df, columns=features)

    # Feature selection info if requested
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

        print("\nSelected features used for training (first 100 shown):")
        print(list(features)[:100])

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

    # Scale features
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_df)
    X_test_sc = scaler.transform(X_test_df)

    # Prepare for perceptron (flatten)
    try:
        X_train_p = prepare_for_perceptron(pd.DataFrame(X_train_sc, columns=X_train_df.columns), timesteps=timesteps)
        X_test_p = prepare_for_perceptron(pd.DataFrame(X_test_sc, columns=X_test_df.columns), timesteps=timesteps)
    except Exception as e:
        raise RuntimeError(f"Error preparing data for Perceptron: {e}")

    # Build model
    input_dim = X_train_p.shape[1]
    model = build_perceptron_model(input_dim, hidden_layers=hidden_layers, hidden_units=hidden_units, dropout=dropout, lr=optimizer_lr)

    # Callbacks
    callbacks = []
    early = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    callbacks.append(early)
    if save_outputs:
        ckpt_path = str(model_dir / f"best_model_{stamp}.h5")
        callbacks.append(ModelCheckpoint(ckpt_path, monitor='val_loss', save_best_only=True, save_weights_only=False))

    # Class weights for imbalanced unbalanced variants (if requested)
    class_weight = None
    if use_class_weight:
        try:
            classes = np.unique(y_train)
            cw = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
            class_weight = {int(c): float(w) for c, w in zip(classes, cw)}
        except Exception:
            class_weight = None

    # Train
    history = model.fit(
        X_train_p, y_train,
        validation_split=validation_split,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=2
    )

    # Predict probabilities and labels
    y_prob = model.predict(X_test_p).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1v = f1_score(y_test, y_pred, zero_division=0)
    try:
        roc_auc = float(roc_auc_score(y_test, y_prob))
    except Exception:
        roc_auc = None
    report = classification_report(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    # Print
    print(f"\n=== {name} ({algo_name}) ===")
    print(f"Accuracy: {acc:.4f}  Precision: {prec:.4f}  Recall: {rec:.4f}  F1: {f1v:.4f}  AUC: {roc_auc if roc_auc is not None else 'N/A'}")
    print("\nClassification report:\n", report)

    # Confusion & ROC plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        ax=axes[0],
        cbar=False,
        xticklabels=[0, 1],
        yticklabels=[0, 1]
    )
    axes[0].set_title(f"Confusion Matrix - {name} ({algo_name})")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.6)
    try:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        if roc_auc is not None:
            axes[1].plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.3f}")
    except Exception:
        axes[1].text(0.5, 0.5, 'ROC N/A', ha='center')
    axes[1].set_title(f"ROC Curve - {name} ({algo_name})")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].legend(loc='lower right')
    plt.tight_layout()
    plt.show()

    # Learning curve from history
    try:
        plot_history_learning_curve(history, title=f"Learning Curve ({name} - {algo_name})")
    except Exception as e:
        print("Unable to plot training history:", e)

    # Save artifacts
    if save_outputs:
        # Save model (SavedModel directory) and scaler and features and metrics and history
        try:
            model_save_dir = model_dir / f"perceptron_model_{stamp}"
            model.save(str(model_save_dir), include_optimizer=True)
        except Exception as e:
            # fallback to h5
            try:
                model.save(str(model_dir / f"perceptron_model_{stamp}.h5"))
            except Exception as e2:
                print("Unable to save model:", e, e2)

        try:
            joblib.dump(scaler, model_dir / f"scaler_{stamp}.joblib")
        except Exception as e:
            print("Unable to save scaler:", e)

        if features is not None:
            pd.Series(list(features), name="feature").to_csv(
                model_dir / f"features_{stamp}.csv", index=False
            )

        # metrics
        metrics = {
            "algorithm": algo_name,
            "hidden_layers": int(hidden_layers),
            "hidden_units": int(hidden_units),
            "dropout": float(dropout),
            "epochs_ran": int(len(history.history.get('loss', []))),
            "batch_size": int(batch_size),
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1v),
            "auc": float(roc_auc) if roc_auc is not None else None
        }
        with open(model_dir / f"metrics_{stamp}.json", "w", encoding="utf8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        with open(model_dir / f"classification_report_{stamp}.txt", "w", encoding="utf8") as f:
            f.write(report)

        # save history
        try:
            hist_obj = history.history
            with open(model_dir / f"history_{stamp}.json", "w", encoding="utf8") as f:
                json.dump(hist_obj, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

        try:
            fig.savefig(model_dir / f"confusion_roc_{stamp}.png", dpi=150)
            plt.close(fig)
        except Exception:
            pass

    if pause_between:
        try:
            input("Press Enter to continue to next model...")
        except Exception:
            time.sleep(2)

    return {
        "config": name,
        "algorithm": algo_name,
        "hidden_layers": int(hidden_layers),
        "hidden_units": int(hidden_units),
        "batch_size": int(batch_size),
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1v,
        "auc": roc_auc
    }


def load_and_prepare(csv_path):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path, skipinitialspace=True, low_memory=False)
    # Drop unnamed index columns
    unnamed = [c for c in df.columns if c.startswith("Unnamed")]
    if unnamed:
        df.drop(columns=unnamed, inplace=True)
    # Map or ensure Attack Type
    if "Attack Type" not in df.columns:
        if "Class" not in df.columns:
            raise KeyError("CSV must contain 'Attack Type' or 'Class'.")
        df['Attack Type'] = df['Class'].astype(str).str.strip()

    # Use safe mapping helper so various Attack Type strings map consistently
    df['Attack_Mapped'] = safe_map_attack_type(df['Attack Type'])
    df['Attack_Binary_Label'] = df['Attack_Mapped'].map({'normal': 0, 'attack': 1}).astype(int)

    # One-hot encode object cols except labels
    exclude = {'Class', 'Attack Type', 'Attack_Binary_Label', 'Attack_Mapped'}
    obj_cols = [c for c in df.select_dtypes(include=['object', 'category']).columns if c not in exclude]
    if obj_cols:
        df = pd.get_dummies(df, columns=obj_cols, drop_first=True)
    # Numeric cleanup
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
        df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
    drop_cols = [c for c in ['Class', 'Attack Type', 'Attack_Binary_Label', 'Attack_Mapped'] if c in df.columns]
    X = df.drop(columns=drop_cols, errors='ignore')
    y = df['Attack_Binary_Label'].astype(int)
    return X, y


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="../SDN-Net.csv", help="Path to SDN-Net CSV")
    parser.add_argument("--outdir", default="outputs", help="Output dir (only used if --save)")
    parser.add_argument("--k", type=int, default=30, help="Number of features for FS")
    parser.add_argument("--save", action="store_true", help="If set, save artifacts to outdir (default: False)")
    parser.add_argument("--pause", action="store_true", help="Pause between models (press Enter). Default: False")
    parser.add_argument("--algo", default="Perceptron", help="Algorithm name label to include in results (default: Perceptron)")
    parser.add_argument("--no-gpu", action="store_true", help="If set, disable GPU usage (use CPU only)")
    parser.add_argument("--timesteps", type=int, default=1, help="Timesteps (ignored for perceptron; kept for compatibility)")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs (default: 30)")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size (default: 128)")
    parser.add_argument("--patience", type=int, default=5, help="EarlyStopping patience (default: 5)")
    parser.add_argument("--hidden_layers", type=int, default=1, help="Number of hidden Dense layers (0 => single-layer perceptron)")
    parser.add_argument("--hidden_units", type=int, default=64, help="Units per hidden Dense layer (default: 64)")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate between hidden layers (default: 0.3)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        print("Ignored unknown args (likely from Jupyter):", unknown)

    # GPU control
    if args.no_gpu:
        try:
            tf.config.set_visible_devices([], 'GPU')
            print("GPU disabled; using CPU.")
        except Exception as e:
            print("Could not disable GPU (continuing):", e)
    else:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            print("GPUs available:", gpus)
        else:
            print("No GPUs found; running on CPU.")

    X_full, y_full = load_and_prepare(args.csv)
    print("Dataset shape:", X_full.shape)

    # Train/test split (stratified)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_full, y_full, test_size=0.33, random_state=RND, stratify=y_full
    )
    print("Train / Test shapes:", X_train_raw.shape, X_test_raw.shape)

    results = []

    # Model 1: Unbalanced full
    results.append(
        train_perceptron_display(
            "Model1_Unbalanced_Full",
            X_train_raw,
            X_test_raw,
            y_train,
            y_test,
            X_full.columns,
            algo_name=args.algo,
            hidden_layers=args.hidden_layers,
            hidden_units=args.hidden_units,
            dropout=args.dropout,
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            validation_split=0.15,
            save_outputs=args.save,
            outdir=args.outdir,
            pause_between=args.pause,
            optimizer_lr=args.lr,
            timesteps=args.timesteps
        )
    )

    # Model 2: Unbalanced + RF-FS
    sel2 = rf_feature_selection(X_train_raw, y_train, args.k)
    results.append(
        train_perceptron_display(
            "Model2_Unbalanced_RF-FS",
            X_train_raw[sel2],
            X_test_raw[sel2],
            y_train,
            y_test,
            sel2,
            algo_name=args.algo,
            hidden_layers=args.hidden_layers,
            hidden_units=args.hidden_units,
            dropout=args.dropout,
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            validation_split=0.15,
            save_outputs=args.save,
            outdir=args.outdir,
            pause_between=args.pause,
            show_fs_info=True,
            top_k_fs=args.k,
            optimizer_lr=args.lr,
            timesteps=args.timesteps
        )
    )

    # Model 3: Unbalanced + IG-FS
    sel3, ig_scores_full = ig_feature_selection_with_scores(X_train_raw, y_train, k=args.k)
    results.append(
        train_perceptron_display(
            "Model3_Unbalanced_IG-FS",
            X_train_raw[sel3],
            X_test_raw[sel3],
            y_train,
            y_test,
            sel3,
            algo_name=args.algo,
            hidden_layers=args.hidden_layers,
            hidden_units=args.hidden_units,
            dropout=args.dropout,
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            validation_split=0.15,
            save_outputs=args.save,
            outdir=args.outdir,
            pause_between=args.pause,
            show_fs_info=True,
            top_k_fs=args.k,
            optimizer_lr=args.lr,
            timesteps=args.timesteps
        )
    )

    # Model 4: SMOTE balanced (full)
    smote = SMOTE(random_state=RND)
    X_train_m4_arr, y_train_m4 = smote.fit_resample(X_train_raw, y_train)
    X_train_m4 = pd.DataFrame(X_train_m4_arr, columns=X_train_raw.columns)
    results.append(
        train_perceptron_display(
            "Model4_SMOTE_Full",
            X_train_m4,
            X_test_raw,
            y_train_m4,
            y_test,
            X_full.columns,
            algo_name=args.algo,
            hidden_layers=args.hidden_layers,
            hidden_units=args.hidden_units,
            dropout=args.dropout,
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            validation_split=0.15,
            save_outputs=args.save,
            outdir=args.outdir,
            pause_between=args.pause,
            optimizer_lr=args.lr,
            timesteps=args.timesteps
        )
    )

    # Model 5: SMOTE + RF-FS
    sel5 = rf_feature_selection(X_train_m4, y_train_m4, args.k)
    results.append(
        train_perceptron_display(
            "Model5_SMOTE_RF-FS",
            X_train_m4[sel5],
            X_test_raw[sel5],
            y_train_m4,
            y_test,
            sel5,
            algo_name=args.algo,
            hidden_layers=args.hidden_layers,
            hidden_units=args.hidden_units,
            dropout=args.dropout,
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            validation_split=0.15,
            save_outputs=args.save,
            outdir=args.outdir,
            pause_between=args.pause,
            show_fs_info=True,
            top_k_fs=args.k,
            optimizer_lr=args.lr,
            timesteps=args.timesteps
        )
    )

    # Model 6: SMOTE + IG-FS
    sel6, ig_scores_m4 = ig_feature_selection_with_scores(X_train_m4, y_train_m4, k=args.k)
    results.append(
        train_perceptron_display(
            "Model6_SMOTE_IG-FS",
            X_train_m4[sel6],
            X_test_raw[sel6],
            y_train_m4,
            y_test,
            sel6,
            algo_name=args.algo,
            hidden_layers=args.hidden_layers,
            hidden_units=args.hidden_units,
            dropout=args.dropout,
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            validation_split=0.15,
            save_outputs=args.save,
            outdir=args.outdir,
            pause_between=args.pause,
            show_fs_info=True,
            top_k_fs=args.k,
            optimizer_lr=args.lr,
            timesteps=args.timesteps
        )
    )

    # Model 7: SMOTE-Tomek balanced (full)
    smt = SMOTETomek(random_state=RND)
    X_train_m7_arr, y_train_m7 = smt.fit_resample(X_train_raw, y_train)
    X_train_m7 = pd.DataFrame(X_train_m7_arr, columns=X_train_raw.columns)
    results.append(
        train_perceptron_display(
            "Model7_SMOTETomek_Full",
            X_train_m7,
            X_test_raw,
            y_train_m7,
            y_test,
            X_full.columns,
            algo_name=args.algo,
            hidden_layers=args.hidden_layers,
            hidden_units=args.hidden_units,
            dropout=args.dropout,
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            validation_split=0.15,
            save_outputs=args.save,
            outdir=args.outdir,
            pause_between=args.pause,
            optimizer_lr=args.lr,
            timesteps=args.timesteps
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
        print("Saved summary to:", summary_path)


if __name__ == "__main__":
    main()
