#!/usr/bin/env python3
"""
perceptron_multiclass.py

Perceptron / MLP multiclass 7-variant benchmarking script for intrusion detection on the SDN-Net dataset.

This script replaces the supervised autoencoder with a multiclass perceptron/MLP classifier
and preserves the 7 experimental pipeline variants:
  - Unbalanced
  - Unbalanced + RF Feature Selection (RF-FS)
  - Unbalanced + Information Gain / Mutual Information FS (IG-FS)
  - SMOTE
  - SMOTE + RF-FS
  - SMOTE + IG-FS
  - SMOTE-Tomek

Notes:
 - Classifier head uses softmax with n_classes outputs and sparse_categorical_crossentropy.
 - Class weights are computed for multiclass training.
 - ROC AUC is computed as macro-average (one-vs-rest) when possible.
 - Saves class mapping (index -> label) when --save is used.

Usage example:
  python perceptron_multiclass.py --csv ../SDN-Net.csv --k 30 --epochs 30 --batch_size 128 --hidden_layers 1 --hidden_units 64 --save

Dependencies:
  pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn joblib tensorflow
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

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
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

# TensorFlow import
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, InputLayer
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


def rf_feature_importances(X_train, y_train):
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


def build_perceptron_multiclass(input_dim, n_classes,
                                hidden_layers=1, hidden_units=64,
                                dropout=0.3, lr=1e-3):
    """
    Build a multiclass perceptron/MLP:
      - hidden_layers = 0 -> single-layer softmax (logistic multinomial)
      - otherwise stack hidden_layers Dense(ReLU) + BatchNorm + Dropout, then Dense(n_classes, softmax)
    """
    model = Sequential()
    model.add(InputLayer(input_shape=(input_dim,)))
    if hidden_layers <= 0:
        model.add(Dense(n_classes, activation='softmax'))
    else:
        for i in range(hidden_layers):
            model.add(Dense(hidden_units, activation='relu'))
            model.add(BatchNormalization())
            if dropout and dropout > 0:
                model.add(Dropout(dropout))
        model.add(Dense(n_classes, activation='softmax'))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def plot_history_learning_curve(history, title="Training History"):
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
    plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epochs, hs.get('accuracy', []), 'r-', label='Training acc')
    if 'val_accuracy' in hs:
        plt.plot(epochs, hs.get('val_accuracy', []), 'g--', label='Validation acc')
    plt.title(f"{title} - Accuracy")
    plt.xlabel('Epochs'); plt.ylabel('Accuracy'); plt.legend()
    plt.tight_layout(); plt.show()


def prepare_flat(X_df):
    if not isinstance(X_df, pd.DataFrame):
        X_df = pd.DataFrame(X_df)
    return X_df.values.astype('float32')


def train_perceptron_display_multiclass(
    name,
    X_train_df,
    X_test_df,
    y_train,
    y_test,
    features,
    class_names,
    algo_name="Perceptron-Multiclass",
    hidden_layers=1,
    hidden_units=64,
    dropout=0.3,
    epochs=50,
    batch_size=128,
    patience=6,
    validation_split=0.15,
    save_outputs=False,
    outdir="outputs",
    pause_between=False,
    show_fs_info=False,
    top_k_fs=20,
    use_class_weight=True,
    optimizer_lr=1e-3
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
            plot_feature_selection_comparison(rf_imp, ig_scores, top_k=top_k_fs, title_suffix=f"({name})")
        else:
            if not rf_imp.empty:
                plt.figure(figsize=(8, 6))
                rf_imp.head(top_k_fs).sort_values().plot.barh(color='tab:blue')
                plt.title(f"RF Feature Importances ({name})"); plt.tight_layout(); plt.show()
            if not ig_scores.empty:
                plt.figure(figsize=(8, 6))
                ig_scores.head(top_k_fs).sort_values().plot.barh(color='tab:green')
                plt.title(f"IG Scores ({name})"); plt.tight_layout(); plt.show()

    # Scale
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_df)
    X_test_sc = scaler.transform(X_test_df)

    X_train_p = prepare_flat(pd.DataFrame(X_train_sc, columns=X_train_df.columns))
    X_test_p = prepare_flat(pd.DataFrame(X_test_sc, columns=X_test_df.columns))

    n_classes = len(class_names)
    input_dim = X_train_p.shape[1]
    model = build_perceptron_multiclass(input_dim=input_dim, n_classes=n_classes,
                                        hidden_layers=hidden_layers, hidden_units=hidden_units,
                                        dropout=dropout, lr=optimizer_lr)

    callbacks = []
    early = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    callbacks.append(early)
    if save_outputs:
        ckpt_path = str(model_dir / f"best_model_{stamp}.h5")
        callbacks.append(ModelCheckpoint(ckpt_path, monitor='val_loss', save_best_only=True, save_weights_only=False))

    class_weight = None
    if use_class_weight:
        try:
            classes = np.unique(y_train)
            cw = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
            class_weight = {int(c): float(w) for c, w in zip(classes, cw)}
        except Exception:
            class_weight = None

    history = model.fit(
        X_train_p, y_train,
        validation_split=validation_split,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=2
    )

    y_prob = model.predict(X_test_p)  # shape (n_samples, n_classes)
    y_pred = np.argmax(y_prob, axis=1)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1v = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    try:
        y_test_bin = label_binarize(y_test, classes=list(range(n_classes)))
        roc_auc = float(roc_auc_score(y_test_bin, y_prob, average='macro', multi_class='ovr'))
    except Exception:
        roc_auc = None
    report = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n=== {name} ({algo_name}) ===")
    print(f"Classes: {class_names}")
    print(f"Accuracy: {acc:.4f}  Precision (weighted): {prec:.4f}  Recall (weighted): {rec:.4f}  F1 (weighted): {f1v:.4f}  AUC (macro OVR): {roc_auc if roc_auc is not None else 'N/A'}")
    print("\nClassification report:\n", report)

    # Plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                cbar=False, xticklabels=class_names, yticklabels=class_names)
    axes[0].set_title(f"Confusion Matrix - {name} ({algo_name})")
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("True"); axes[0].tick_params(axis='x', rotation=45)

    axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.6)
    if roc_auc is not None:
        try:
            fpr = dict(); tpr = dict()
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_test == i, y_prob[:, i])
            for i in range(n_classes):
                axes[1].plot(fpr[i], tpr[i], lw=1.5, label=f"{class_names[i]} (AUC {auc(fpr[i], tpr[i]):.2f})")
            axes[1].legend(loc='lower right', fontsize='small')
        except Exception:
            axes[1].text(0.5, 0.5, 'ROC N/A', ha='center')
    else:
        axes[1].text(0.5, 0.5, 'ROC N/A', ha='center')
    axes[1].set_title(f"ROC Curves - {name} ({algo_name})"); axes[1].set_xlabel("False Positive Rate"); axes[1].set_ylabel("True Positive Rate")
    plt.tight_layout(); plt.show()

    try:
        plot_history_learning_curve(history, title=f"Learning Curve ({name} - {algo_name})")
    except Exception as e:
        print("Unable to plot training history:", e)

    if save_outputs:
        try:
            model_save_dir = model_dir / f"perceptron_multiclass_model_{stamp}"
            model.save(str(model_save_dir), include_optimizer=True)
        except Exception as e:
            try:
                model.save(str(model_dir / f"perceptron_multiclass_model_{stamp}.h5"))
            except Exception as e2:
                print("Unable to save model:", e, e2)

        try:
            joblib.dump(scaler, model_dir / f"scaler_{stamp}.joblib")
        except Exception as e:
            print("Unable to save scaler:", e)

        if features is not None:
            pd.Series(list(features), name="feature").to_csv(model_dir / f"features_{stamp}.csv", index=False)

        classes_path = model_dir / f"classes_{stamp}.json"
        try:
            with open(classes_path, "w", encoding="utf8") as f:
                json.dump({"classes": class_names}, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print("Unable to save class mapping:", e)

        metrics = {
            "algorithm": algo_name,
            "hidden_layers": int(hidden_layers),
            "hidden_units": int(hidden_units),
            "epochs_ran": int(len(history.history.get('loss', []))),
            "batch_size": int(batch_size),
            "n_classes": int(n_classes),
            "accuracy": float(acc),
            "precision_weighted": float(prec),
            "recall_weighted": float(rec),
            "f1_weighted": float(f1v),
            "auc_macro_ovr": float(roc_auc) if roc_auc is not None else None
        }
        with open(model_dir / f"metrics_{stamp}.json", "w", encoding="utf8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        with open(model_dir / f"classification_report_{stamp}.txt", "w", encoding="utf8") as f:
            f.write(report)

        try:
            hist_obj = history.history
            with open(model_dir / f"history_{stamp}.json", "w", encoding="utf8") as f:
                json.dump(hist_obj, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

        try:
            fig.savefig(model_dir / f"confusion_roc_{stamp}.png", dpi=150); plt.close(fig)
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
        "n_classes": int(n_classes),
        "accuracy": acc,
        "precision_weighted": prec,
        "recall_weighted": rec,
        "f1_weighted": f1v,
        "auc_macro_ovr": roc_auc
    }


def load_and_prepare_multiclass(csv_path):
    """
    Loads CSV and returns (X, y, class_names).
    """
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
    else:
        df['Attack Type'] = df['Attack Type'].astype(str).str.strip()
    df['Attack Type'] = df['Attack Type'].fillna("UNKNOWN")

    # Use categorical codes to preserve mapping
    cat = pd.Categorical(df['Attack Type'])
    class_names = list(cat.categories)
    df['Attack_Label_Code'] = cat.codes
    y = df['Attack_Label_Code'].astype(int)

    exclude = {'Attack Type', 'Attack_Label_Code'}
    obj_cols = [c for c in df.select_dtypes(include=['object', 'category']).columns if c not in exclude]
    if obj_cols:
        df = pd.get_dummies(df, columns=obj_cols, drop_first=True)

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    drop_cols = ['Attack Type', 'Attack_Label_Code']
    feature_cols = [c for c in num_cols if c not in drop_cols]
    if not feature_cols:
        raise ValueError("No numeric or encoded features found after preprocessing.")
    X = df[feature_cols]
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
    parser.add_argument("--algo", default="Perceptron-Multiclass", help="Algorithm name label to include in results")
    parser.add_argument("--no-gpu", action="store_true", help="If set, disable GPU usage (use CPU only)")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs (default: 30)")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size (default: 128)")
    parser.add_argument("--patience", type=int, default=6, help="EarlyStopping patience (default: 6)")
    parser.add_argument("--hidden_layers", type=int, default=1, help="Number of hidden Dense layers (0 => multinomial logistic)")
    parser.add_argument("--hidden_units", type=int, default=64, help="Units per hidden Dense layer (default: 64)")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate (default: 0.3)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        print("Ignored unknown args (likely from Jupyter):", unknown)

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

    X_full, y_full, class_names = load_and_prepare_multiclass(args.csv)
    print("Dataset shape (features):", X_full.shape)
    print("Number of classes:", len(class_names))
    print("Classes:", class_names)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_full, y_full, test_size=0.33, random_state=RND, stratify=y_full
    )
    print("Train / Test shapes:", X_train_raw.shape, X_test_raw.shape)

    results = []

    results.append(
        train_perceptron_display_multiclass(
            "Model1_Unbalanced_Full",
            X_train_raw, X_test_raw, y_train, y_test,
            X_full.columns, class_names,
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
            optimizer_lr=args.lr
        )
    )

    sel2 = rf_feature_selection(X_train_raw, y_train, args.k)
    results.append(
        train_perceptron_display_multiclass(
            "Model2_Unbalanced_RF-FS",
            X_train_raw[sel2], X_test_raw[sel2], y_train, y_test,
            sel2, class_names,
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
            optimizer_lr=args.lr
        )
    )

    sel3, ig_scores_full = ig_feature_selection_with_scores(X_train_raw, y_train, k=args.k)
    results.append(
        train_perceptron_display_multiclass(
            "Model3_Unbalanced_IG-FS",
            X_train_raw[sel3], X_test_raw[sel3], y_train, y_test,
            sel3, class_names,
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
            optimizer_lr=args.lr
        )
    )

    smote = SMOTE(random_state=RND)
    X_train_m4_arr, y_train_m4 = smote.fit_resample(X_train_raw, y_train)
    X_train_m4 = pd.DataFrame(X_train_m4_arr, columns=X_train_raw.columns)
    results.append(
        train_perceptron_display_multiclass(
            "Model4_SMOTE_Full",
            X_train_m4, X_test_raw, y_train_m4, y_test,
            X_full.columns, class_names,
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
            optimizer_lr=args.lr
        )
    )

    sel5 = rf_feature_selection(X_train_m4, y_train_m4, args.k)
    results.append(
        train_perceptron_display_multiclass(
            "Model5_SMOTE_RF-FS",
            X_train_m4[sel5], X_test_raw[sel5], y_train_m4, y_test,
            sel5, class_names,
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
            optimizer_lr=args.lr
        )
    )

    sel6, ig_scores_m4 = ig_feature_selection_with_scores(X_train_m4, y_train_m4, k=args.k)
    results.append(
        train_perceptron_display_multiclass(
            "Model6_SMOTE_IG-FS",
            X_train_m4[sel6], X_test_raw[sel6], y_train_m4, y_test,
            sel6, class_names,
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
            optimizer_lr=args.lr
        )
    )

    smt = SMOTETomek(random_state=RND)
    X_train_m7_arr, y_train_m7 = smt.fit_resample(X_train_raw, y_train)
    X_train_m7 = pd.DataFrame(X_train_m7_arr, columns=X_train_raw.columns)
    results.append(
        train_perceptron_display_multiclass(
            "Model7_SMOTETomek_Full",
            X_train_m7, X_test_raw, y_train_m7, y_test,
            X_full.columns, class_names,
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
            optimizer_lr=args.lr
        )
    )

    results_df = pd.DataFrame(results)
    print("\n=== Summary of all 7 models ===")
    print(results_df)

    if args.save:
        ensure_dir(args.outdir)
        summary_path = Path(args.outdir) / f"summary_7models_{args.algo}_{timestamp()}.csv"
        results_df.to_csv(summary_path, index=False)
        classes_root = Path(args.outdir) / f"classes_{timestamp()}.json"
        with open(classes_root, "w", encoding="utf8") as f:
            json.dump({"classes": class_names}, f, indent=2, ensure_ascii=False)
        print("Saved summary to:", summary_path)
        print("Saved class mapping to:", classes_root)


if __name__ == "__main__":
    main()
