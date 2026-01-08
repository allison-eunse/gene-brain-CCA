#!/usr/bin/env python3
"""
Stage 2: Supervised Clinical Prediction from CCA Joint Embeddings

This script takes the canonical variates (joint embeddings) from Stage 1 CCA/SCCA
and uses them to predict clinical outcomes (e.g., MDD, PHQ-9) using:
  - Logistic Regression (for binary classification)
  - Multi-Layer Perceptron (MLP)

Supports:
  - Cross-validation with stratified K-fold
  - Evaluation metrics: AUC-ROC, accuracy, precision, recall, F1
  - Comparison of gene-only, fmri-only, and joint (concatenated) embeddings

Inputs:
  --u-gene: Gene canonical variates from Stage 1 (N × k)
  --v-fmri: fMRI canonical variates from Stage 1 (N × k)
  --labels: Clinical labels (N,), binary 0/1
  --ids: Subject IDs for verification (optional)

Outputs:
  - stage2_results.json: CV metrics for all models and feature sets
  - stage2_predictions.npy: Out-of-fold predictions (optional)
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Suppress sklearn convergence warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)


def load_sklearn():
    """Load sklearn components with error handling."""
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.neural_network import MLPClassifier
        from sklearn.model_selection import StratifiedKFold
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import (
            roc_auc_score,
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            confusion_matrix,
        )

        return {
            "LogisticRegression": LogisticRegression,
            "MLPClassifier": MLPClassifier,
            "StratifiedKFold": StratifiedKFold,
            "StandardScaler": StandardScaler,
            "roc_auc_score": roc_auc_score,
            "accuracy_score": accuracy_score,
            "precision_score": precision_score,
            "recall_score": recall_score,
            "f1_score": f1_score,
            "confusion_matrix": confusion_matrix,
        }
    except ImportError as e:
        raise SystemExit(
            "[ERROR] scikit-learn required. Install: pip install scikit-learn"
        ) from e


def create_feature_sets(
    U_gene: np.ndarray,
    V_fmri: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Create feature sets for comparison experiments."""
    return {
        "gene_only": U_gene,
        "fmri_only": V_fmri,
        "joint": np.hstack([U_gene, V_fmri]),
        # Additional: mean of gene and fmri (simple fusion)
        "joint_mean": (U_gene + V_fmri) / 2,
    }


def run_cv_experiment(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str,
    n_folds: int,
    seed: int,
    mlp_hidden: Tuple[int, ...] = (64, 32),
    mlp_max_iter: int = 500,
) -> Dict[str, Any]:
    """
    Run cross-validated classification experiment.

    Returns dict with metrics and out-of-fold predictions.
    """
    sk = load_sklearn()

    skf = sk["StratifiedKFold"](n_splits=n_folds, shuffle=True, random_state=seed)

    # Collect fold results
    aucs = []
    accs = []
    precs = []
    recs = []
    f1s = []

    oof_preds = np.zeros(len(y), dtype=np.float64)
    oof_binary = np.zeros(len(y), dtype=np.int64)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Standardize features within each fold
        scaler = sk["StandardScaler"]()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        # Create model
        if model_type == "logreg":
            model = sk["LogisticRegression"](
                max_iter=1000,
                solver="lbfgs",
                class_weight="balanced",
                random_state=seed,
            )
        elif model_type == "mlp":
            model = sk["MLPClassifier"](
                hidden_layer_sizes=mlp_hidden,
                max_iter=mlp_max_iter,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10,
                random_state=seed,
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        # Fit
        model.fit(X_train, y_train)

        # Predict probabilities
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_val)[:, 1]
        else:
            y_prob = model.decision_function(X_val)

        y_pred = model.predict(X_val)

        # Store out-of-fold predictions
        oof_preds[val_idx] = y_prob
        oof_binary[val_idx] = y_pred

        # Compute metrics
        try:
            auc = sk["roc_auc_score"](y_val, y_prob)
        except ValueError:
            # If only one class in val set
            auc = 0.5

        aucs.append(auc)
        accs.append(sk["accuracy_score"](y_val, y_pred))
        precs.append(sk["precision_score"](y_val, y_pred, zero_division=0))
        recs.append(sk["recall_score"](y_val, y_pred, zero_division=0))
        f1s.append(sk["f1_score"](y_val, y_pred, zero_division=0))

    # Aggregate metrics
    results = {
        "model": model_type,
        "n_folds": n_folds,
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
        "auc_folds": [float(a) for a in aucs],
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "precision_mean": float(np.mean(precs)),
        "recall_mean": float(np.mean(recs)),
        "f1_mean": float(np.mean(f1s)),
    }

    # Overall confusion matrix from OOF predictions
    cm = sk["confusion_matrix"](y, oof_binary)
    results["confusion_matrix"] = cm.tolist()

    return results, oof_preds


def main():
    ap = argparse.ArgumentParser(
        description="Stage 2: Clinical prediction from CCA joint embeddings"
    )

    # Input data
    ap.add_argument(
        "--u-gene",
        required=True,
        help="Gene canonical variates (N × k), from Stage 1",
    )
    ap.add_argument(
        "--v-fmri",
        required=True,
        help="fMRI canonical variates (N × k), from Stage 1",
    )
    ap.add_argument(
        "--labels",
        required=True,
        help="Clinical labels (N,), binary 0/1 (e.g., MDD case/control)",
    )
    ap.add_argument(
        "--ids",
        default=None,
        help="Subject IDs for verification (optional)",
    )

    # Experiment settings
    ap.add_argument(
        "--models",
        nargs="+",
        choices=["logreg", "mlp"],
        default=["logreg", "mlp"],
        help="Models to evaluate",
    )
    ap.add_argument(
        "--feature-sets",
        nargs="+",
        choices=["gene_only", "fmri_only", "joint", "joint_mean"],
        default=["gene_only", "fmri_only", "joint"],
        help="Feature sets to evaluate",
    )
    ap.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of CV folds (default: 5)",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    # MLP settings
    ap.add_argument(
        "--mlp-hidden",
        type=str,
        default="64,32",
        help="MLP hidden layer sizes (comma-separated, default: 64,32)",
    )
    ap.add_argument(
        "--mlp-max-iter",
        type=int,
        default=500,
        help="MLP max iterations (default: 500)",
    )

    # Output
    ap.add_argument("--out-dir", required=True, help="Output directory")
    ap.add_argument(
        "--prefix",
        default="stage2_",
        help="Output file prefix",
    )

    args = ap.parse_args()

    # Setup output
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Parse MLP hidden layers
    mlp_hidden = tuple(int(x) for x in args.mlp_hidden.split(","))

    # Load data
    print(f"[load] Gene variates: {args.u_gene}", flush=True)
    print(f"[load] fMRI variates: {args.v_fmri}", flush=True)
    print(f"[load] Labels: {args.labels}", flush=True)

    U_gene = np.load(args.u_gene).astype(np.float64)
    V_fmri = np.load(args.v_fmri).astype(np.float64)
    labels = np.load(args.labels)

    # Ensure labels are binary
    y = labels.astype(np.int64).ravel()
    unique_labels = np.unique(y)
    if not np.array_equal(unique_labels, np.array([0, 1])):
        print(f"[warn] Labels are not binary 0/1: {unique_labels}", flush=True)
        if len(unique_labels) == 2:
            # Map to 0/1
            y = (y == unique_labels[1]).astype(np.int64)
            print(f"[info] Mapped labels: {unique_labels[0]}→0, {unique_labels[1]}→1", flush=True)
        else:
            raise SystemExit("[ERROR] Labels must be binary for classification")

    # Verify dimensions
    if U_gene.shape[0] != V_fmri.shape[0]:
        raise SystemExit(
            f"[ERROR] Gene/fMRI row mismatch: {U_gene.shape[0]} vs {V_fmri.shape[0]}"
        )
    if U_gene.shape[0] != len(y):
        raise SystemExit(
            f"[ERROR] Features/labels row mismatch: {U_gene.shape[0]} vs {len(y)}"
        )

    N = U_gene.shape[0]
    n_pos = int(y.sum())
    n_neg = N - n_pos

    print(f"[info] N={N}, n_components_gene={U_gene.shape[1]}, n_components_fmri={V_fmri.shape[1]}", flush=True)
    print(f"[info] Labels: {n_neg} controls, {n_pos} cases (prevalence={n_pos/N:.2%})", flush=True)

    # Create feature sets
    feature_sets = create_feature_sets(U_gene, V_fmri)

    # Run experiments
    all_results: List[Dict[str, Any]] = []
    all_oof_preds: Dict[str, Dict[str, np.ndarray]] = {}

    for fs_name in args.feature_sets:
        X = feature_sets[fs_name]
        print(f"\n[exp] Feature set: {fs_name} (dim={X.shape[1]})", flush=True)
        all_oof_preds[fs_name] = {}

        for model_type in args.models:
            print(f"  [model] {model_type}...", end=" ", flush=True)

            results, oof_preds = run_cv_experiment(
                X,
                y,
                model_type,
                args.n_folds,
                args.seed,
                mlp_hidden=mlp_hidden,
                mlp_max_iter=args.mlp_max_iter,
            )

            results["feature_set"] = fs_name
            results["feature_dim"] = int(X.shape[1])
            all_results.append(results)
            all_oof_preds[fs_name][model_type] = oof_preds

            print(f"AUC={results['auc_mean']:.4f}±{results['auc_std']:.4f}", flush=True)

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY: AUC-ROC (mean ± std)")
    print("=" * 70)
    print(f"{'Feature Set':<15} {'Model':<10} {'AUC':>12} {'Accuracy':>12} {'F1':>12}")
    print("-" * 70)
    for res in all_results:
        print(
            f"{res['feature_set']:<15} {res['model']:<10} "
            f"{res['auc_mean']:.4f}±{res['auc_std']:.3f} "
            f"{res['accuracy_mean']:.4f}±{res['accuracy_std']:.3f} "
            f"{res['f1_mean']:.4f}"
        )
    print("=" * 70)

    # Find best model
    best = max(all_results, key=lambda x: x["auc_mean"])
    print(f"\n[best] {best['feature_set']} + {best['model']}: AUC={best['auc_mean']:.4f}")

    # Interpretation guidance
    if "joint" in args.feature_sets:
        joint_results = [r for r in all_results if r["feature_set"] == "joint"]
        gene_results = [r for r in all_results if r["feature_set"] == "gene_only"]
        fmri_results = [r for r in all_results if r["feature_set"] == "fmri_only"]

        if joint_results and gene_results and fmri_results:
            joint_auc = max(r["auc_mean"] for r in joint_results)
            gene_auc = max(r["auc_mean"] for r in gene_results)
            fmri_auc = max(r["auc_mean"] for r in fmri_results)

            print("\n[interpretation]")
            if joint_auc > max(gene_auc, fmri_auc) + 0.02:
                print("  Joint embeddings outperform unimodal → Gene-brain coupling is predictive")
            elif gene_auc > fmri_auc + 0.02:
                print("  Gene embeddings dominate → Genetic variation drives clinical phenotype")
            elif fmri_auc > gene_auc + 0.02:
                print("  fMRI embeddings dominate → Brain features drive clinical phenotype")
            else:
                print("  Similar performance across modalities → Complementary information")

    # Save results
    summary = {
        "n_subjects": int(N),
        "n_cases": int(n_pos),
        "n_controls": int(n_neg),
        "n_folds": args.n_folds,
        "seed": args.seed,
        "models": args.models,
        "feature_sets": args.feature_sets,
        "results": all_results,
        "best_config": {
            "feature_set": best["feature_set"],
            "model": best["model"],
            "auc": best["auc_mean"],
            "auc_std": best["auc_std"],
        },
    }

    results_path = out_dir / f"{args.prefix}results.json"
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[save] {results_path}", flush=True)

    # Save OOF predictions
    for fs_name, model_preds in all_oof_preds.items():
        for model_type, preds in model_preds.items():
            pred_path = out_dir / f"{args.prefix}oof_{fs_name}_{model_type}.npy"
            np.save(pred_path, preds.astype(np.float32))

    # Copy labels and IDs for reference
    np.save(out_dir / f"{args.prefix}labels.npy", y)
    if args.ids:
        ids = np.load(args.ids, allow_pickle=True).astype(str)
        np.save(out_dir / f"{args.prefix}ids.npy", ids)

    print(f"[done] Stage 2 complete. Results in {out_dir}", flush=True)


if __name__ == "__main__":
    main()


