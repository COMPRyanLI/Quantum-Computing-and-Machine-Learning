#!/usr/bin/env python3
"""

Runs Logistic Regression, RBF-SVM, and an MLP on:
  - two_moons
  - iris reduced with PCA
  - MNIST 0-vs-1 reduced with PCA


Examples to run this program:
  python classical_classifiers.py --dataset two_moons --model all
  python classical_classifiers.py --dataset iris --model mlp --mlp-target-params 12 --pca-components 3
  python classical_classifiers.py --dataset mnist01 --n-samples 800 --pca-components 4 --seeds 42 123 7
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml, load_iris, make_moons
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


@dataclass
class RunResult:
    dataset: str
    model: str
    seed: int
    train_accuracy: float
    val_accuracy: float
    test_accuracy: float
    fit_seconds: float
    n_train: int
    n_val: int
    n_test: int
    n_features: int
    n_classes: int
    details: dict[str, Any] = field(default_factory=dict)


def load_project_dataset(
    name: str,
    *,
    seed: int,
    n_samples: int,
    pca_components: int,
    moons_noise: float,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    if name == "two_moons":
        x, y = make_moons(n_samples=n_samples, noise=moons_noise, random_state=seed)
        return x.astype(np.float64), y.astype(int)

    if name == "iris":
        iris = load_iris()
        x = iris.data.astype(np.float64)
        y = iris.target.astype(int)
        if pca_components and pca_components < x.shape[1]:
            x = PCA(n_components=pca_components, random_state=seed).fit_transform(x)
        return x, y

    if name == "mnist01":
        mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
        x = mnist.data.astype(np.float32)
        y = mnist.target.astype(str)
        mask = np.isin(y, ["0", "1"])
        x = x[mask] / 255.0
        y = y[mask].astype(int)
        if n_samples and n_samples < len(y):
            idx = rng.choice(len(y), size=n_samples, replace=False)
            x, y = x[idx], y[idx]
        x = PCA(n_components=pca_components, random_state=seed).fit_transform(x)
        return x.astype(np.float64), y.astype(int)

    raise ValueError(f"Unknown dataset: {name}")


def split_70_15_15(
    x: np.ndarray, y: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    stratify = y if np.min(np.bincount(y)) >= 2 else None
    x_train, x_tmp, y_train, y_tmp = train_test_split(
        x, y, test_size=0.30, random_state=seed, stratify=stratify
    )
    stratify_tmp = y_tmp if np.min(np.bincount(y_tmp)) >= 2 else None
    x_val, x_test, y_val, y_test = train_test_split(
        x_tmp, y_tmp, test_size=0.50, random_state=seed, stratify=stratify_tmp
    )
    return x_train, x_val, x_test, y_train, y_val, y_test


def count_mlp_params(n_features: int, n_classes: int, hidden: tuple[int, ...]) -> int:
    dims = [n_features, *hidden, n_classes]
    return int(sum((dims[i] + 1) * dims[i + 1] for i in range(len(dims) - 1)))


def mlp_hidden_layers(
    n_features: int, n_classes: int, target_params: int | None, layers: int,
) -> tuple[int, ...]:
    """Scan widths and pick the one whose actual parameter count is closest
    to target_params (BUG FIX vs. the closed-form formula)."""
    if not target_params:
        return (16,) if layers == 1 else (16, 8)

    best_widths: tuple[int, ...] = (1,) * layers
    best_gap = float("inf")
    for width in range(1, 1024):
        widths = (width,) if layers == 1 else (width, width)
        params = count_mlp_params(n_features, n_classes, widths)
        gap = abs(params - target_params)
        if gap < best_gap:
            best_widths, best_gap = widths, gap
        if params > target_params * 4:
            break
    return best_widths


def build_model(
    model_name: str,
    *,
    seed: int,
    n_features: int,
    n_classes: int,
    mlp_target_params: int | None,
    mlp_layers: int,
) -> tuple[Pipeline, dict]:
    if model_name == "logistic":
        clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)]), {"regularization": "l2"}

    if model_name == "svm_rbf":
        clf = SVC(kernel="rbf", C=1.0, gamma="scale")
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)]), {"kernel": "rbf"}

    if model_name == "mlp":
        hidden = mlp_hidden_layers(n_features, n_classes, mlp_target_params, mlp_layers)
        clf = MLPClassifier(
            hidden_layer_sizes=hidden,
            activation="relu",
            alpha=1e-4,
            batch_size="auto",
            learning_rate_init=1e-3,
            max_iter=2000,
            random_state=seed,
            early_stopping=False,
            n_iter_no_change=30,
        )
        details = {
            "hidden_layer_sizes": list(hidden),
            "trainable_params": count_mlp_params(n_features, n_classes, hidden),
            "target_params": mlp_target_params,
        }
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)]), details

    raise ValueError(f"Unknown model: {model_name}")


def evaluate_one(
    model_name: str,
    x_train: np.ndarray, x_val: np.ndarray, x_test: np.ndarray,
    y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray,
    *,
    dataset: str,
    seed: int,
    mlp_target_params: int | None,
    mlp_layers: int,
) -> tuple[RunResult, str, np.ndarray]:
    n_features = x_train.shape[1]
    n_classes = len(np.unique(np.concatenate([y_train, y_val, y_test])))
    model, details = build_model(
        model_name, seed=seed, n_features=n_features, n_classes=n_classes,
        mlp_target_params=mlp_target_params, mlp_layers=mlp_layers,
    )

    start = time.perf_counter()
    model.fit(x_train, y_train)
    fit_seconds = time.perf_counter() - start

    y_pred = model.predict(x_test)
    result = RunResult(
        dataset=dataset, model=model_name, seed=seed,
        train_accuracy=accuracy_score(y_train, model.predict(x_train)),
        val_accuracy=accuracy_score(y_val, model.predict(x_val)),
        test_accuracy=accuracy_score(y_test, y_pred),
        fit_seconds=fit_seconds,
        n_train=len(y_train), n_val=len(y_val), n_test=len(y_test),
        n_features=n_features, n_classes=n_classes, details=details,
    )
    report = classification_report(y_test, y_pred, digits=4, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    return result, report, cm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classical ML baselines for proposal datasets.")
    parser.add_argument("--dataset", choices=["two_moons", "iris", "mnist01"], default="two_moons")
    parser.add_argument("--model", choices=["all", "logistic", "svm_rbf", "mlp"], default="all")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42],
                        help="Run with each seed and report mean ± std.")
    parser.add_argument("--n-samples", type=int, default=300, help="Used by two_moons and MNIST.")
    parser.add_argument("--pca-components", type=int, default=2, help="Used by iris and MNIST.")
    parser.add_argument("--moons-noise", type=float, default=0.25)
    parser.add_argument("--mlp-target-params", type=int, default=None,
                        help="Pick hidden width(s) so MLP parameter count matches the VQC's ansatz.")
    parser.add_argument("--mlp-layers", type=int, choices=[1, 2], default=1)
    parser.add_argument("--output-csv", type=Path, default=None)
    return parser.parse_args()


def summarize_multiseed(rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    if len(df) <= 1:
        return
    print("\n" + "=" * 80)
    print("Multi-seed summary (mean ± std across seeds)")
    print("=" * 80)
    for model_name in sorted(df["model"].unique()):
        sub = df[df["model"] == model_name]
        seeds = sorted(sub["seed"].tolist())
        print(f"\n[{model_name}] seeds={seeds}")
        for col in ["train_accuracy", "val_accuracy", "test_accuracy", "fit_seconds"]:
            mean = sub[col].mean()
            std = sub[col].std(ddof=1) if len(sub) > 1 else 0.0
            print(f"  {col:<18} = {mean:.4f} ± {std:.4f}")


def main() -> None:
    args = parse_args()

    models = ["logistic", "svm_rbf", "mlp"] if args.model == "all" else [args.model]
    all_rows: list[dict] = []

    for seed in args.seeds:
        x, y = load_project_dataset(
            args.dataset, seed=seed, n_samples=args.n_samples,
            pca_components=args.pca_components, moons_noise=args.moons_noise,
        )
        x_train, x_val, x_test, y_train, y_val, y_test = split_70_15_15(x, y, seed)

        for model_name in models:
            result, report, cm = evaluate_one(
                model_name, x_train, x_val, x_test, y_train, y_val, y_test,
                dataset=args.dataset, seed=seed,
                mlp_target_params=args.mlp_target_params, mlp_layers=args.mlp_layers,
            )
            row = asdict(result)
            all_rows.append(row)
            print(f"[seed={seed}] {model_name:<10} "
                  f"train={result.train_accuracy:.4f}  "
                  f"val={result.val_accuracy:.4f}  "
                  f"test={result.test_accuracy:.4f}  "
                  f"fit={result.fit_seconds:.3f}s")
            if len(args.seeds) == 1:
                print(json.dumps(row, indent=2, default=str))
                print("Confusion matrix:")
                print(cm)
                print("Classification report:")
                print(report)

    summarize_multiseed(all_rows)

    if args.output_csv:
        df = pd.json_normalize(all_rows)
        df.to_csv(args.output_csv, index=False)
        print(f"\nSaved results to {args.output_csv}")


if __name__ == "__main__":
    main()
