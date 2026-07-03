#!/usr/bin/env python3
""".

Implements:
  - Variational Quantum Classifier (VQC), with COBYLA or SPSA optimizer
  - Quantum kernel SVM (QSVC) with angle, ZZ, or data-reuploading feature maps
  - Optional noisy simulation: synthetic depolarizing+readout OR FakeBackend
  - Parameter-shift gradient-variance probe (barren-plateau, proposal RQ3)
  - Multi-seed runner that reports mean ± std for headline numbers

Datasets:
  - two_moons
  - iris reduced with PCA  (use --pca-components 3 for 3-class iris with 3 qubits)
  - MNIST 0-vs-1 reduced with PCA

Examples:
  # Single-seed VQC, statevector
  python quantum_classifiers.py --dataset two_moons --classifier vqc

  # Multi-seed VQC, reports mean ± std (recommended for final paper)
  python quantum_classifiers.py --dataset iris --classifier vqc \
      --pca-components 3 --seeds 42 123 7 --output-csv results.csv

  # QSVC with a non-ZZ feature map (RQ1 exploration)
  python quantum_classifiers.py --dataset two_moons --classifier qsvc \
      --feature-map angle --seeds 42 123 7

  # Noisy run with synthetic depolarizing + readout
  python quantum_classifiers.py --dataset two_moons --classifier vqc \
      --shots 1024 --noise depolarizing --depol-p1 0.005 --depol-p2 0.05

  # Noisy run with a calibrated FakeBackend snapshot
  python quantum_classifiers.py --dataset two_moons --classifier vqc \
      --shots 1024 --noise fake_backend --fake-backend FakeBrisbane

  # Barren-plateau sweep
  python quantum_classifiers.py --barren-plateau --bp-qubits 2 3 4 5 6 \
      --bp-depths 1 2 4 8 --bp-samples 200 --output-csv bp.csv

Quantum simulation can be slow. Start with small --n-samples and --maxiter.
"""

from __future__ import annotations

import argparse
import inspect
import json
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml, load_iris, make_moons
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# ---- Qiskit core ------------------------------------------------------------
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

# Library circuits: prefer the new lowercase functions (Qiskit >= 2.1).
try:
    from qiskit.circuit.library import ( 
        efficient_su2 as _efficient_su2_fn,
        real_amplitudes as _real_amplitudes_fn,
        zz_feature_map as _zz_feature_map_fn,
    )

    def _make_zz(num_qubits, reps, entanglement):
        return _zz_feature_map_fn(num_qubits, reps=reps, entanglement=entanglement)

    def _make_real_amplitudes(num_qubits, reps, entanglement):
        return _real_amplitudes_fn(num_qubits, reps=reps, entanglement=entanglement)

    def _make_efficient_su2(num_qubits, reps, entanglement):
        return _efficient_su2_fn(
            num_qubits, su2_gates=["ry", "rz"], reps=reps, entanglement=entanglement
        )

except ImportError:  # pragma: no cover — Qiskit < 2.1
    from qiskit.circuit.library import EfficientSU2, RealAmplitudes, ZZFeatureMap

    def _make_zz(num_qubits, reps, entanglement):
        return ZZFeatureMap(feature_dimension=num_qubits, reps=reps, entanglement=entanglement)

    def _make_real_amplitudes(num_qubits, reps, entanglement):
        return RealAmplitudes(num_qubits=num_qubits, reps=reps, entanglement=entanglement)

    def _make_efficient_su2(num_qubits, reps, entanglement):
        return EfficientSU2(
            num_qubits=num_qubits, reps=reps, entanglement=entanglement, su2_gates=["ry", "rz"]
        )

# ---- Primitives -------------------------------------------------------------
try:
    from qiskit.primitives import StatevectorSampler
except Exception:  # pragma: no cover
    StatevectorSampler = None  # type: ignore[assignment]

try:
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error
    from qiskit_aer.primitives import SamplerV2 as AerSamplerV2
except Exception:  
    AerSimulator = None  
    AerSamplerV2 = None 
    NoiseModel = None  
    ReadoutError = None  
    depolarizing_error = None  

# FakeBackend snapshots (calibrated noise from real IBM devices)
_FAKE_BACKENDS: dict[str, type] = {}
try:
    from qiskit_ibm_runtime.fake_provider import (
        FakeBrisbane,
        FakeManilaV2,
        FakeSherbrooke,
    )

    _FAKE_BACKENDS = {
        "FakeBrisbane": FakeBrisbane,
        "FakeSherbrooke": FakeSherbrooke,
        "FakeManilaV2": FakeManilaV2,
    }
except Exception:  # pragma: no cover
    pass

# ---- Optimizers / ML helpers ------------------------------------------------
try:
    from qiskit_algorithms.optimizers import COBYLA, SPSA
except Exception as exc:  
    raise ImportError("Install qiskit-algorithms: pip install qiskit-algorithms") from exc

try:
    from qiskit_machine_learning.algorithms import QSVC, VQC
except Exception:  
    from qiskit_machine_learning.algorithms.classifiers import QSVC, VQC  # type: ignore

try:
    from qiskit_machine_learning.kernels import FidelityQuantumKernel
except Exception as exc:  
    raise ImportError(
        "Install qiskit-machine-learning: pip install qiskit-machine-learning"
    ) from exc

# ---- Transpiler -------------------------------------------------------------
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager


# Result container

@dataclass
class RunResult:
    dataset: str
    classifier: str
    feature_map: str
    ansatz: str | None
    depth: int
    entanglement: str
    optimizer: str | None
    seed: int
    train_accuracy: float
    val_accuracy: float
    test_accuracy: float
    fit_seconds: float
    n_train: int
    n_val: int
    n_test: int
    n_features: int
    n_qubits: int
    n_classes: int
    trainable_params: int | None
    details: dict[str, Any] = field(default_factory=dict)


# Data loading
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


def scale_for_quantum(
    x_train: np.ndarray, x_val: np.ndarray, x_test: np.ndarray, *, upper: float = np.pi
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scaler = MinMaxScaler(feature_range=(0.0, upper))
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)
    return x_train, x_val, x_test



# Circuit builders
def angle_feature_map(num_qubits: int) -> QuantumCircuit:
    x = ParameterVector("x", num_qubits)
    qc = QuantumCircuit(num_qubits, name="AngleEncoding")
    for i in range(num_qubits):
        qc.ry(x[i], i)
    return qc


def add_entanglers(qc: QuantumCircuit, entanglement: str) -> None:
    n = qc.num_qubits
    if n < 2 or entanglement == "none":
        return
    if entanglement in {"linear", "circular"}:
        for i in range(n - 1):
            qc.cx(i, i + 1)
        if entanglement == "circular" and n > 2:
            qc.cx(n - 1, 0)
    elif entanglement == "full":
        for i in range(n):
            for j in range(i + 1, n):
                qc.cx(i, j)
    else:
        raise ValueError(f"Unknown entanglement: {entanglement}")


def reupload_feature_map(num_qubits: int, reps: int, entanglement: str) -> QuantumCircuit:
    """Repeated angle encoding with entanglers (proposal's data re-uploading)."""
    x = ParameterVector("x", num_qubits)
    qc = QuantumCircuit(num_qubits, name="DataReuploading")
    for _ in range(reps):
        for i in range(num_qubits):
            qc.ry(x[i], i)
            qc.rz(x[i], i)
        add_entanglers(qc, entanglement)
    return qc


def qiskit_library_entanglement(entanglement: str) -> str:
    return "linear" if entanglement == "none" else entanglement


def build_feature_map(name: str, num_qubits: int, reps: int, entanglement: str) -> QuantumCircuit:
    if name == "angle":
        return angle_feature_map(num_qubits)
    if name == "zz":
        return _make_zz(num_qubits, reps, qiskit_library_entanglement(entanglement))
    if name == "reupload":
        return reupload_feature_map(num_qubits, reps=reps, entanglement=entanglement)
    raise ValueError(f"Unknown feature map: {name}")


def build_ansatz(name: str, num_qubits: int, depth: int, entanglement: str) -> QuantumCircuit:
    ent = qiskit_library_entanglement(entanglement)
    if name == "real_amplitudes":
        return _make_real_amplitudes(num_qubits, reps=depth, entanglement=ent)
    if name == "efficient_su2":
        return _make_efficient_su2(num_qubits, reps=depth, entanglement=ent)
    raise ValueError(f"Unknown ansatz: {name}")



# Initialization
def grant_identity_block_init(num_params: int, seed: int) -> np.ndarray:
    """Near-identity initialization (Grant et al. 2019, practical surrogate)."""
    rng = np.random.default_rng(seed)
    return rng.normal(loc=0.0, scale=0.01, size=num_params)


# Multi-class interpret function 
def make_interpret(num_qubits: int, num_classes: int):

    if num_qubits < int(np.ceil(np.log2(max(num_classes, 2)))):
        warnings.warn(
            f"num_qubits={num_qubits} is too small for {num_classes} classes; "
            f"need at least ceil(log2({num_classes})). "
            "Increase --pca-components.",
            stacklevel=2,
        )

    def interpret(outcome: int) -> int:
        if outcome < num_classes:
            return int(outcome)
        return int(outcome % num_classes)

    return interpret



# Noise model construction

def build_synthetic_noise_model(
    *, depol_p1: float, depol_p2: float, readout_error: float
) -> "NoiseModel | None":
    if NoiseModel is None or depolarizing_error is None or ReadoutError is None:
        return None
    if depol_p1 <= 0 and depol_p2 <= 0 and readout_error <= 0:
        return None

    nm = NoiseModel()
    if depol_p1 > 0:
        err1 = depolarizing_error(depol_p1, 1)
        nm.add_all_qubit_quantum_error(err1, ["u1", "u2", "u3", "rx", "ry", "rz", "x", "h", "sx"])
    if depol_p2 > 0:
        err2 = depolarizing_error(depol_p2, 2)
        nm.add_all_qubit_quantum_error(err2, ["cx", "cz"])
    if readout_error > 0:
        p = float(readout_error)
        ro = ReadoutError([[1 - p, p], [p, 1 - p]])
        nm.add_all_qubit_readout_error(ro)
    return nm


def build_fake_backend_noise_model(fake_backend_name: str) -> "NoiseModel | None":
    """Build a calibrated noise model from an IBM FakeBackend snapshot."""
    if NoiseModel is None:
        return None
    if not _FAKE_BACKENDS:
        raise ImportError(
            "FakeBackend support requires qiskit-ibm-runtime. "
            "Install with: pip install qiskit-ibm-runtime"
        )
    if fake_backend_name not in _FAKE_BACKENDS:
        raise ValueError(
            f"Unknown fake backend '{fake_backend_name}'. "
            f"Available: {list(_FAKE_BACKENDS)}"
        )
    backend = _FAKE_BACKENDS[fake_backend_name]()
    return NoiseModel.from_backend(backend)



# Sampler / pass-manager wiring

def build_sampler_and_passmanager(
    *, shots: int | None, seed: int, noise_model: "NoiseModel | None"
):
    """AerSamplerV2 requires transpiled circuits; pass manager is bound to the
    same AerSimulator that carries the noise model so noise actually flows."""
    if shots and AerSamplerV2 is not None:
        backend_options: dict[str, Any] = {}
        if noise_model is not None:
            backend_options["noise_model"] = noise_model
        sampler = AerSamplerV2(
            default_shots=shots,
            seed=seed,
            options={"backend_options": backend_options} if backend_options else None,
        )
        sim_backend = AerSimulator(noise_model=noise_model) if noise_model else AerSimulator()
        pm = generate_preset_pass_manager(optimization_level=1, backend=sim_backend)
        return sampler, pm

    if StatevectorSampler is not None:
        sampler = StatevectorSampler(default_shots=shots or 1024)
        return sampler, None
    return None, None



# Fitting

def fit_vqc(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    feature_map: QuantumCircuit,
    ansatz: QuantumCircuit,
    optimizer_name: str,
    maxiter: int,
    shots: int | None,
    seed: int,
    num_classes: int,
    noise_model: "NoiseModel | None" = None,
) -> VQC:
    optimizer = COBYLA(maxiter=maxiter) if optimizer_name == "cobyla" else SPSA(maxiter=maxiter)
    sampler, pass_manager = build_sampler_and_passmanager(
        shots=shots, seed=seed, noise_model=noise_model
    )

    initial_point = grant_identity_block_init(ansatz.num_parameters, seed)
    interpret = make_interpret(ansatz.num_qubits, num_classes)

    kwargs: dict[str, Any] = {
        "feature_map": feature_map,
        "ansatz": ansatz,
        "optimizer": optimizer,
        "initial_point": initial_point,
        "interpret": interpret,
        "output_shape": num_classes,
    }
    if sampler is not None:
        kwargs["sampler"] = sampler
    if pass_manager is not None:
        kwargs["pass_manager"] = pass_manager

    sig_params = set(inspect.signature(VQC).parameters)
    kwargs = {k: v for k, v in kwargs.items() if k in sig_params}

    model = VQC(**kwargs)
    model.fit(x_train, y_train)
    return model


def fit_qsvc(x_train: np.ndarray, y_train: np.ndarray, *, feature_map: QuantumCircuit) -> QSVC:
    kernel = FidelityQuantumKernel(feature_map=feature_map)
    model = QSVC(quantum_kernel=kernel)
    model.fit(x_train, y_train)
    return model


# Single-seed train+eval

def run_single_seed(args: argparse.Namespace, seed: int) -> tuple[RunResult, np.ndarray, str]:
    x, y = load_project_dataset(
        args.dataset,
        seed=seed,
        n_samples=args.n_samples,
        pca_components=args.pca_components,
        moons_noise=args.moons_noise,
    )
    x_train, x_val, x_test, y_train, y_val, y_test = split_70_15_15(x, y, seed)
    x_train, x_val, x_test = scale_for_quantum(
        x_train, x_val, x_test, upper=args.feature_scale_upper
    )

    num_qubits = x_train.shape[1]
    n_classes = len(np.unique(np.concatenate([y_train, y_val, y_test])))

    feature_map = build_feature_map(
        args.feature_map, num_qubits, args.feature_reps, args.entanglement
    )

    noise_model = None
    shots = args.shots
    if args.noise == "depolarizing":
        if not shots:
            shots = 1024
        noise_model = build_synthetic_noise_model(
            depol_p1=args.depol_p1, depol_p2=args.depol_p2, readout_error=args.readout_error,
        )
    elif args.noise == "fake_backend":
        if not shots:
            shots = 1024
        noise_model = build_fake_backend_noise_model(args.fake_backend)

    start = time.perf_counter()
    ansatz: QuantumCircuit | None = None
    trainable_params: int | None = None
    if args.classifier == "vqc":
        ansatz = build_ansatz(args.ansatz, num_qubits, args.depth, args.entanglement)
        trainable_params = ansatz.num_parameters
        model = fit_vqc(
            x_train, y_train,
            feature_map=feature_map, ansatz=ansatz,
            optimizer_name=args.optimizer, maxiter=args.maxiter,
            shots=shots, seed=seed, num_classes=n_classes,
            noise_model=noise_model,
        )
    else:
        if shots or noise_model is not None:
            print("  Note: noisy/shot-based QSVC not implemented; using statevector QSVC.")
        model = fit_qsvc(x_train, y_train, feature_map=feature_map)
    fit_seconds = time.perf_counter() - start

    y_pred = model.predict(x_test)

    details: dict[str, Any] = {
        "feature_map_circuit_depth": feature_map.decompose().depth(),
        "ansatz_circuit_depth": ansatz.decompose().depth() if ansatz is not None else None,
        "feature_map_parameters": feature_map.num_parameters,
        "shots": shots,
        "noise": args.noise,
        "feature_scale_upper": args.feature_scale_upper,
        "maxiter": args.maxiter,
        "fake_backend": args.fake_backend if args.noise == "fake_backend" else None,
    }
    if args.noise == "depolarizing":
        details["noise_params"] = {
            "depol_p1": args.depol_p1,
            "depol_p2": args.depol_p2,
            "readout_error": args.readout_error,
        }

    result = RunResult(
        dataset=args.dataset,
        classifier=args.classifier,
        feature_map=args.feature_map,
        ansatz=args.ansatz if args.classifier == "vqc" else None,
        depth=args.depth,
        entanglement=args.entanglement,
        optimizer=args.optimizer if args.classifier == "vqc" else None,
        seed=seed,
        train_accuracy=accuracy_score(y_train, model.predict(x_train)),
        val_accuracy=accuracy_score(y_val, model.predict(x_val)),
        test_accuracy=accuracy_score(y_test, y_pred),
        fit_seconds=fit_seconds,
        n_train=len(y_train),
        n_val=len(y_val),
        n_test=len(y_test),
        n_features=x_train.shape[1],
        n_qubits=num_qubits,
        n_classes=n_classes,
        trainable_params=trainable_params,
        details=details,
    )
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4, zero_division=0)
    return result, cm, report



# Barren-plateau probe (RQ3)

def gradient_variance_probe(
    *,
    n_qubits: int,
    depth: int,
    n_samples: int,
    seed: int,
    ansatz_name: str = "real_amplitudes",
    entanglement: str = "linear",
) -> dict[str, float]:
    from qiskit.quantum_info import SparsePauliOp, Statevector

    ansatz = build_ansatz(ansatz_name, n_qubits, depth, entanglement)
    n_params = ansatz.num_parameters
    if n_params == 0:
        return {"n_qubits": n_qubits, "depth": depth, "var_grad": float("nan"), "n_params": 0}

    obs = SparsePauliOp.from_list([("Z" + "I" * (n_qubits - 1), 1.0)])

    rng = np.random.default_rng(seed + 1000 * n_qubits + depth)
    grads = np.empty(n_samples, dtype=float)
    shift = np.pi / 2.0
    for s in range(n_samples):
        theta = rng.uniform(-np.pi, np.pi, size=n_params)
        tp = theta.copy(); tp[0] += shift
        tm = theta.copy(); tm[0] -= shift
        cp = ansatz.assign_parameters(tp)
        cm_ = ansatz.assign_parameters(tm)
        ep = Statevector.from_instruction(cp).expectation_value(obs).real
        em = Statevector.from_instruction(cm_).expectation_value(obs).real
        grads[s] = 0.5 * (ep - em)

    return {
        "n_qubits": n_qubits, "depth": depth,
        "ansatz": ansatz_name, "entanglement": entanglement,
        "n_params": int(n_params),
        "var_grad": float(np.var(grads)),
        "mean_grad": float(np.mean(grads)),
        "n_samples": int(n_samples),
    }


def run_barren_plateau_sweep(
    qubits_list: list[int], depths_list: list[int],
    n_samples: int, seed: int, ansatz_name: str, entanglement: str,
    output_csv: Path | None,
) -> pd.DataFrame:
    rows = []
    for nq in qubits_list:
        for d in depths_list:
            print(f"Barren plateau probe: qubits={nq}, depth={d} ...", flush=True)
            row = gradient_variance_probe(
                n_qubits=nq, depth=d, n_samples=n_samples, seed=seed,
                ansatz_name=ansatz_name, entanglement=entanglement,
            )
            print(f"  Var[grad] = {row['var_grad']:.3e}  (n_params={row['n_params']})")
            rows.append(row)
    df = pd.DataFrame(rows)
    print("\nBarren-plateau sweep:")
    print(df.to_string(index=False))
    if output_csv is not None:
        df.to_csv(output_csv, index=False)
        print(f"\nSaved barren-plateau results to {output_csv}")
    return df



# CLI
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantum classifiers for proposal datasets.")
    parser.add_argument("--barren-plateau", action="store_true",
                        help="Run the barren-plateau gradient-variance probe.")
    parser.add_argument("--bp-qubits", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    parser.add_argument("--bp-depths", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--bp-samples", type=int, default=100)

    parser.add_argument("--dataset", choices=["two_moons", "iris", "mnist01"], default="two_moons")
    parser.add_argument("--n-samples", type=int, default=300, help="Used by two_moons and MNIST.")
    parser.add_argument("--pca-components", type=int, default=2,
                        help="For Iris use 3 to avoid the parity-collision ceiling.")
    parser.add_argument("--moons-noise", type=float, default=0.25)
    parser.add_argument("--feature-scale-upper", type=float, default=np.pi)

    parser.add_argument("--classifier", choices=["vqc", "qsvc"], default="vqc")
    parser.add_argument("--feature-map", choices=["angle", "zz", "reupload"], default="angle")
    parser.add_argument("--feature-reps", type=int, default=1)
    parser.add_argument("--ansatz", choices=["real_amplitudes", "efficient_su2"],
                        default="real_amplitudes")
    parser.add_argument("--depth", type=int, choices=[1, 2, 3, 4, 8], default=1)
    parser.add_argument("--entanglement", choices=["linear", "circular", "full", "none"],
                        default="linear")
    parser.add_argument("--optimizer", choices=["cobyla", "spsa"], default="cobyla")
    parser.add_argument("--maxiter", type=int, default=100)
    parser.add_argument("--shots", type=int, default=None)

    parser.add_argument("--noise", choices=["none", "depolarizing", "fake_backend"], default="none")
    parser.add_argument("--depol-p1", type=float, default=0.001)
    parser.add_argument("--depol-p2", type=float, default=0.01)
    parser.add_argument("--readout-error", type=float, default=0.02)
    parser.add_argument("--fake-backend",
                        choices=list(_FAKE_BACKENDS) if _FAKE_BACKENDS else ["FakeBrisbane"],
                        default="FakeBrisbane")

    # Multi-seed support
    parser.add_argument("--seeds", type=int, nargs="+", default=[42],
                        help="Run with each seed and report mean ± std.")

    parser.add_argument("--output-csv", type=Path, default=None)
    return parser.parse_args()


def summarize_multiseed(rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    if len(df) <= 1:
        return
    print("\n" + "=" * 80)
    print(f"Multi-seed summary across {len(df)} seeds: {sorted(df['seed'].tolist())}")
    print("=" * 80)
    for col in ["train_accuracy", "val_accuracy", "test_accuracy", "fit_seconds"]:
        mean = df[col].mean()
        std = df[col].std(ddof=1) if len(df) > 1 else 0.0
        print(f"  {col:<18} = {mean:.4f} ± {std:.4f}")


def main() -> None:
    args = parse_args()

    if args.barren_plateau:
        seed = args.seeds[0]
        run_barren_plateau_sweep(
            args.bp_qubits, args.bp_depths, args.bp_samples, seed,
            args.ansatz, args.entanglement, args.output_csv,
        )
        return

    all_rows: list[dict] = []
    for seed in args.seeds:
        print(f"\n[seed={seed}] {args.classifier.upper()} on {args.dataset} "
              f"({args.feature_map}{'/' + args.ansatz if args.classifier == 'vqc' else ''})")
        result, cm, report = run_single_seed(args, seed)
        row = asdict(result)
        all_rows.append(row)
        print(f"  train={result.train_accuracy:.4f}  "
              f"val={result.val_accuracy:.4f}  "
              f"test={result.test_accuracy:.4f}  "
              f"fit={result.fit_seconds:.1f}s")
        if len(args.seeds) == 1:
            print(json.dumps(row, indent=2, default=str))
            print("Confusion matrix:")
            print(cm)
            print("Classification report:")
            print(report)

    summarize_multiseed(all_rows)

    if args.output_csv:
        pd.json_normalize(all_rows).to_csv(args.output_csv, index=False)
        print(f"\nSaved results to {args.output_csv}")


if __name__ == "__main__":
    main()
