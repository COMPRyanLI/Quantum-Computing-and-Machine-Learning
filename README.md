# Quantum Classifiers vs. Classical Classifiers


## Install

Requires Python 3.12.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install numpy pandas scikit-learn matplotlib
pip install qiskit==2.4.1 qiskit-aer==0.17.2 qiskit-algorithms==0.4.0 qiskit-machine-learning==0.9.0
pip install qiskit-ibm-runtime==0.46.1
mkdir results
```

On macOS, use `python3 -m venv .venv` and `source .venv/bin/activate`.

## Run

All commands below use PowerShell syntax (backtick `` ` `` for line continuation). On macOS/Linux, replace `` ` `` with `\`.

### RQ1

```powershell
python quantum_classifiers.py --dataset two_moons --classifier vqc `
    --feature-map angle --ansatz real_amplitudes --depth 2 `
    --n-samples 300 --maxiter 100 --seeds 42 123 7 `
    --output-csv results\rq1_two_moons_vqc.csv

python quantum_classifiers.py --dataset iris --classifier vqc `
    --pca-components 3 --feature-map angle --ansatz real_amplitudes --depth 2 `
    --maxiter 150 --seeds 42 123 7 `
    --output-csv results\rq1_iris_vqc.csv

python quantum_classifiers.py --dataset mnist01 --classifier vqc `
    --feature-map zz --ansatz efficient_su2 --depth 2 `
    --n-samples 300 --pca-components 4 --maxiter 100 --seeds 42 123 7 `
    --output-csv results\rq1_mnist_vqc.csv

python quantum_classifiers.py --dataset two_moons --classifier qsvc `
    --feature-map zz --n-samples 300 --seeds 42 123 7 `
    --output-csv results\rq1_two_moons_qsvc_zz.csv

python quantum_classifiers.py --dataset iris --classifier qsvc `
    --pca-components 3 --feature-map zz --seeds 42 123 7 `
    --output-csv results\rq1_iris_qsvc_zz.csv

python quantum_classifiers.py --dataset mnist01 --classifier qsvc `
    --feature-map zz --n-samples 300 --pca-components 4 --seeds 42 123 7 `
    --output-csv results\rq1_mnist_qsvc_zz.csv

python quantum_classifiers.py --dataset two_moons --classifier qsvc `
    --feature-map angle --n-samples 300 --seeds 42 123 7 `
    --output-csv results\rq1_two_moons_qsvc_angle.csv

python quantum_classifiers.py --dataset two_moons --classifier qsvc `
    --feature-map reupload --feature-reps 2 --n-samples 300 --seeds 42 123 7 `
    --output-csv results\rq1_two_moons_qsvc_reupload.csv

python classical_classifiers.py --dataset two_moons --model all `
    --n-samples 300 --mlp-target-params 6 --seeds 42 123 7 `
    --output-csv results\rq1_two_moons_classical.csv

python classical_classifiers.py --dataset iris --model all `
    --pca-components 3 --mlp-target-params 9 --seeds 42 123 7 `
    --output-csv results\rq1_iris_classical.csv

python classical_classifiers.py --dataset mnist01 --model all `
    --n-samples 300 --pca-components 4 --mlp-target-params 24 --seeds 42 123 7 `
    --output-csv results\rq1_mnist_classical.csv
```

### RQ2

```powershell
foreach ($fm in @("angle","zz","reupload")) {
  foreach ($depth in 1,2,3,4) {
    foreach ($ent in @("linear","circular","full")) {
      $tag = "two_moons_${fm}_d${depth}_${ent}"
      python quantum_classifiers.py --dataset two_moons --classifier vqc `
        --feature-map $fm --ansatz real_amplitudes --depth $depth `
        --entanglement $ent --n-samples 300 --maxiter 100 --seeds 42 `
        --output-csv "results\rq2_${tag}.csv"
    }
  }
}

python quantum_classifiers.py --dataset two_moons --classifier vqc `
    --feature-map angle --ansatz real_amplitudes --depth 2 --entanglement full `
    --n-samples 300 --maxiter 100 --seeds 42 123 7 `
    --output-csv results\rq2_best_angle.csv

python quantum_classifiers.py --dataset two_moons --classifier vqc `
    --feature-map reupload --ansatz real_amplitudes --depth 2 --entanglement linear `
    --n-samples 300 --maxiter 100 --seeds 42 123 7 `
    --output-csv results\rq2_best_reupload.csv

python quantum_classifiers.py --dataset two_moons --classifier vqc `
    --feature-map zz --ansatz real_amplitudes --depth 2 --entanglement linear `
    --n-samples 300 --maxiter 100 --seeds 42 123 7 `
    --output-csv results\rq2_best_zz.csv
```

### RQ3

```powershell
python quantum_classifiers.py --barren-plateau `
    --ansatz real_amplitudes --bp-qubits 2 3 4 5 6 --bp-depths 1 2 4 8 `
    --bp-samples 300 --output-csv results\rq3_real_amplitudes.csv

python quantum_classifiers.py --barren-plateau `
    --ansatz efficient_su2 --bp-qubits 2 3 4 5 6 --bp-depths 1 2 4 8 `
    --bp-samples 300 --output-csv results\rq3_efficient_su2.csv
```

### RQ4

```powershell
foreach ($depth in 2,4,8) {
  python quantum_classifiers.py --dataset two_moons --classifier vqc `
    --feature-map angle --ansatz real_amplitudes --depth $depth `
    --entanglement linear --n-samples 400 --maxiter 100 `
    --shots 1024 --seeds 42 123 7 `
    --output-csv "results\rq4_d${depth}_noiseless.csv"

  foreach ($p1 in 0.001, 0.005, 0.01, 0.02) {
    $p2 = $p1 * 10
    $ro = $p1 * 5
    $tag = "d${depth}_p1_$(($p1).ToString().Replace('.','p'))"
    python quantum_classifiers.py --dataset two_moons --classifier vqc `
      --feature-map angle --ansatz real_amplitudes --depth $depth `
      --entanglement linear --n-samples 400 --maxiter 100 --shots 1024 `
      --noise depolarizing --depol-p1 $p1 --depol-p2 $p2 --readout-error $ro `
      --seeds 42 123 7 --output-csv "results\rq4_${tag}.csv"
  }
}

python quantum_classifiers.py --dataset two_moons --classifier vqc `
    --feature-map angle --ansatz real_amplitudes --depth 4 --entanglement linear `
    --n-samples 400 --maxiter 100 --shots 1024 `
    --noise fake_backend --fake-backend FakeBrisbane `
    --seeds 42 123 7 --output-csv results\rq4_fake_brisbane.csv

foreach ($opt in @("cobyla","spsa")) {
  python quantum_classifiers.py --dataset two_moons --classifier vqc `
    --feature-map angle --ansatz real_amplitudes --depth 4 --entanglement linear `
    --n-samples 400 --maxiter 100 --shots 1024 `
    --noise depolarizing --depol-p1 0.005 --depol-p2 0.05 --readout-error 0.025 `
    --optimizer $opt --seeds 42 123 7 `
    --output-csv "results\rq4_opt_${opt}.csv"
}
```

### Aggregate all results

```powershell
python -c "import pandas as pd, glob; pd.concat([pd.read_csv(f).assign(file=f) for f in glob.glob('results/*.csv')], ignore_index=True).to_csv('results/all_results.csv', index=False); print('Wrote results/all_results.csv')"
```
