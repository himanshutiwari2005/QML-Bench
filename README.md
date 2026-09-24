# QML-Bench: Quantum-Classical Machine Learning Benchmark

A rigorous, scientifically honest comparison of quantum and classical machine learning methods on small and medium-scale classification problems.

## Research Objective

Under controlled experimental conditions, characterize how quantum-kernel and variational quantum classifiers compare with strong classical baselines in terms of:

- Performance (accuracy, F1, balanced accuracy)
- Computational cost (wall-clock time, circuit evaluations)
- Circuit complexity (qubits, depth, gate counts)
- Dimensionality effects
- Noise sensitivity
- Data encoding limitations
- Scaling behavior

**Important**: This project does NOT claim quantum advantage. It performs honest empirical comparison and reports what the data shows.

## Project Structure

```text
QML-Bench/
├── src/
│   ├── datasets/          # Dataset loaders (synthetic + real)
│   │   ├── synthetic.py   # Linearly separable, moons
│   │   ├── real.py        # Iris, Breast Cancer Wisconsin
│   │   └── data_info.py   # Dataset documentation utilities
│   ├── classical/         # Classical ML baselines
│   │   ├── logistic_regression.py
│   │   ├── svm.py
│   │   ├── random_forest.py
│   │   ├── mlp.py
│   │   ├── xgboost_model.py
│   │   └── benchmark.py   # Unified classical benchmark runner
│   ├── quantum/           # Quantum ML methods
│   │   ├── feature_maps/  # Feature map factory (ZZ, Pauli, Z, linear)
│   │   │   └── factory.py
│   │   ├── kernels/       # Quantum kernel method with caching
│   │   │   └── quantum_kernel.py
│   │   ├── circuits/      # Parameterized ansatz circuits
│   │   │   └── ansatz.py
│   │   ├── optimizers/    # VQC training loop with SPSA/COBYLA
│   │   │   └── vqc_optimizer.py
│   │   └── vqc.py         # VQC experiment runner
│   ├── experiments/       # Experiment orchestration
│   │   └── benchmark.py   # Main QMLBenchmark class
│   ├── evaluation/        # Metrics and statistical analysis
│   │   └── metrics.py     # Aggregate, t-test, Wilcoxon
│   ├── noise/             # Noise model construction
│   │   └── models.py      # Depolarizing, readout, combined noise
│   ├── visualization/     # Plot generation
│   │   └── plots.py       # All required figures (10+ plot types)
│   └── utils/             # Config, preprocessing, version tracking
│       ├── config.py      # Dataclass configs (Dataset/Classical/Quantum/Experiment)
│       ├── preprocessing.py  # Standardization + PCA + range scaling
│       └── version.py     # Environment version logging
├── tests/                 # Unit tests (preprocessing, config, metrics, circuits)
│   ├── test_preprocessing.py
│   ├── test_config.py
│   ├── test_metrics.py
│   └── test_circuit_construction.py
├── results/               # Generated experiment outputs (NOT tracked in git)
│   └── README.md          # Results directory notes
├── run_benchmark.py       # CLI entry point
├── TASK.md                # Development task plan (phases 0-13)
├── PROMPT.md              # Full project specification (source of truth)
├── LICENSE                # MIT
└── .gitignore             # Excludes results/, bytecode, venv, notebooks, configs/
```

### What's NOT in the repo

The following directories exist as placeholders or hold generated/user-specific content — none are tracked in git:

| `results/` | JSON/CSV experiment outputs, generated figures | No — regenerate with `run_benchmark.py` |

**Source code** lives exclusively under `src/` plus `run_benchmark.py` at the root. That's the entire tracked build.

## Installation

### Requirements

- Python 3.11+
- Qiskit 2.5+ with Qiskit Machine Learning 0.9+
- NumPy, SciPy, scikit-learn
- Matplotlib, pandas, seaborn
- XGBoost (optional)
- qiskit-algorithms

### Setup

```bash
cd QML-Bench
uv pip install qiskit qiskit-machine-learning qiskit-aer \
    matplotlib pandas seaborn scikit-learn xgboost plotly qiskit-algorithms
```

Or use the provided requirements:

```bash
pip install -r requirements.txt  # (if available)
```

## Quick Start

### Run a smoke test first

```bash
python run_benchmark.py --quick
```

This runs minimal configurations to verify everything works.

### Run the full benchmark (Iris dataset)

```bash
python run_benchmark.py --dataset iris --seeds 3
```

### Run on other datasets

```bash
python run_benchmark.py --dataset breast_cancer
python run_benchmark.py --dataset linearly_separable
python run_benchmark.py --dataset moons
```

### Run noise sensitivity study

```bash
python run_benchmark.py --noise-study
```

### Generate plots from saved results

```bash
python run_benchmark.py --plots
```

## How It Works

### Data Pipeline

Raw features → Standardization → Optional PCA → Range scaling to [0, π] → Quantum feature map

### Classical Baselines

Strong, simple baselines tuned using validation data (not test data):

- **Logistic Regression** with L2 regularization (C swept over [0.01, 100])
- **SVM** with linear, RBF, and polynomial kernels (C and gamma tuned)
- **Random Forest** with tuned n_estimators, max_depth, min_samples_split/leaf
- **MLP** (Multi-Layer Perceptron) with tuned hidden layers and L2 penalty
- **XGBoost** (optional)

### Quantum Methods

#### Quantum Kernel Method

1. Feature map encodes classical data into quantum state space
2. Fidelity quantum kernel computes K(x_i, x_j) = |⟨φ(x_i)|φ(x_j)⟩|²
3. Classical SVM uses the precomputed quantum kernel matrix
4. Kernel caching avoids expensive recomputation

#### Variational Quantum Classifier (VQC)

1. Feature map encodes input
2. Parameterized ansatz (hardware-efficient or RealAmplitudes)
3. EstimatorQNN computes expectation values
4. Optimizer (SPSA or COBYLA) minimizes classification loss
5. Predictions via thresholding or argmax of expectations

### Feature Maps

- **ZZ feature map**: Entangling ZZ interactions, widely studied
- **Pauli feature map**: General Pauli rotations with configurable entropy
- **Z feature map**: Simple single-qubit Z rotations
- **Linear feature map**: Basic angle encoding

### Noise Models

- **Depolarizing noise**: Gate error probability applied to all qubits/gates
- **Readout noise**: Measurement error probability
- **Combined**: Both gate and readout errors simultaneously

## Experiment Configuration

Experiments are driven by configuration objects (`src/utils/config.py`):

```python
from src.utils.config import DatasetConfig, ClassicalConfig, QuantumConfig, ExperimentConfig

config = ExperimentConfig(
    dataset=DatasetConfig(name='iris', n_samples=150, n_features=4, n_classes=3),
    classical=[
        ClassicalConfig(model_type='logreg', cv_folds=5),
        ClassicalConfig(model_type='svm', hyperparameters={'kernel': 'rbf'}, cv_folds=5),
        ClassicalConfig(model_type='rf', cv_folds=5),
    ],
    quantum=[
        QuantumConfig(n_qubits=2, feature_map_type='zz', feature_map_depth=1,
                     ansatz_type='hardware_efficient', ansatz_depth=1,
                     optimizer='COBYLA', optimizer_kwargs={'maxiter': 50}),
    ],
    n_seeds=1,
)
```

## Metrics

### Classification
- Accuracy
- Balanced accuracy
- Macro F1 score
- Precision (macro)
- Recall (macro)
- Confusion matrix

### Quantum-specific
- Number of qubits
- Circuit depth
- Gate counts
- Number of trainable parameters
- Number of circuit evaluations
- Number of shots
- Kernel evaluation cost

### System
- Wall-clock runtime
- Memory (where measurable)

## Results Format

Results are saved as JSON in `results/all_results.json`:

```json
{
  "experiment_id": "iris_Full_s42_QK_zz_q2_d1",
  "dataset_name": "iris_Full_s42",
  "model_type": "Quantum Kernel",
  "n_features": 4,
  "n_qubits": 2,
  "circuit_depth": 1,
  "accuracy": 0.933,
  "balanced_accuracy": 0.925,
  "macro_f1": 0.928,
  "runtime_seconds": 2.34,
  "n_trainable_params": 0,
  "n_circuit_evaluations": 0,
  "n_shots": 1024,
  "kernel_evaluation_cost": 1.89,
  "seed": 42,
  "extra": {
    "feature_map_type": "zz",
    "feature_map_info": {...}
  }
}
```

## Research Questions Answered

- **RQ1**: Do quantum methods outperform strong classical baselines? (Tested empirically)
- **RQ2**: Under what conditions do they approach classical performance?
- **RQ3**: How strongly does quantum performance degrade with noise?
- **RQ4**: What is the relationship between circuit depth and predictive performance?
- **RQ5**: How does PCA/dimensionality reduction affect the comparison?
- **RQ6**: What computational costs accompany the quantum approaches?
- **RQ7**: Are observed performance differences stable across random seeds?

## Statistical Rigor

- Multi-seed repetition (configurable, default 1)
- Mean, standard deviation, and sample count reported
- Paired t-tests and Wilcoxon signed-rank tests for paired comparisons
- Classical baselines tuned on validation data (nested cross-validation via GridSearchCV)

## Important Caveats

1. **Simulated execution**: This project uses Qiskit Aer simulator (statevector), NOT actual quantum hardware by default.
2. **Noisy simulation**: Separate noise study uses simulated noise models.
3. **Small scale**: Qubit counts (2, 4) and sample sizes (150, 200, 569) are small — suitable for simulation.
4. **Neutral language**: Results are reported factually without claiming quantum advantage.
5. **Classical baselines are strong**: They are tuned on validation data, not test data.

## What This Experiment Does NOT Demonstrate

See `docs/research_report.md` for the full discussion. In brief:

- It does NOT demonstrate quantum advantage or quantum supremacy
- It does NOT claim quantum computers are faster (simulator runtime ≠ hardware runtime)
- It does NOT prove quantum ML is superior to classical ML
- It does NOT make claims about barren plateaus unless data supports them
- Results from simulation do NOT necessarily generalize to real hardware

## Development Phases

See `TASK.md` for the full phased development plan:

0. Environment setup (DONE)
1. Data pipeline (DONE)
2. Classical baselines (DONE)
3. Quantum feature maps (DONE)
4. Quantum kernel method (DONE)
5. VQC (DONE)
6. Ideal simulation benchmark (READY)
7. Noise study (READY)
8. Scaling study (READY)
9. Repeated-seed experiments (READY)
10. Optional hardware (SKIP if unavailable)
11. Visualization (READY)
12. Final report (TODO)

## License

MIT License - see `LICENSE` file.

## References

- Qiskit documentation: https://docs.quantum.ibm.com/
- Qiskit Machine Learning: https://qiskit.org/ecosystem/machine-learning/
- The project follows the specification in `PROMPT.md`
