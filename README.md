# SCC-VFL: Selective Counterfactual Consistency for Vertical FL

This repository contains the official implementation of **SCC-VFL**, a Vertical Federated Learning framework designed to ensure individual-level counterfactual fairness on tabular datasets. SCC-VFL introduces a selective consistency mechanism that aligns model predictions with counterfactuals generated in a privacy-preserving manner.

## 📂 Repository Structure

The codebase is organized by dataset, with separate scripts for Independent and Identically Distributed (IID) and Non-IID split configurations.

```text
SCC-VFL/
├── compass-recidivism-experiments/
│   ├── compass-iid.py
│   └── compass-non-iid.py
├── german-credit-experiments/
│   ├── german-credit-iid.py
│   ├── german-credit-non-iid.py
│   ├── ablation.py           # Component analysis on German Credit
│   └── attack.py             # Privacy/Robustness attacks on German Credit
├── heart-disease-experiments/
│   ├── heart-disease-iid.py
│   └── heart-disease-non-iid.py
├── requirements.txt
└── README.md
```

## 🚀 Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/dawoodwasif/SCC-VFL
   cd SCC-VFL
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   # Linux/MacOS
   python -m venv .venv
   source .venv/bin/activate

   # Windows
   .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 📊 Datasets

The experiments use three standard fairness benchmarks. The scripts automatically handle downloading and preprocessing (via `ucimlrepo` or direct URLs).

| Dataset | Sensitive Attribute | Task |
| :--- | :--- | :--- |
| **German Credit** | Age (>= 25) | Credit Risk Classification |
| **COMPAS** | Race  | Recidivism Prediction |
| **Heart Disease** | Sex | Disease Presence Prediction |

## 🏃 Running Experiments

Each script runs a complete evaluation pipeline for 30 seed runs:
1. Loads and preprocesses data.
2. Performs graph-free mediator discovery.
3. Trains multiple baselines (Baseline, Uniform CF, Policy-Blind, Server-Only).
4. Trains the **SCC-VFL** model.
5. Repeats experiments across **30 random seeds**.
6. Reports aggregated metrics (Accuracy, LogLoss, SCG, FR).

### German Credit Experiments
```bash
# Run IID split evaluation
python german-credit-experiments/german-credit-iid.py

# Run Non-IID split evaluation
python german-credit-experiments/german-credit-non-iid.py
```

### Heart Disease Experiments
```bash
# Run IID split evaluation
python heart-disease-experiments/heart-disease-iid.py

# Run Non-IID split evaluation
python heart-disease-experiments/heart-disease-non-iid.py
```

### COMPAS Recidivism Experiments
```bash
# Run IID split evaluation
python compass-recidivism-experiments/compass-iid.py

# Run Non-IID split evaluation
python compass-recidivism-experiments/compass-non-iid.py
```

## 🔬 Ablation & Robustness Studies

We provide specialized scripts to analyze the contribution of SCC-VFL components and evaluate resilience against privacy attacks. These studies are configured for the **German Credit** dataset.

### Ablation Study
Evaluates model variants (e.g., removing the generator, removing consistency regularization) to demonstrate the necessity of each SCC-VFL component.
```bash
python german-credit-experiments/ablation.py
```

### Attack & Privacy Evaluation
Runs Attribute Inference Attacks (AIA) and Subspace PGD attacks to measure the privacy leakage and robustness of the learned representations.
```bash
python german-credit-experiments/attack.py
```

## 📈 Results

The scripts output tables with Mean ± Std Dev for:
- **Accuracy**: Utility metric (Higher is better).
- **LogLoss**: Calibration metric (Lower is better).
- **SCG (Selective Counterfactual Gap)**: Fairness metric (Lower is better).
- **FR (Flip Rate)**: Counterfactual fairness metric (Lower is better).

## 📝 License
This project is licensed under the MIT License.
