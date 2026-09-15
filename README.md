# Class-Incremental Continual Learning on Split-MNIST

This project compares two approaches to class-incremental continual learning on **Split-MNIST**:

- **FeCAM**: an exemplar-free classifier that stores each class's feature mean and covariance and predicts with Mahalanobis distance.
- **Experience Replay (ER)**: a CNN trained with a reservoir-sampled memory of previous images, evaluated with buffers of 500 and 100 samples.

MNIST classes are introduced sequentially as `(0,1)`, `(2,3)`, `(4,5)`, `(6,7)`, and `(8,9)`. After every experience, evaluation covers every class seen so far; task identity is not given at test time.

## Repository layout

```text
.
├── run_experiment.py       # Standalone, reproducible experiment entry point
├── notebooks/
│   └── cl-notebook.ipynb   # Original analysis notebook
├── figures/                # Saved figures from the supplied experiment run
├── results/                # Saved accuracy reports from the supplied experiment run
├── presentation.pptx       # Project presentation
├── requirements.txt
└── .gitignore
```

`data/` and `models/` are created automatically when the experiment is run and are intentionally ignored by Git. Dataset downloads are sizeable, and the generated Fashion-MNIST checkpoint can always be recreated.

## Requirements

- Python 3.10 or newer
- `pip`
- Optional: an NVIDIA GPU with a CUDA-compatible PyTorch build. The project automatically uses CUDA when PyTorch can see it; otherwise it runs on CPU.

## Run from a fresh clone

```powershell
git clone <YOUR-REPOSITORY-URL>
cd continual-learning-fecam
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python run_experiment.py
```

On macOS or Linux, activate the environment with:

```bash
source .venv/bin/activate
```

The first run downloads Fashion-MNIST and MNIST. It then pre-trains the frozen FeCAM encoder for 40 epochs, evaluates FeCAM and its covariance ablation, and trains each ER configuration for 5 epochs per task. This can take a while on CPU. Later runs reuse `models/fmnist_pretrained.pth`.

## Faster smoke test

Use fewer epochs and skip the ablation to check that the full pipeline works:

```powershell
python run_experiment.py --pretrain-epochs 1 --replay-epochs 1 --skip-ablation
```

To write generated datasets, checkpoints, figures, and reports somewhere other than the repository root:

```powershell
python run_experiment.py --output-dir .\experiment-output
```

## Notebook

To use the original notebook interactively:

```powershell
jupyter notebook notebooks\cl-notebook.ipynb
```

Run its cells from top to bottom. It will create `data/`, `models/`, `figures/`, and `results/` relative to the directory from which Jupyter starts.

## Outputs

Running `run_experiment.py` produces:

- `models/fmnist_pretrained.pth` — pre-trained feature extractor (generated locally)
- `figures/fecam_confusion_matrices.png` — FeCAM confusion matrices
- `figures/experience_replay_500_confusion_matrices.png` and `figures/experience_replay_100_confusion_matrices.png` — ER confusion matrices
- `results/continual_learning_metrics.txt` — accuracy report
- `results/fecam_ablation.txt` — FeCAM covariance ablation report

The committed `figures/` and `results/` directories contain the artifacts from the supplied original run. Its reported average accuracies are 78.19% for FeCAM, 97.24% for ER with a 500-sample buffer, and 91.52% for ER with a 100-sample buffer. Exact rerun results may differ by platform and PyTorch version.

## Push to GitHub

Create an empty GitHub repository first, then run these commands from this project folder:

```powershell
git init
git add .
git commit -m "Initial commit: Split-MNIST continual learning comparison"
git branch -M main
git remote add origin <YOUR-REPOSITORY-URL>
git push -u origin main
```

## Reference

FeCAM: Goswami et al., *FeCAM: Exploiting the Heterogeneity of Class Distributions in Exemplar-Free Continual Learning*, NeurIPS 2023.
