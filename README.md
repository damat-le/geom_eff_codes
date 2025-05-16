# The Geometry of Efficient Codes

This repository contains the code for the paper **"The geometry of efficient codes: how rate-distortion trade-offs distort the latent representations of generative models."** 


- Paper: [![Static Badge](https://img.shields.io/badge/DOI-https%3A%2F%2Fdoi.org%2F10.1371%2Fjournal.pcbi.1012952-007ec6)](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012952)
- Datasets and models' checkpoints: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14844111.svg)](https://doi.org/10.5281/zenodo.14844111)

## Overview

The study investigates how constraints on capacity, data distributions, and tasks influence the geometry of latent representations in generative models based on rate-distortion theory.

We employ **Beta Variational Autoencoders ($\beta$-VAEs)** to explore how different constraints lead to three primary distortions in latent space:

- **Prototypization** – Collapsing representations into category prototypes.
- **Specialization** – Prioritizing frequent or task-relevant stimuli.
- **Orthogonalization** – Aligning representations with task-relevant dimensions.

## Repository Structure

```
geom_eff_codes
│-- configs/               # Configuration files for different model settings
│-- data/                  # Placeholder for dataset storage
│-- logs/                  # Model checkpoints and logs
│-- notebooks/             # Utility functions for training and evaluation
│-- src/
│   │--models/             # Implementation of the β-VAE and classifiers
│   │--datasets/           # Dataset and Dataloader classes for the Corridors dataset
│   │--experiments/        # Lightning modules for training and evaluation
│   │--utils/              # Utility functions for training and evaluation
│-- run.py                 # Main script to run experiments
│-- requirements.txt       # List of dependencies
│-- README.md              # This file
```

(*Note:* The files `run_clf_onDiffTask.py`, `run_clf_only.py`, and `run_clf_onDiffTask.sh` are not included in this README but their are similar to the `run.py` script.)

## Installation

Ensure you have Python 3.10+ installed. Then, in a new virtual environment, install the required dependencies:

```bash
pip install -r requirements.txt
```

Note: the Python version used in the experiments is `3.10.12`.

## Usage

### Training the β-VAE Models

To train one of the models presented in the paper use the `run.py` script. This script requires a configuration file that specifies the model, the dataset, the training parameters, and the evaluation parameters.

Available configuration files are:
- `bbvae.yaml` → Baseline β-VAE model
- `bbvae_CLF.yaml` → Hybrid β-VAE with classification objective
- `bbvae_MultiCLF.yaml` → β-VAE with multiple classification objectives

To train and evaluate the β-VAE on the **Corridors dataset**, run:

```bash
python run.py --config configs/bbvae.yaml
```

You can modify the configuration file to adjust hyperparameters such as model capacity, β value, or dataset settings.

Available model names are reported in the `models/__init__.py` file and also reported below:

```python
models = {
    'BetaVAE': BetaVAE, # use this to train baseline, E1M1 and E1M2 models
    'BetaVAE_CLF': BetaVAE_CLF, # use this to train E2M1 - E2M4 models
    'BetaVAE_MultiCLF': BetaVAE_MultiCLF, # use this to train E2M5 model
    'LatentClassifier': LatentClassifier, # use this to train classifiers on precomputed latent spaces 
}
```

Additionaly, to train only classifiers on a given precomputed latent space, use the `run_clf_only.py` script and the `configs/clf_only.yaml` configuration file.

To train multiple classifiers on multiple precomputed latent spaces in sequential order, use the `run_clf_onDiffTask.sh` script. (This is used to produce Figure 6D in the paper)


## Experimental Setup

### Datasets

The **Corridors dataset** (`mazes_200k_2corridors_13x13.csv`) consists of 13×13 black-and-white images, each containing two vertical corridors whose positions vary independently. This dataset is designed to test how generative models encode independent factors of variation.

### Main Experiments

1. **Effect of Data Distributions:**
   - Training the model on balanced vs. unbalanced datasets to observe specialization effects.
   - Training datasets: `mazes_100k_2corridors_BottomBiasL1000R100.csv`, `mazes_85k_2corridors_AlignedBias3000vs300.csv`
2. **Effect of Classification Tasks:**
   - Adding classification objectives to the β-VAE to examine orthogonalization.
   - Training datasets: `data/mazes_200k_2corridors_13x13.csv` (the model auto-adjusts the labels according to the task)
3. **Effect of Encoding Capacity:**
   - Comparing how low vs. high capacity training regimes to examine prototypization.


## Results

The study found that varying capacity, data distributions, and classification tasks led to systematic changes in the geometry of latent representations:

- **Lower capacity models tend to prototype representations.**
- **Unbalanced datasets result in specialized encoding, focusing on frequent stimuli.**
- **Classification tasks encourage orthogonalization of latent representations.**

These findings contribute to understanding how generative models compress information efficiently under capacity constraints.

## Citation

If you use this code, please cite:

```
@article{d2024geometry,
  title={The geometry of efficient codes: how rate-distortion trade-offs distort the latent representations of generative models},
  author={D'Amato, Leo and Lancia, Gian Luca and Pezzulo, Giovanni},
  journal={arXiv preprint arXiv:2406.07269},
  year={2024}
}
```

