# The geometry of efficient codes

This is the codebase for the paper "The geometry of efficient codes: how rate-distortion trade-offs distort the latent representations of generative models".

To train one of the models presented in the paper use the `run.py` script. This script requires a configuration file that specifies the model, the dataset, the training parameters, and the evaluation parameters. An example configuration file is provided in `configs/bbvae.yaml`.

Available model names are reported in the `models/__init__.py` file and also reported below:

```python
models = {
    'BetaVAE': BetaVAE, # use this to train baseline, E1M1 and E1M2 models
    'BetaVAE_CLF': BetaVAE_CLF, # use this to train E2M1 - E2M4 models
    'BetaVAE_MultiCLF': BetaVAE_MultiCLF, # use this to train E2M5 model
    'LatentClassifier': LatentClassifier, # use this to train classifiers on precomputed latent spaces 
}
```

To train only classifiers on a given precomputed latent space, use the `run_clf_only.py` script and the `configs/clf_only.yaml` configuration file.

To train multiple classifiers on multiple precomputed latent spaces in sequential order, use the `run_clf_onDiffTask.sh` script. (This is used in the paper to produce Figure 6D in the paper)

