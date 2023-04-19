from .base import *
from .beta_vae import *
from .beta_vae_CLF import *
from .base_clf import *

vae_models = {
    'BetaVAE':BetaVAE,
    'BetaVAE_CLF':BetaVAE_CLF,
    'LatentClassifier':LatentClassifier,
}
