from .base import *
from .beta_vae import *
from .beta_vae_CLF import *
from .beta_vae_MultiCLF import *
from .base_clf import *

vae_models = {
    'BetaVAE':BetaVAE,
    'BetaVAE_CLF':BetaVAE_CLF,
    'BetaVAE_MultiCLF':BetaVAE_MultiCLF,
    'LatentClassifier':LatentClassifier,
}
