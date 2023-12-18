from .base import BaseVAE
from .beta_vae import BetaVAE
from .beta_vae_CLF import BetaVAE_CLF
from .beta_vae_MultiCLF import BetaVAE_MultiCLF
from .base_clf import LatentClassifier

models = {
    'BetaVAE':BetaVAE,
    'BetaVAE_CLF':BetaVAE_CLF,
    'BetaVAE_MultiCLF':BetaVAE_MultiCLF,
    'LatentClassifier':LatentClassifier,
}
