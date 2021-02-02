REGISTRY = {}

from .rnn_agent import RNNAgent
from .latent_ce_dis_rnn_agent import LatentCEDisRNNAgent
from .hierarchical_rnn_agent import HieRNNAgent
from .hierarchical_rode_agent import HieRodeAgent
from .hierarchical_noise_agent import HieNoiseAgent
from .rode_agent import RODEAgent

REGISTRY["rnn"] = RNNAgent
REGISTRY["latent_ce_dis_rnn"] = LatentCEDisRNNAgent
REGISTRY["hierarchical_rnn"] = HieRNNAgent
REGISTRY["hierarchical_rode"] = HieRodeAgent
REGISTRY["hierarchical_noise"] = HieNoiseAgent
REGISTRY["rode"] = RODEAgent


