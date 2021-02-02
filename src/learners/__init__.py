from .q_learner import QLearner
from .coma_learner import COMALearner
from .latent_q_learner import LatentQLearner
from .hierarchical_q_learner import HierarchicalQLearner
from .hierarchical_rode_learner import HierarchicalRODELearner
from .hierarchical_noise_q_learner import HierarchicalNoiseQLearner

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY['latent_q_learner'] =LatentQLearner
REGISTRY['hierarchical_q_learner'] =HierarchicalQLearner
REGISTRY['hierarchical_rode_learner'] =HierarchicalRODELearner
REGISTRY['hierarchical_noise_q_learner'] =HierarchicalNoiseQLearner

