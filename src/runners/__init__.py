REGISTRY = {}

from .episode_runner import EpisodeRunner
REGISTRY["episode"] = EpisodeRunner

from .parallel_runner import ParallelRunner
REGISTRY["parallel"] = ParallelRunner

from .meta_runner import MetaRunner
REGISTRY["meta"] = MetaRunner

from .meta_noise_runner import MetaNoiseRunner
REGISTRY["meta_noise"] = MetaNoiseRunner
