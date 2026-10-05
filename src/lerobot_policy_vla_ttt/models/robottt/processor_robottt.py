"""The same observation/action normalization contract as the source baseline."""

from ...evo1.processor_evo1 import make_evo1_pre_post_processors


def make_robottt_pre_post_processors(config, dataset_stats=None):
    return make_evo1_pre_post_processors(config, dataset_stats=dataset_stats)
