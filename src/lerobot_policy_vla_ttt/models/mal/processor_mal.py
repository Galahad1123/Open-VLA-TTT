"""Baseline-compatible observation and action normalization."""

from ...evo1.processor_evo1 import make_evo1_pre_post_processors


def make_mal_pre_post_processors(config, dataset_stats=None):
    return make_evo1_pre_post_processors(config, dataset_stats=dataset_stats)
