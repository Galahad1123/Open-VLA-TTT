"""Thin entry points for LeRobot 0.6.1's runners with local processor restoration.

The pinned generic factory special-cases the upstream Evo1Config. Route only
our config through the local loader in the runner's namespace; never modify
LeRobot source files, global policy registries, or the generic factory itself.
"""

from contextlib import contextmanager
from importlib import import_module

from lerobot.policies.factory import make_pre_post_processors as upstream_processors

from .configuration_vla_ttt import VLATTTConfig
from .processor_vla_ttt import make_vla_ttt_pre_post_processors_from_pretrained


def make_pre_post_processors(policy_cfg, pretrained_path=None, pretrained_revision=None, **kwargs):
    if isinstance(policy_cfg, VLATTTConfig) and pretrained_path:
        return make_vla_ttt_pre_post_processors_from_pretrained(
            policy_cfg, pretrained_path, revision=pretrained_revision, **kwargs
        )
    return upstream_processors(policy_cfg, pretrained_path, pretrained_revision, **kwargs)


@contextmanager
def local_processor_routing(*modules):
    """Process-local runner adaptation; restore all aliases even after failure.

    Intended for single CLI invocations, not concurrent embedded runners.
    """
    originals = [(module, module.make_pre_post_processors) for module in modules]
    try:
        for module, _ in originals:
            module.make_pre_post_processors = make_pre_post_processors
        yield
    finally:
        for module, original in originals:
            module.make_pre_post_processors = original


def train():
    runner = import_module("lerobot.scripts.lerobot_train")
    evaluator = import_module("lerobot.scripts.lerobot_eval")
    with local_processor_routing(runner, evaluator):
        runner.main()


def evaluate():
    runner = import_module("lerobot.scripts.lerobot_eval")
    with local_processor_routing(runner):
        runner.main()


def rollout():
    runner = import_module("lerobot.scripts.lerobot_rollout")
    context = import_module("lerobot.rollout.context")
    with local_processor_routing(context):
        runner.main()
