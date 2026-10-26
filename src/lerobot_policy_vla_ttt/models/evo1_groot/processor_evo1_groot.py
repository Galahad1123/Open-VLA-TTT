"""Evo-1 processors with explicitly versioned RoboMME delta semantics."""

from dataclasses import dataclass

import torch
from lerobot.configs.types import NormalizationMode
from lerobot.lerobot_types import TransitionKey
from lerobot.processor import (
    NormalizerProcessorStep,
    ProcessorStep,
    ProcessorStepRegistry,
    UnnormalizerProcessorStep,
)

from ...evo1.processor_evo1 import (
    Evo1ActionProcessorStep,
    Evo1PadStateProcessorStep,
    make_evo1_pre_post_processors,
)


def validate_delta_config(config):
    if config.robomme_action_representation != "delta_joint":
        return
    if (
        config.input_features.get("observation.state") is None
        or config.input_features["observation.state"].shape != (8,)
        or config.output_features.get("action") is None
        or config.output_features["action"].shape != (8,)
        or config.postprocess_action_dim != 8
        or config.binarize_gripper
        or any(
            config.normalization_mapping.get(key) != NormalizationMode.QUANTILES
            for key in ("STATE", "ACTION")
        )
    ):
        raise ValueError(
            "delta_joint requires eight state/action dims, crop8, no gripper binarization, "
            "and STATE/ACTION QUANTILES"
        )


@dataclass
@ProcessorStepRegistry.register(name="vla_ttt_robomme_delta_joint_v1")
class RoboMMEDeltaJointProcessorStep(ProcessorStep):
    """Subtract the raw observation anchor before padding or normalization."""

    version: int = 1

    def __post_init__(self):
        if self.version != 1:
            raise ValueError("Unsupported delta_joint processor version")

    def __call__(self, transition):
        action = transition.get(TransitionKey.ACTION)
        if action is None:
            return transition
        state = transition[TransitionKey.OBSERVATION]["observation.state"]
        if state.shape[-1] != 8 or action.shape[-1] != 8:
            raise ValueError("delta_joint requires raw eight-dimensional state and action")
        anchor = state[..., :7]
        while anchor.ndim < action.ndim:
            anchor = anchor.unsqueeze(-2)
        result = transition.copy()
        result[TransitionKey.ACTION] = action.clone()
        result[TransitionKey.ACTION][..., :7] -= anchor
        return result

    def transform_features(self, features):
        return features

    def get_config(self):
        return {"version": self.version}


@dataclass
@ProcessorStepRegistry.register(name="vla_ttt_robomme_delta_output_v1")
class RoboMMEDeltaOutputProcessorStep(ProcessorStep):
    """Serialized declaration that the postprocessor outputs unnormalized deltas."""

    version: int = 1

    def __post_init__(self):
        if self.version != 1:
            raise ValueError("Unsupported delta_joint processor version")

    def __call__(self, transition):
        return transition

    def transform_features(self, features):
        return features

    def get_config(self):
        return {"version": self.version}


def validate_robomme_processor_pair(config, pre, post):
    validate_delta_config(config)
    delta_steps = [s for s in pre.steps if isinstance(s, RoboMMEDeltaJointProcessorStep)]
    output_steps = [s for s in post.steps if isinstance(s, RoboMMEDeltaOutputProcessorStep)]
    if config.robomme_action_representation == "absolute":
        if delta_steps or output_steps:
            raise ValueError(
                "Checkpoint action representation disagrees with delta_joint processors"
            )
        return
    if len(delta_steps) != 1 or len(output_steps) != 1:
        raise ValueError("delta_joint checkpoint requires versioned delta pre/post processors")
    delta_index = pre.steps.index(delta_steps[0])
    if any(
        isinstance(step, (Evo1PadStateProcessorStep, NormalizerProcessorStep))
        for step in pre.steps[:delta_index]
    ):
        raise ValueError("delta_joint subtraction must precede padding and normalization")
    normalizers = [s for s in pre.steps if isinstance(s, NormalizerProcessorStep)]
    unnormalizers = [s for s in post.steps if isinstance(s, UnnormalizerProcessorStep)]
    crops = [s for s in post.steps if isinstance(s, Evo1ActionProcessorStep)]
    if len(normalizers) != 1 or len(unnormalizers) != 1 or len(crops) != 1:
        raise ValueError("delta_joint requires a matched normalization and crop pipeline")
    if crops[0].action_dim != 8 or crops[0].binarize_gripper:
        raise ValueError("delta_joint requires crop8 and continuous gripper")
    for step in (*normalizers, *unnormalizers):
        for key in ("STATE", "ACTION") if step is normalizers[0] else ("ACTION",):
            if step.norm_map.get(key) != NormalizationMode.QUANTILES:
                raise ValueError("delta_joint processors require QUANTILES")
    for key in ("q01", "q99"):
        left = normalizers[0].stats.get("action", {}).get(key)
        right = unnormalizers[0].stats.get("action", {}).get(key)
        if (
            left is None
            or right is None
            or not torch.equal(torch.as_tensor(left), torch.as_tensor(right))
        ):
            raise ValueError("delta_joint pre/post action quantiles do not match")


def make_evo1_groot_pre_post_processors(config, dataset_stats=None):
    validate_delta_config(config)
    pre, post = make_evo1_pre_post_processors(config, dataset_stats=dataset_stats)
    if config.robomme_action_representation == "delta_joint":
        index = next(
            i for i, step in enumerate(pre.steps) if isinstance(step, Evo1PadStateProcessorStep)
        )
        pre.steps.insert(index, RoboMMEDeltaJointProcessorStep())
        post.steps.append(RoboMMEDeltaOutputProcessorStep())
    return pre, post
