"""Hiwonder NexArm 6-DOF data configuration and dataset mixture for TurboVLA."""

from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import (
    StateActionToTensor,
    StateActionTransform,
)


class NexArmDataConfig:
    """Two-camera (front, wrist), 6-DOF NexArm robot data with 16-step action horizon."""

    video_keys = [
        "video.front",
        "video.wrist",
    ]
    state_keys = [
        "state.state",
    ]
    action_keys = [
        "action.action",
    ]
    language_keys = [
        "annotation.human.action.task_description",
    ]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        return {
            "video": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.video_keys,
            ),
            "state": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.state_keys,
            ),
            "action": ModalityConfig(
                delta_indices=self.action_indices,
                modality_keys=self.action_keys,
            ),
            "language": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.language_keys,
            ),
        }

    def transform(self):
        state_modes = {
            "state.state": "min_max",
        }
        action_modes = {
            "action.action": "min_max",
        }
        return ComposedModalityTransform(
            transforms=[
                StateActionToTensor(apply_to=self.state_keys),
                StateActionTransform(
                    apply_to=self.state_keys,
                    normalization_modes=state_modes,
                ),
                StateActionToTensor(apply_to=self.action_keys),
                StateActionTransform(
                    apply_to=self.action_keys,
                    normalization_modes=action_modes,
                ),
            ]
        )


ROBOT_TYPE_CONFIG_MAP = {"nexarm": NexArmDataConfig()}
ROBOT_TYPE_TO_EMBODIMENT_TAG = {}

DATASET_NAMED_MIXTURES = {
    "nexarm_stack_bowls": [
        ("nexarm_stack_bowls", 1.0, "nexarm"),
    ]
}
