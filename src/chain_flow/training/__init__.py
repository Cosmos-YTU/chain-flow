"""Training: the offline teacher-state collection and the drafter/VAE trainers.

The drafter is trained against CACHED target hidden states, not a live target -- see
`collect_teacher` for the collection step and `window_dataset` for how a cache becomes
training windows.
"""
from chain_flow.training.losses import DrafterLossConfig, DrafterLossOutput, compute_drafter_loss
from chain_flow.training.window_dataset import TeacherWindowDataset

__all__ = ["DrafterLossConfig", "DrafterLossOutput", "compute_drafter_loss",
           "TeacherWindowDataset"]
