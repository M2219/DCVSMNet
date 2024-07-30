import torch.nn.functional as F
import torch
from typing import List


def model_loss_train(
    disp_ests: List[torch.Tensor],
    disp_gts: List[torch.Tensor],
    img_masks: List[torch.Tensor],
) -> torch.Tensor:
    weights = [1.0, 0.3]
    all_losses = []
    for disp_est, disp_gt, weight, mask_img in zip(
        disp_ests, disp_gts, weights, img_masks
    ):
        all_losses.append(
            weight
            * F.smooth_l1_loss(disp_est[mask_img], disp_gt[mask_img], reduction="mean")
        )
    return sum(all_losses)


def model_loss_test(
    disp_ests: List[torch.Tensor],
    disp_gts: List[torch.Tensor],
    img_masks: List[torch.Tensor],
) -> torch.Tensor:
    weights = [1.0]
    all_losses = []
    for disp_est, disp_gt, weight, mask_img in zip(
        disp_ests, disp_gts, weights, img_masks
    ):
        all_losses.append(
            weight * F.l1_loss(disp_est[mask_img], disp_gt[mask_img], reduction="mean")
        )
    return sum(all_losses)
