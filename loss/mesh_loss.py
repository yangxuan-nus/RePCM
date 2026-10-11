"""Reconstruction and KL losses for biventricular motion completion."""

import torch


def MSE_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared Euclidean distance per vertex over the cardiac sequence."""
    if prediction.shape != target.shape or prediction.shape[-1] != 3:
        raise ValueError("Expected matching (B,T,N,3) vertex sequences")
    return (prediction - target).square().sum(dim=-1).mean()


def VAECELoss(v_pre, v_gt, f, logvar, mu, beta=1e-3, lambd=1.0,
               lambd_s=0.0, loss="mse"):
    """RePCM reconstruction loss and diagonal-Gaussian KL term."""
    if loss != "mse" or lambd_s != 0:
        raise ValueError("The official RePCM objective uses MSE and KL only")
    reconstruction = MSE_loss(v_pre, v_gt)
    kl = -0.5 * (1 + logvar - mu.square() - logvar.exp()).mean()
    return lambd * reconstruction + beta * kl, reconstruction
