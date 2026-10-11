"""Lightweight checks for the RePCM model and prior."""

import torch

from loss.mesh_loss import MSE_loss
from model.repcm import RePCM


def make_model():
    model = RePCM(
        enc_depth=2, dec_depth=2, dim=32, heads=2, dim_head=16,
        T=3, latent_dim=16, num_traj=8, num_tokens=4,
    )
    model.set_fixed_idx(torch.arange(8))
    model.set_vertex_token_id(torch.arange(12) % 4)
    model.build_anchor_token_id()
    model.set_token_graph(torch.ones(4, 4, dtype=torch.bool))
    model.set_anchor_graph(torch.ones(8, 8, dtype=torch.bool))
    return model


def test_prior_variance_uses_weighted_expert_variances():
    model = make_model()
    with torch.no_grad():
        for index, (mean, logvar) in enumerate(zip(model.prior_mean_xt_experts,
                                                   model.prior_logvar_xt_experts)):
            mean.weight.zero_()
            mean.bias.fill_(index)
            logvar.weight.zero_()
            logvar.bias.zero_()
        model._gate_weights = lambda emb: emb.new_full((len(emb), 4), 0.25)
        mu, lv, weights = model._moe_prior_params(torch.randn(2, 8, 32), torch.randn(2, 16))
    assert torch.allclose(mu, mu.new_full(mu.shape, 1.5))
    assert torch.allclose(lv, torch.zeros_like(lv), atol=1e-6)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2))


def test_ema_prototypes_are_updated():
    model = make_model()
    embeddings = torch.randn(8, 16)
    model.update_shape_prototypes(embeddings)
    assert model.shape_protos_valid.item() == 1
    before = model.shape_protos.clone()
    model.update_shape_prototypes(embeddings + 0.5, momentum=0.9)
    assert not torch.equal(before, model.shape_protos)


def test_ed_only_generation_and_backprop():
    model = make_model()
    sequence = torch.randn(2, 3, 12, 3)
    train_output = model(sequence, sequence[:, 0], single_frame=False)
    loss = MSE_loss(train_output["recon_pc"], sequence) + train_output["kl_temp"].mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert model.to_outputs.weight.grad is not None
    model.update_shape_prototypes(train_output["e_s"])
    with torch.no_grad():
        pred = model(sequence[:, :1], sequence[:, 0], single_frame=True)["recon_pc"]
    assert pred.shape == sequence.shape
