import pytest
import torch

from chained_flow.training.vae_losses import HiddenVAELossConfig, compute_hidden_vae_loss, latent_kl_loss


def test_hidden_vae_loss_zero_reconstruction_has_only_kl():
    hidden = torch.zeros(2, 8)
    mu = torch.zeros(2, 3)
    logvar = torch.zeros(2, 3)
    output = compute_hidden_vae_loss(
        hidden,
        hidden,
        mu=mu,
        logvar=logvar,
        config=HiddenVAELossConfig(beta=0.1),
    )

    assert output.components["hidden.mse"].item() == pytest.approx(0.0)
    assert output.components["hidden.norm"].item() == pytest.approx(0.0)
    assert output.components["latent.kl"].item() == pytest.approx(0.0)
    assert output.total.item() == pytest.approx(0.0)


def test_latent_kl_positive_for_shifted_mu():
    mu = torch.ones(2, 3)
    logvar = torch.zeros(2, 3)
    assert latent_kl_loss(mu, logvar).item() > 0.0
