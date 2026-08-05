import pytest
import torch

from chained_flow.training.losses import DrafterLossConfig, compute_drafter_loss


def test_hidden_only_loss_does_not_require_lm_head():
    pred = torch.zeros(2, 3, 4)
    target = torch.zeros(2, 3, 4)
    output = compute_drafter_loss(
        pred,
        target,
        config=DrafterLossConfig(lambda_ce=0.0, lambda_expected_accept=0.0),
    )

    assert output.total.item() == pytest.approx(0.0)
    assert set(output.components) == {
        "hidden.mse",
        "hidden.cos",
        "hidden.norm",
    }


def test_latent_losses_are_optional_components():
    pred = torch.zeros(2, 3, 4)
    target = torch.zeros(2, 3, 4)
    pred_latent = torch.zeros(2, 3, 2)
    target_latent = torch.ones(2, 3, 2)

    output = compute_drafter_loss(
        pred,
        target,
        pred_latent=pred_latent,
        target_latent=target_latent,
        config=DrafterLossConfig(
            lambda_ce=0.0,
            lambda_expected_accept=0.0,
            lambda_latent_mse=1.0,
            lambda_latent_cos=0.5,
        ),
    )

    assert "latent.mse" in output.components
    assert "latent.cos" in output.components
    assert output.components["latent.mse"].item() == pytest.approx(1.0)


def test_logit_losses_require_lm_head_and_tokens():
    pred = torch.zeros(1, 2, 4)
    target = torch.zeros(1, 2, 4)

    with pytest.raises(ValueError, match="lm_head is required"):
        compute_drafter_loss(pred, target, future_tokens=torch.tensor([[1, 2]]))

    with pytest.raises(ValueError, match="future_tokens is required"):
        compute_drafter_loss(pred, target, lm_head=torch.nn.Linear(4, 8, bias=False))


def test_combined_loss_has_expected_categories(fake_wrapper):
    target_ids = torch.tensor([[2, 3]])
    target = torch.nn.functional.one_hot(target_ids, num_classes=8).float()
    pred = target.clone()
    future_tokens = torch.tensor([[3, 4]])

    output = compute_drafter_loss(
        pred,
        target,
        future_tokens=future_tokens,
        lm_head=fake_wrapper.lm_head,
    )

    assert "hidden.mse" in output.components
    assert "logit.ce" in output.components
    assert "verifier.expected_accept" in output.components
    assert output.components["hidden.mse"].item() == pytest.approx(0.0)
    assert output.components["hidden.norm"].item() == pytest.approx(0.0)
    assert output.components["logit.ce"].item() < 0.001
    assert output.components["verifier.expected_accept"].item() < -0.99


def test_combined_loss_backpropagates_to_pred_hidden(fake_wrapper):
    target_ids = torch.tensor([[2, 3]])
    target = torch.nn.functional.one_hot(target_ids, num_classes=8).float()
    pred = torch.randn(1, 2, 8, requires_grad=True)
    future_tokens = torch.tensor([[3, 4]])

    output = compute_drafter_loss(
        pred,
        target,
        future_tokens=future_tokens,
        lm_head=fake_wrapper.lm_head,
    )
    output.total.backward()

    assert pred.grad is not None
    assert pred.grad.abs().sum().item() > 0.0
