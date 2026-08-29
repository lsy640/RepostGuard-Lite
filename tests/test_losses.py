from __future__ import annotations

import torch

from repostguard.losses import symmetric_bernoulli_kl


def test_symmetric_bernoulli_kl_is_finite_for_extreme_float16_logits() -> None:
    clean_logits = torch.tensor([20.0, -20.0, 0.0, 12.0], dtype=torch.float16, requires_grad=True)
    augmented_logits = torch.tensor([-20.0, 20.0, 0.5, 10.0], dtype=torch.float16, requires_grad=True)

    loss = symmetric_bernoulli_kl(clean_logits, augmented_logits)
    loss.backward()

    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)
    assert clean_logits.grad is not None
    assert augmented_logits.grad is not None
    assert torch.isfinite(clean_logits.grad).all()
    assert torch.isfinite(augmented_logits.grad).all()


def test_symmetric_bernoulli_kl_is_zero_for_identical_logits() -> None:
    logits = torch.tensor([-8.0, -1.0, 0.0, 2.0, 8.0], dtype=torch.float16)

    loss = symmetric_bernoulli_kl(logits, logits.clone())

    torch.testing.assert_close(loss, torch.zeros((), dtype=torch.float32))


def test_symmetric_bernoulli_kl_is_symmetric() -> None:
    first = torch.tensor([-3.0, 0.25, 4.0], dtype=torch.float32)
    second = torch.tensor([2.0, -0.5, 1.0], dtype=torch.float32)

    forward = symmetric_bernoulli_kl(first, second)
    reverse = symmetric_bernoulli_kl(second, first)

    torch.testing.assert_close(forward, reverse)
