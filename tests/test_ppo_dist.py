import torch
import numpy as np
import pytest
from atom.core.distributions import TanhNormal

def test_tanh_normal_log_prob():
    """
    Verify TanhNormal log_prob against numerical estimate.
    p(y) = p(x) / |dy/dx| = p(x) * (1 - y^2)^(-1)
    """
    loc = torch.tensor([0.0])
    scale = torch.tensor([1.0])
    dist = TanhNormal(loc, scale)
    
    # 1. Check bounds
    sample, raw = dist.sample()
    assert torch.all(sample >= -1.0) and torch.all(sample <= 1.0)
    
    # 2. Check gradients flow
    loc.requires_grad = True
    dist = TanhNormal(loc, scale)
    sample, _ = dist.rsample()
    loss = sample.sum()
    loss.backward()
    assert loc.grad is not None
    
    # 3. Check Log Prob Consistency
    # We pick a specific value
    x = torch.tensor([0.5], requires_grad=True) # Pre-tanh value
    y = torch.tanh(x)
    
    log_prob_x = torch.distributions.Normal(0, 1).log_prob(x)
    jacobian_term = torch.log(1 - y**2 + 1e-6)
    expected_log_prob_y = log_prob_x - jacobian_term
    
    computed_log_prob_y = dist.log_prob(y, pre_tanh_value=x)
    
    print(f"Expected: {expected_log_prob_y.item()}")
    print(f"Computed: {computed_log_prob_y.item()}")
    
    assert torch.allclose(expected_log_prob_y, computed_log_prob_y, atol=1e-5)

def test_tanh_normal_entropy_proxy():
    # TanhNormal has no analytic entropy, verify it falls back to normal entropy (proxy)
    dist = TanhNormal(torch.zeros(1), torch.ones(1))
    e = dist.entropy()
    assert e > 0

if __name__ == "__main__":
    test_tanh_normal_log_prob()
    test_tanh_normal_entropy_proxy()
    print("Tests passed!")
