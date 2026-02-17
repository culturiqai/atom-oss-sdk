import math
from typing import Optional, Tuple

import torch
import torch.distributions as td


class TanhNormal(td.Distribution):
    """
    Tanh-squashed Normal distribution.

    We define:
        z ~ Normal(loc, scale)
        a = tanh(z)   elementwise

    For PPO, we need log p(a). Change-of-variables gives:
        log p(a) = log p(z) - log |det(da/dz)|
                = log p(z) - sum_i log(1 - tanh(z_i)^2)

    Notes:
    - This implementation returns *elementwise* log_prob with the same shape as `loc`.
      For vector actions, the caller should sum over the last dim (action dim):
          logp = dist.log_prob(a).sum(-1, keepdim=True)
    - If `pre_tanh_value` is provided, we avoid atanh inversion and use it directly
      for numerical stability.
    """

    arg_constraints = {}  # we don't enforce here; upstream should ensure valid scales
    has_rsample = True

    def __init__(self, loc: torch.Tensor, scale: torch.Tensor, epsilon: float = 1e-6):
        if not torch.is_tensor(loc) or not torch.is_tensor(scale):
            raise TypeError("TanhNormal expects torch.Tensor loc and scale")

        # Critical: torch Distribution semantics for vector actions:
        # loc shape = (..., D) where D is action_dim
        self.epsilon = float(epsilon)
        self.normal = td.Normal(loc, scale)

        batch_shape = loc.shape[:-1]
        event_shape = loc.shape[-1:]
        super().__init__(batch_shape=batch_shape, event_shape=event_shape)

    # --- sampling -------------------------------------------------------------

    def sample(self, sample_shape: torch.Size = torch.Size()) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Non-reparameterized sample.

        Returns:
            action: tanh(z)
            pre_tanh: z
        """
        z = self.normal.sample(sample_shape)
        return torch.tanh(z), z

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reparameterized sample.

        Returns:
            action: tanh(z)
            pre_tanh: z
        """
        z = self.normal.rsample(sample_shape)
        return torch.tanh(z), z

    # --- log prob -------------------------------------------------------------

    @staticmethod
    def _atanh(x: torch.Tensor) -> torch.Tensor:
        # torch.atanh exists in modern torch, but keep a safe fallback for portability.
        if hasattr(torch, "atanh"):
            return torch.atanh(x)
        return 0.5 * torch.log((1.0 + x) / (1.0 - x))

    def log_prob(self, value: torch.Tensor, pre_tanh_value: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Elementwise log probability log p(a).

        Args:
            value: action `a` in (-1, 1), shape (..., D)
            pre_tanh_value: `z` used to generate `a=tanh(z)`, same shape as `value`.
                            If provided, avoids inversion and is more stable.

        Returns:
            elementwise log_prob with same shape as `value` / `loc`.
            Caller typically sums across action dim.
        """
        if pre_tanh_value is None:
            # Clamp only for inversion stability; does NOT change the conceptual distribution.
            eps = self.epsilon
            value = torch.clamp(value, -1.0 + eps, 1.0 - eps)
            z = self._atanh(value)
        else:
            z = pre_tanh_value
            # If a caller passed a clamped/modified `value`, do NOT use it for correction.
            # We compute correction from z directly (stable form), which is the true source variable.

        # Base log-prob in z-space (elementwise)
        log_prob_z = self.normal.log_prob(z)

        # Jacobian correction: log(1 - tanh(z)^2), computed stably.
        # log(1 - tanh(z)^2) = 2 * (log(2) - z - softplus(-2z))
        correction = 2.0 * (math.log(2.0) - z - torch.nn.functional.softplus(-2.0 * z))

        return log_prob_z - correction

    # --- entropy --------------------------------------------------------------

    def entropy(self) -> torch.Tensor:
        """
        Proxy entropy.

        True tanh-squashed entropy has no simple analytic form.
        Returning Normal entropy is a common PPO proxy, but it can overestimate action-space entropy
        when tanh saturates.

        If you want a better entropy signal, use Monte Carlo outside:
            a, _ = dist.rsample()
            ent_mc = -dist.log_prob(a).sum(-1).mean()
        """
        return self.normal.entropy()
