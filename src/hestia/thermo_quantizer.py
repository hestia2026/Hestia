import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math

def _reshape_for_grouping(
    x: torch.Tensor,
    group_size: int
) -> tuple[torch.Tensor, torch.Size]:
    """
    Reshape tensor based on group_size for group-wise quantization.

    Args:
        x: Input tensor
        group_size: Group size for grouping

    Returns:
        Tuple of (reshaped_tensor, original_x_shape)
    """
    org_x_shape = x.shape

    if group_size > 0:
        assert org_x_shape[-1] % group_size == 0, \
            f"For group_size={group_size}, last dimension {org_x_shape[-1]} must be divisible by group_size"
        shape = (-1, group_size) # Block-wise: (-1, group_size)
    elif group_size == -1:
        shape = (-1, org_x_shape[-1]) # Channel-wise: (-1, org_w_shape[-1]) - keep last dimension
    elif group_size == 0:
        shape = (1, -1) # Per-tensor: (1, -1) - flatten all
    else:
        raise ValueError(f"Invalid group_size: {group_size}")

    return x.reshape(*shape), org_x_shape


class ThermoQuantizer(nn.Module):
    """
    Differentiable quantizer using Gumbel-Softmax
    """

    def __init__(
        self,
        codebook: torch.Tensor,
        group_size: int = 0,
    ) -> None:
        super().__init__()
        # Keep base codebook as buffer; cache adjusted copies per device/dtype.
        self.register_buffer("codebook", codebook)
        self.group_size = group_size
        self._cached_codebook = None
        self._cached_device = None
        self._cached_dtype = None

    def forward(
        self,
        x: torch.Tensor,
        pressure: float,
        temp: float,
        is_training: bool = True,
    ) -> torch.Tensor:
        """
        Quantizer Forward Pass With Gumbel-Softmax-Approximate Quantization

        Args:
            x: Input tensor
            pressure: Interpolation factor between full-precision and quantized output
            temp: Temperature parameter (already scaled by temp_scale in scheduler)
            is_training: Whether in training mode

        Returns:
            Quantized tensor
        """
        # Avoid per-step copies; refresh only when device/dtype changes.
        if (
            self._cached_codebook is None
            or self._cached_device != x.device
            or self._cached_dtype != x.dtype
        ):
            self._cached_codebook = self.codebook.to(device=x.device, dtype=x.dtype)
            self._cached_device = x.device
            self._cached_dtype = x.dtype

        codebook = self._cached_codebook

        reshaped_x, org_x_shape = _reshape_for_grouping(x, self.group_size)
        scales = 1 / reshaped_x.abs().mean(dim=1, keepdim=True).clamp(min=1e-5)
        x_norm = reshaped_x * scales

        if temp > 0.0:
            # Use temperature (already scaled by temp_scale in scheduler)
            logits = -(x_norm.unsqueeze(-1) - codebook).pow(2)
            prob = F.softmax(logits / (temp + 1e-6), dim=-1)
            qx_norm = torch.sum(prob * codebook, dim=-1)
            qx = (qx_norm / scales).to(dtype=x.dtype)
        else:
            # Hard quantization (temp = 0)
            qx = x_norm.round().clamp(-1, 1) / scales
            if is_training:
                qx = reshaped_x + qx - reshaped_x.detach()

        if pressure == 1.0:
            return qx.reshape(org_x_shape)
        elif pressure == 0.0:
            return x
        else:
            return torch.lerp(x, qx.reshape(org_x_shape), pressure)


def ternary_quant(
    x: torch.Tensor,
    group_size: int = 0,
) -> torch.Tensor:
    reshaped_x, org_x_shape = _reshape_for_grouping(x, group_size)
    scales = 1 / reshaped_x.abs().mean(dim=1, keepdim=True).clamp(min=1e-5)
    x_norm = reshaped_x * scales
    qx = x_norm.round().clamp(-1, 1) / scales
    return qx.reshape(org_x_shape)


def bitwidth_quant(
    x: torch.Tensor,
    group_size: int = 0,
    bitwidth: int = 2,
    is_symmetric: bool = True,
) -> torch.Tensor:
    reshaped_x, org_x_shape = _reshape_for_grouping(x, group_size)
    if is_symmetric:
        qmax = 2 ** (bitwidth - 1) - 1
        qmin = -2 ** (bitwidth - 1)
        max_abs = reshaped_x.abs().amax(dim=1, keepdim=True).clamp(min=1e-5)
        scales = qmax / max_abs
        x_norm = reshaped_x * scales
        qx = x_norm.round().clamp(qmin, qmax) / scales
    else:
        qmin = 0
        qmax = 2**bitwidth - 1
        min_val = reshaped_x.amin(dim=1, keepdim=True)
        max_val = reshaped_x.amax(dim=1, keepdim=True)
        scales = (max_val - min_val).clamp(min=1e-5) / (qmax - qmin)
        zero_point = (qmin - torch.round(min_val / scales)).clamp(qmin, qmax)
        x_norm = reshaped_x / scales + zero_point
        qx = ((x_norm.round()).clamp(qmin, qmax) - zero_point) * scales
    return qx.reshape(org_x_shape)


def _phase_quant_base(
    w_real: torch.Tensor,
    w_imag: torch.Tensor,
    temp: float,
    is_training: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    phase = torch.angle(w_real + 1j * w_imag)

    real_pos = (phase >= -math.pi / 4) & (phase < math.pi / 4)
    real_neg = (phase >= 3 * math.pi / 4) | (phase < -3 * math.pi / 4)
    imag_pos = (phase >= math.pi / 4) & (phase < 3 * math.pi / 4)
    imag_neg = (phase >= -3 * math.pi / 4) & (phase < -math.pi / 4)

    mask_real = real_pos | real_neg
    mask_imag = imag_pos | imag_neg

    s_re = (
        w_real[mask_real].abs().mean()
        if mask_real.any()
        else torch.tensor(0.0, device=w_real.device)
    )
    s_im = (
        w_imag[mask_imag].abs().mean()
        if mask_imag.any()
        else torch.tensor(0.0, device=w_imag.device)
    )

    s_re = torch.clamp(s_re, min=1e-6)
    s_im = torch.clamp(s_im, min=1e-6)

    if temp > 0.0:
        centers = torch.tensor(
            [0.0, math.pi / 2, math.pi, -math.pi / 2],
            device=phase.device,
            dtype=phase.dtype,
        )
        delta = torch.atan2(
            torch.sin(phase.unsqueeze(-1) - centers),
            torch.cos(phase.unsqueeze(-1) - centers),
        )
        logits = -(delta ** 2)
        prob = torch.softmax(logits / (temp + 1e-6), dim=-1)
        qw_real = (prob[..., 0] - prob[..., 2]) * s_re
        qw_imag = (prob[..., 1] - prob[..., 3]) * s_im
        return qw_real.to(w_real.dtype), qw_imag.to(w_imag.dtype)

    qw_real = torch.zeros_like(w_real)
    qw_imag = torch.zeros_like(w_imag)

    qw_real[real_pos] = 1.0
    qw_real[real_neg] = -1.0
    qw_imag[imag_pos] = 1.0
    qw_imag[imag_neg] = -1.0

    qw_real = qw_real * s_re
    qw_imag = qw_imag * s_im

    if is_training:
        qw_real = w_real + (qw_real - w_real).detach()
        qw_imag = w_imag + (qw_imag - w_imag).detach()

    return qw_real.to(w_real.dtype), qw_imag.to(w_imag.dtype)


def _phase_quant_residual(
    w_real: torch.Tensor,
    w_imag: torch.Tensor,
    steps: int,
    temp: float,
    is_training: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    qw_real = torch.zeros_like(w_real)
    qw_imag = torch.zeros_like(w_imag)
    err_real = w_real
    err_imag = w_imag
    for _ in range(steps):
        q_real, q_imag = _phase_quant_base(err_real, err_imag, temp, is_training)
        qw_real = qw_real + q_real
        qw_imag = qw_imag + q_imag
        err_real = err_real - q_real
        err_imag = err_imag - q_imag
    return qw_real, qw_imag


class PhaseQuantizer(nn.Module):
    """Complex-phase quantizer with residual steps and temperature control."""

    def __init__(self, phase_quant_version: str = "v2") -> None:
        super().__init__()
        self.phase_quant_version = phase_quant_version

    def _resolve_phase_steps(self) -> int:
        version = (self.phase_quant_version or "v2").lower()
        mapping = {"v1": 1, "v2": 2, "v3": 3, "v4": 4}
        if version not in mapping:
            raise ValueError(
                f"Unsupported phase_quant_version: {version}. Use v1/v2/v3/v4."
            )
        return mapping[version]

    def forward(
        self,
        w_real: torch.Tensor,
        w_imag: torch.Tensor,
        temp: float,
        is_training: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        steps = self._resolve_phase_steps()
        return _phase_quant_residual(
            w_real,
            w_imag,
            steps=steps,
            temp=temp,
            is_training=is_training,
        )

    def quantize_weight(
        self,
        weight: torch.Tensor,
        temp: float,
        is_training: bool = True,
    ) -> torch.Tensor:
        n, m = weight.shape[0] // 2, weight.shape[1] // 2
        a11, a12 = weight[:n, :m], weight[:n, m:]
        a21, a22 = weight[n:, :m], weight[n:, m:]

        u_re = 0.5 * (a11 + a22)
        u_im = 0.5 * (a21 - a12)
        w_re = 0.5 * (a11 - a22)
        w_im = 0.5 * (a12 + a21)

        u_re_q, u_im_q = self.forward(u_re, u_im, temp=temp, is_training=is_training)
        w_re_q, w_im_q = self.forward(w_re, w_im, temp=temp, is_training=is_training)

        a11_q = w_re_q + u_re_q
        a12_q = w_im_q - u_im_q
        a21_q = w_im_q + u_im_q
        a22_q = -w_re_q + u_re_q

        a_quant_top = torch.cat([a11_q, a12_q], dim=1)
        a_quant_bottom = torch.cat([a21_q, a22_q], dim=1)
        return torch.cat([a_quant_top, a_quant_bottom], dim=0)