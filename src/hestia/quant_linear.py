import math
import torch
import torch.nn as nn
from typing import Optional

from hestia.thermo_scheduler import ThermoScheduler
from hestia.thermo_quantizer import ThermoQuantizer, ternary_quant, bitwidth_quant, PhaseQuantizer
from hestia.quant_config import QuantConfig


class HestiaLinear(nn.Linear):
    """
    Hestia Quantized Linear Layer with integrated Scheduler and Quantizer.

    Features:
    - Hessian Trace Informed Softmax Annealing (Hestia)
    - Three-phase training (compress, anneal, solid)
    - Softmax Gibbs distribution for smooth quantization
    - Dead zone prevention through thermal annealing
    - Periodic Hessian calibration support
    """
    def __init__(
        self,
        in_features,
        out_features,
        bias: bool = False,
        qconfig: Optional[QuantConfig] = None,
        layer_id: Optional[str] = None,
        temp_scale: Optional[float] = None,
    ):
        super().__init__(in_features, out_features, bias)
        self.qconfig = qconfig
        self.layer_id = layer_id
        self.temp_scale = temp_scale  # Pre-computed temperature scale factor

        # Hestia Components
        self._init_scheduler()
        self._init_quantizer()

        # Training state
        self.training_step = 0
        self.quantization_enabled = (self.qconfig is not None)
        self.cur_pressure = 0.0
        self.cur_temp = self.qconfig.init_temp

    def _init_scheduler(self):
        """Initialize ThermoScheduler with config parameters."""
        self.scheduler = ThermoScheduler(
            compress_ratio=self.qconfig.compress_ratio,
            init_temp=self.qconfig.init_temp,
            temp_decay_style=self.qconfig.temp_decay_style,
            anneal_ratio=self.qconfig.anneal_ratio,
            end_temp=self.qconfig.end_temp,
            temp_scale=self.temp_scale,
        )

    def _init_quantizer(self):
        """Initialize ThermoQuantizer with codebook."""
        target_device = getattr(self.weight, "device", None)
        target_dtype = getattr(self.weight, "dtype", torch.float32)
        codebook = torch.tensor(self.qconfig.codebook, dtype=target_dtype)

        if target_device is not None:
            codebook = codebook.to(target_device)
        
        self.quantizer = ThermoQuantizer(
            codebook,
            group_size=self.qconfig.group_size
        )

    def _ensure_quantizer_device(self):
        """Ensure quantizer codebook is on the same device as weights."""
        if not hasattr(self.weight, "device"):
            return
        
        target_device = self.weight.device
        target_dtype = self.weight.dtype
        
        if (
            self.quantizer.codebook.device != target_device
            or self.quantizer.codebook.dtype != target_dtype
        ):
            self.quantizer.codebook = self.quantizer.codebook.to(
                device=target_device, dtype=target_dtype
            )
            # Reset cached adjusted copy inside ThermoQuantizer
            if hasattr(self.quantizer, "_cached_codebook"):
                self.quantizer._cached_codebook = None
                self.quantizer._cached_device = None
                self.quantizer._cached_dtype = None

    def update_training_state(self, step: int):
        """Update training state for current step."""
        self.training_step = step
        self.cur_pressure = self.scheduler.get_pressure(step)
        self.cur_temp = self.scheduler.get_temp(step)

    def forward(self, input):
        """
        Hestia Forward pass with thermal quantization.

        Phases:
        1. Compress: Gradually enable quantization (pressure ramp)
        2. Anneal: Softmax Gibbs -> hard discrete (temp decay)
        3. Solid: Pure discrete quantization (default to be disabled)
        """
        if not self.quantization_enabled:
            return super().forward(input)

        self._ensure_quantizer_device()

        # Quantization using cached scales
        # Temperature is already scaled by temp_scale in the scheduler
        quantized_weight = self.quantizer.forward(
            x=self.weight,
            pressure=self.cur_pressure,
            temp=self.cur_temp,
            is_training=True,
        )

        # Forward pass with quantized weights, no bias
        return nn.functional.linear(input=input, weight=quantized_weight)

    def get_quantization_stats(self) -> dict:
        """Get current quantization statistics."""
        return {
            'step': self.training_step,
            'pressure': self.cur_pressure,
            'temperature': self.cur_temp,
            'temp_scale': self.temp_scale,
            'layer_id': self.layer_id,
        }


class FairyHestiaLinear(nn.Linear):
    """
    Hestia Linear layer with complex-phase quantization from Fairy2i.
    """
    def __init__(
        self,
        in_features,
        out_features,
        bias: bool = True,
        qconfig: Optional[QuantConfig] = None,
        layer_id: Optional[str] = None,
        temp_scale: Optional[float] = None,
    ):
        super().__init__(in_features, out_features, bias)
        if self.in_features % 2 != 0 or self.out_features % 2 != 0:
            raise ValueError(
                "FairyHestiaLinear requires even in/out features for complex-phase quantization."
            )
        # Configuration
        self.qconfig = qconfig
        self.layer_id = layer_id
        self.temp_scale = temp_scale  # Pre-computed temperature scale factor

        # Hestia Components
        self._init_scheduler()
        self._init_quantizer()

        # Training state
        self.training_step = 0
        self.quantization_enabled = (self.qconfig is not None)
        self.cur_pressure = 0.0
        self.cur_temp = self.qconfig.init_temp

    def _init_scheduler(self):
        """Initialize ThermoScheduler with config parameters."""
        self.scheduler = ThermoScheduler(
            compress_ratio=self.qconfig.compress_ratio,
            init_temp=self.qconfig.init_temp,
            temp_decay_style=self.qconfig.temp_decay_style,
            anneal_ratio=self.qconfig.anneal_ratio,
            end_temp=self.qconfig.end_temp,
            temp_scale=self.temp_scale,
        )

    def _init_quantizer(self):
        version = getattr(self.qconfig, "phase_quant_version", "v2")
        self.quantizer = PhaseQuantizer(phase_quant_version=version)

    def update_training_state(self, step: int):
        """Update training state for current step."""
        self.training_step = step
        self.cur_pressure = self.scheduler.get_pressure(step)
        self.cur_temp = self.scheduler.get_temp(step)

    def forward(self, input):
        """
        Hestia Forward pass with thermal quantization.

        Phases:
        1. Compress: Gradually enable quantization (pressure ramp)
        2. Anneal: Softmax Gibbs -> hard discrete (temp decay)
        3. Solid: Pure discrete quantization
        """
        if not self.quantization_enabled:
            return super().forward(input)

        # Complex-phase quantization
        quantized_weight = self.quantizer.quantize_weight(
            self.weight,
            temp=self.cur_temp,
            is_training=self.training,
        )

        if self.cur_pressure == 1.0:
            weight = quantized_weight
        elif self.cur_pressure == 0.0:
            weight = self.weight
        else:
            weight = torch.lerp(self.weight, quantized_weight, self.cur_pressure)

        # Forward pass with quantized weights
        return nn.functional.linear(input, weight, self.bias)

    def get_quantization_stats(self) -> dict:
        """Get current quantization statistics."""
        return {
            'step': self.training_step,
            'pressure': self.cur_pressure,
            'temperature': self.cur_temp,
            'temp_scale': self.temp_scale,
            'layer_id': self.layer_id,
        }

class TernaryLinear(nn.Linear):
    """
    Deterministic ternary quantization with a straight-through estimator.
    The forward path uses ternary-quantized weights while gradients flow through
    the original weights unchanged.
    """

    def __init__(
        self,
        in_features,
        out_features,
        bias: bool = False,
        qconfig: Optional[QuantConfig] = None,
    ):
        super().__init__(in_features, out_features, bias)
        self.qconfig = qconfig

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Hard ternary quantization
        group_size = self.qconfig.group_size if self.qconfig else 0
        q_weight = ternary_quant(self.weight, group_size=group_size)
        # STE: replace forward value but keep backward path to full-precision weight
        weight = self.weight + (q_weight - self.weight).detach()
        return nn.functional.linear(input, weight, self.bias)


class BitwidthLinear(nn.Linear):
    """
    Deterministic bitwidth quantization with a straight-through estimator.
    The forward path uses bitwidth-quantized weights while gradients flow through
    the original weights unchanged.
    """

    def __init__(
        self,
        in_features,
        out_features,
        bias: bool = False,
        qconfig: Optional[QuantConfig] = None,
    ):
        super().__init__(in_features, out_features, bias)
        self.qconfig = qconfig

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        group_size = self.qconfig.group_size if self.qconfig else 0
        bitwidth = getattr(self.qconfig, "bitwidth", 2) if self.qconfig else 2
        is_symmetric = getattr(self.qconfig, "is_symmetric", True) if self.qconfig else True
        q_weight = bitwidth_quant(
            self.weight,
            group_size=group_size,
            bitwidth=bitwidth,
            is_symmetric=is_symmetric,
        )
        weight = self.weight + (q_weight - self.weight).detach()
        return nn.functional.linear(input, weight, self.bias)

