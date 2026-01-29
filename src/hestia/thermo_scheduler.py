import math
from typing import Optional


class ThermoScheduler:
    """
    Scheduler for Pressure and Temperature, including 3 Phases:
        - stage1: Compress (Increase Pressure)
        - stage2: Anneal (Decrease Temperature)

    Hestia Extension: Hessian Trace Informed Softmax Annealing
    - Implements eff_temp = base_temp * temp_scale
    - Higher temp_scale => Slower time progression => Higher temperature for longer
    """
    def __init__(
        self,
        compress_ratio: float,
        init_temp: float,
        temp_decay_style: str,
        anneal_ratio: float,
        end_temp: float,
        temp_scale: Optional[float] = None,
    ):
        self.total_steps = None

        self.compress_ratio = compress_ratio
        self.init_temp = init_temp
        self.anneal_ratio = anneal_ratio
        self.temp_decay_style = temp_decay_style
        self.end_temp = end_temp
        self.temp_scale = temp_scale  # Pre-computed temperature scale factor

    def bind_total_steps(self, total_steps: int):
        """Bind total training steps."""
        self.total_steps = total_steps

    def get_pressure(
        self,
        cur_step: int,
    ) -> float:
        """
        Get the current pressure value based on the training step.

        Args:
            cur_step: current training step

        Returns:
            pressure: current pressure value, in [0, 1]
        """
        assert self.total_steps is not None, "Total steps not bound. Call bind_total_steps first."

        cur_ratio = cur_step / self.total_steps
        if cur_ratio < self.compress_ratio:
            return cur_ratio / self.compress_ratio
        else:
            return 1.0

    def get_temp(
        self,
        cur_step: int,
    ) -> float:
        """
        Get the current temperature values based on the training step.

        If temp_scale is provided, the temperature is scaled by temp_scale.
        temp_scale = exp(-alpha * sensitivity_score) is pre-computed during calibration.

        Args:
            cur_step: current training step

        Returns:
            temp: current temperature value (scaled if temp_scale is provided)
        """
        assert self.total_steps is not None, "Total steps not bound. Call bind_total_steps first."

        cur_ratio = cur_step / self.total_steps
        const_temp_ratio = 1.0 - self.anneal_ratio

        if cur_ratio <= const_temp_ratio:
            eff_temp = self.init_temp
            if self.temp_scale is not None:
                eff_temp = eff_temp * self.temp_scale
        else:
            # Calculate effective temperature
            eff_temp = self._calculate_eff_temp(cur_ratio, const_temp_ratio)

        return eff_temp

    def _calculate_eff_temp(self, cur_ratio: float, const_temp_ratio: float) -> float:
        """
        Calculate effective temperature with temp_scale applied.
        """
        if self.temp_decay_style == "linear":
            coeff = (1 - cur_ratio) / self.anneal_ratio
        elif self.temp_decay_style == "cosine":
            coeff = 0.5 * (1.0 + math.cos(math.pi * (cur_ratio - const_temp_ratio) / self.anneal_ratio))
        elif self.temp_decay_style == "hessian":
            # For hessian style, use cosine as base decay and apply temp_scale
            coeff = 0.5 * (1.0 + math.cos(math.pi * (cur_ratio - const_temp_ratio) / self.anneal_ratio))
            if self.temp_scale is not None:
                coeff = coeff * self.temp_scale
        else:
            raise ValueError(f"Unknown decay style: {self.temp_decay_style}")

        return self.init_temp * coeff

