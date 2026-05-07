import json
from pathlib import Path
from typing import Literal, Optional

from hestia.opt_temp_initializer import OptimalTempInitializer


class QuantConfig:
    """
    Configuration class for Hestia quantization framework.
    Supports all parameters required for the complete Hestia pipeline.
    """

    # === Hessian Calibration Parameters ===
    num_sketch: int = 10
    num_query: int = 20
    num_batches: int = 5
    calibration_granularity: Literal["tensor", "layer", "component"] = "tensor"
    # - "tensor": One trace per Linear layer (q_proj, k_proj, v_proj each have their own)
    # - "layer": One trace per transformer layer/block (all Linear layers in same layer share)
    # - "component": One trace per component type (all q_proj share, all k_proj share, etc.)

    # quant map
    codebook: list = [-1.0, 0.0, 1.0]

    # stage1: compress
    compress_ratio: float = 0.2
    init_temp: float = OptimalTempInitializer.calculate()

    # stage2: anneal
    anneal_ratio: float = 0.8
    temp_decay_style: Literal["linear", "cosine", "hessian"] = "cosine"
    end_temp: float = 0.0

    # === Hestia (Hessian Trace Informed Softmax Annealing) ===
    enable_hestia: bool = False  # Enable Hestia Hessian-aware scheduling
    kappa: float = 1.0  # Gain factor for sensitivity score calculation
    alpha: float = 0.5  # Asynchrony coefficient for time dilation
    trace_thres: Optional[float] = None  # Minimum trace value for numerical stability

    # === Group-wise Quantization Parameters ===
    # group_size controls the grouping strategy:
    # - group_size > 0: block-wise grouping, each block has group_size elements
    # - group_size == -1: channel-wise grouping (per output channel)
    # - group_size == 0: per-tensor grouping (global scaling)
    group_size: int = 0

    # === Fairy complex-phase quantization parameters ===
    phase_quant_version: str = "v2"

    # === Bitwidth Quantization Parameters ===
    bitwidth: int = 2
    is_symmetric: bool = True
    quant_type: Optional[str] = None

    _BITWIDTH_PRESETS = {
        "w2_sym": (2, True),
        "w2_asym": (2, False),
        "w4_sym": (4, True),
        "w4_asym": (4, False),
    }

    def validate(self) -> bool:
        if self.quant_type is not None:
            self.apply_quant_type(self.quant_type)

        if self.anneal_ratio <= 0.0:
            raise ValueError("Anneal Process Is Necessary")
        
        if self.anneal_ratio + self.compress_ratio > 1.0:
            raise ValueError("Planned Process Exceeds 100%")

        if not isinstance(self.group_size, int):
            raise ValueError("group_size must be an int.")
        if self.group_size < -1:
            raise ValueError("group_size must be -1, 0, or a positive int.")
        if self.group_size == -1:
            pass
        elif self.group_size == 0:
            pass
        elif self.group_size > 0:
            pass
        else:
            raise ValueError("group_size must be -1, 0, or a positive int.")

        if not isinstance(self.bitwidth, int):
            raise ValueError("bitwidth must be an int.")
        if self.bitwidth < 2:
            raise ValueError("bitwidth must be >= 2.")
        if not isinstance(self.is_symmetric, bool):
            raise ValueError("is_symmetric must be a bool.")

        assert len(self.codebook) == 3, "Codebook must have exactly 3 values for ternary quantization."
        if self.phase_quant_version not in {"v1", "v2", "v3", "v4"}:
            raise ValueError("phase_quant_version must be one of: v1, v2, v3, v4.")
        return True

    def apply_quant_type(self, quant_type: str) -> None:
        """
        Update bitwidth and symmetric settings based on quant_type presets.
        """
        self.quant_type = quant_type
        if quant_type in self._BITWIDTH_PRESETS:
            bitwidth, is_symmetric = self._BITWIDTH_PRESETS[quant_type]
            self.bitwidth = bitwidth
            self.is_symmetric = is_symmetric
            return
        if quant_type in {"hestia", "fairy_hestia", "ternary", "none"}:
            return
        raise ValueError(f"Unsupported quant_type: {quant_type}")

    def to_dict(self) -> dict:
        """Return a snapshot dict of all config fields (class + instance overrides)."""
        data = {}
        for name, value in self.__class__.__dict__.items():
            if name.startswith("_") or callable(value):
                continue
            data[name] = getattr(self, name)
        for name, value in self.__dict__.items():
            if name.startswith("_"):
                continue
            data[name] = value
        return data
    
    def to_str(self) -> str:
        """Return a string representation of the config."""
        data = self.to_dict()
        lines = []
        for k, v in data.items():
            lines.append(f"{k}: {v}")
        return "\n".join(lines)


def get_quant_config(
    quant_config_path: Optional[str] = None,
) -> QuantConfig:
    """
    Load QuantConfig from a YAML / JSON file.
    If path is None, return default QuantConfig.
    """
    config = QuantConfig()

    if quant_config_path is None:
        config.validate()
        return config
    
    path = Path(quant_config_path)
    if not path.exists():
        raise FileNotFoundError(f"Quant config file not found: {path}")
    
    if path.suffix == ".yaml":
        import yaml
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    elif path.suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        raise ValueError(f"Unsupported config format: {path.suffix}.")
    
    for key, value in data.items():
        if not hasattr(config, key):
            raise ValueError(f"Unknown QuantConfig field: {key}")
        setattr(config, key, value)
    
    config.validate()
    return config
