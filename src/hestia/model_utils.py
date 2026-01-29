#!/usr/bin/env python3
"""
Model conversion utilities for Hestia quantization and ternary quantization.
"""

import logging
from typing import Any, List, Optional

import torch
import torch.nn as nn

from hestia.quant_config import QuantConfig
from hestia.quant_linear import HestiaLinear, TernaryLinear, BitwidthLinear, FairyHestiaLinear


logger = logging.getLogger(__name__)


def _replace_linear_layer(
    module: nn.Module,
    name: str,
    child: nn.Linear,
    quant_class: type,
    qconfig: Optional[Any] = None,
    layer_id: Optional[str] = None,
    temp_scale: Optional[float] = None,
) -> nn.Module:
    """
    Replace a single Linear layer with quantized version.

    Args:
        module: Parent module containing the layer
        name: Name of the child module
        child: Linear layer to replace
        quant_class: Quantization layer class (HestiaLinear or TernaryLinear)
        qconfig: Quantization config
        layer_id: Unique identifier for the layer (for Hestia)
        temp_scale: Pre-computed temperature scale factor (for Hestia)

    Returns:
        New quantized linear layer
    """
    # Build kwargs based on quant_class
    kwargs = {
        'in_features': child.in_features,
        'out_features': child.out_features,
        'bias': child.bias is not None,
        'qconfig': qconfig,
    }

    # Add layer_id only for Hestia variants
    if quant_class.__name__ in {'HestiaLinear', 'FairyHestiaLinear'}:
        kwargs['layer_id'] = layer_id
        kwargs['temp_scale'] = temp_scale

    quantized_layer = quant_class(**kwargs)

    # Reuse existing parameters to preserve ZeRO-3 metadata and avoid empty clones.
    quantized_layer.weight = child.weight
    if child.bias is not None:
        quantized_layer.bias = child.bias
    else:
        quantized_layer.bias = None
    # Ensure any quantizer buffers are aligned with the reused parameter device/dtype.
    if hasattr(quantized_layer, "_ensure_quantizer_device"):
        quantized_layer._ensure_quantizer_device()
    setattr(module, name, quantized_layer)  # Replace in parent module
    return quantized_layer


def _convert_model_recursively(
    model: nn.Module,
    quant_class: type,
    qconfig: Optional[Any] = None,
    skip_layers: Optional[List[str]] = None,
    temp_scales_dict: Optional[dict] = None,
) -> List[nn.Module]:
    """
    Recursively convert Linear layers in model.

    Args:
        model: Model to convert (modified in-place)
        quant_class: Quantization layer class
        qconfig: Quantization config
        skip_layers: List of layer names to skip (default: ['lm_head'])

    Returns:
        List of converted quantized linear layers
    """
    if skip_layers is None:
        skip_layers = ['lm_head']

    quant_layers = []
    layer_counter = [0]  # Use list to allow modification in nested function

    def _convert(module: nn.Module, prefix: str = ""):
        for name, child in list(module.named_children()):
            if name in skip_layers:
                continue

            # Build full layer path for unique ID
            full_path = f"{prefix}.{name}" if prefix else name

            # Replace Linear layers
            if isinstance(child, nn.Linear):
                if quant_class.__name__ == 'FairyHestiaLinear':
                    if child.in_features % 2 != 0 or child.out_features % 2 != 0:
                        logger.warning(
                            "Skipping FairyHestiaLinear replacement for %s (non-even dims: %s, %s)",
                            full_path,
                            child.in_features,
                            child.out_features,
                        )
                        continue
                layer_id = f"layer_{layer_counter[0]}_{full_path}"
                layer_counter[0] += 1
                temp_scale = temp_scales_dict.get(layer_id) if temp_scales_dict else None
                q_layer = _replace_linear_layer(
                    module, name, child, quant_class, qconfig, layer_id, temp_scale
                )
                quant_layers.append(q_layer)
            else:
                # Recurse into child modules
                _convert(child, full_path)

    _convert(model)
    return quant_layers


def convert_model(
    model: nn.Module,
    qconfig: Optional[Any],
    quant_type: str,
    skip_layers: Optional[List[str]] = None,
    temp_scales_dict: Optional[dict] = None,
) -> List[nn.Module]:
    """
    Convert model to Quantized model.

    Recursively replaces all Linear layers with QuantLinear, skipping specified layers.

    Args:
        model: PyTorch model (modified in-place)
        qconfig: quant config
        quant_type: Quant type to quantize linear layer
        skip_layers: List of layer names to skip (default: ['lm_head'])

    Returns:
        List of converted layers
    """
    if quant_type == "hestia":
        quant_layers = _convert_model_recursively(
            model=model,
            quant_class=HestiaLinear,
            qconfig=qconfig,
            skip_layers=skip_layers,
            temp_scales_dict=temp_scales_dict,
        )
    elif quant_type == "fairy_hestia":
        quant_layers = _convert_model_recursively(
            model=model,
            quant_class=FairyHestiaLinear,
            qconfig=qconfig,
            skip_layers=skip_layers,
            temp_scales_dict=temp_scales_dict,
        )
    elif quant_type == "ternary":
        quant_layers = _convert_model_recursively(
            model=model,
            quant_class=TernaryLinear,
            qconfig=qconfig,
            skip_layers=skip_layers,
        )
    elif quant_type in {"w2_sym", "w2_asym", "w4_sym", "w4_asym"}:
        if qconfig is None:
            qconfig = QuantConfig()
        qconfig.apply_quant_type(quant_type)
        quant_layers = _convert_model_recursively(
            model=model,
            quant_class=BitwidthLinear,
            qconfig=qconfig,
            skip_layers=skip_layers,
        )
    elif quant_type == "none":
        quant_layers = []
    else:
        raise ValueError(f"Quant type {quant_type} is not supported!")

    logger.info(f"Converted Linear layers to {quant_type} QuantLinear")
    return quant_layers
