import math
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from typing import Callable, List, Dict, Tuple, Any, Optional
import warnings

from .hutchpp import HutchPlusPlusState


def _compute_hvp_single_vector(
    loss: torch.Tensor,
    param: torch.Tensor,
    vector: torch.Tensor,
) -> torch.Tensor:
    """
    Compute Hessian-vector product for a SINGLE vector using double backward.
    
    This is memory-efficient compared to is_grads_batched=True (vmap),
    which has huge memory overhead for second-order derivatives.
    
    H @ v = d/dp (grad @ v)
    
    Args:
        loss: Scalar loss value
        param: Parameter tensor to compute HVP for
        vector: Single probe vector [numel]
        
    Returns:
        HVP result [numel]
    """
    # First backward: compute gradient with graph retained
    grad = torch.autograd.grad(loss, param, create_graph=True, retain_graph=True)[0]
    grad_flat = grad.view(-1)
    
    # Compute grad @ v (scalar)
    grad_v = torch.dot(grad_flat, vector)
    
    # Second backward: d/dp (grad @ v) = H @ v
    hvp = torch.autograd.grad(grad_v, param, retain_graph=True)[0]
    
    return hvp.view(-1)


class HessianTraceCalibrator:
    """
    Calibrator for estimating and injecting Hessian Trace into target layers.
    
    Memory-optimized implementation:
    - Uses per-vector HVP computation instead of batched vmap (huge memory savings)
    - Supports gradient checkpointing for forward pass
    - Processes layers one at a time with independent forward passes
    """

    def __init__(
        self,
        model: nn.Module,
        loss_func: Callable,
        dataloader: Any,
        device: str = 'cuda',
        use_gradient_checkpointing: bool = True,
        max_seq_len: Optional[int] = None,
    ) -> None:
        self.model = model
        self.loss_func = loss_func
        self.dataloader = dataloader
        self.device = device
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.max_seq_len = max_seq_len  # Truncate sequences if set
        
        # Enable gradient checkpointing if requested
        if use_gradient_checkpointing:
            self._enable_gradient_checkpointing()

    def _enable_gradient_checkpointing(self):
        """Enable gradient checkpointing on supported model architectures."""
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
            print("  [Memory] Gradient checkpointing enabled")
        elif hasattr(self.model, 'enable_input_require_grads'):
            # Some models need this for checkpointing to work
            self.model.enable_input_require_grads()
            
    def _disable_gradient_checkpointing(self):
        """Disable gradient checkpointing."""
        if hasattr(self.model, 'gradient_checkpointing_disable'):
            self.model.gradient_checkpointing_disable()

    def calibrate(
        self,
        target_modules: List[type] = [nn.Linear],
        num_sketch: int = 10,
        num_query: int = 20,
        num_batches: int = 5,
        granularity: str = "layer"
    ) -> Dict[str, Dict[str, float]]:
        """
        Calibrate Hessian Trace for specified layers, compute temperature scaling factors,
        and inject values into the layer's `temp_scale` and `hessian_score` attributes.

        Args:
            target_modules: List of target layer types
            num_sketch: Sketch dimension for Hutch++
            num_query: Number of random samples
            num_batches: Number of batches for estimation
            granularity: Calibration granularity - "tensor", "layer", or "component"

        Returns:
            Dictionary with:
                - "traces": {layer_id: hessian_trace}
                - "temp_scales": {layer_id: temperature_scaling_factor}
                - "scores": {layer_id: sensitivity_score}
                - "stats": {"log_mean": float, "log_std": float}
        """
        # Register target layers and initialize states based on granularity
        layer_states: Dict[nn.Module, HutchPlusPlusState] = {}
        layer_params: Dict[nn.Module, torch.Tensor] = {}

        # Group layers by granularity
        if granularity == "tensor":
            # Per-tensor: Each Linear layer has its own trace
            for name, module in self.model.named_modules():
                if isinstance(module, tuple(target_modules)):
                    if hasattr(module, 'weight') and module.weight is not None:
                        if module.weight.requires_grad:
                            layer_states[module] = HutchPlusPlusState(
                                module.weight.numel(), num_sketch, num_query, self.device
                            )
                            layer_params[module] = module.weight
                        else:
                            warnings.warn(f"Module {name} weight requires_grad=False, skipping.")
                    else:
                        warnings.warn(f"Module {name} has no 'weight' attribute, skipping.")

        elif granularity == "layer":
            # Per-layer: All Linear layers in the same transformer layer share one trace
            # Group by layer_id prefix (e.g., "layer_0" from "layer_0_model.layers.0.self_attn.q_proj")
            layer_groups: Dict[str, List[nn.Module]] = {}
            for name, module in self.model.named_modules():
                if isinstance(module, tuple(target_modules)):
                    if hasattr(module, 'weight') and module.weight is not None:
                        if module.weight.requires_grad:
                            # Extract layer prefix from layer_id
                            if hasattr(module, 'layer_id') and module.layer_id:
                                # layer_id format: "layer_{counter}_{full_path}"
                                # Extract "layer_{counter}" as group key
                                parts = module.layer_id.split('_')
                                if len(parts) >= 2:
                                    group_key = parts[0] + '_' + parts[1]
                                    if group_key not in layer_groups:
                                        layer_groups[group_key] = []
                                    layer_groups[group_key].append(module)
                            else:
                                warnings.warn(f"Module {name} has no layer_id, skipping.")
                        else:
                            warnings.warn(f"Module {name} weight requires_grad=False, skipping.")
                    else:
                        warnings.warn(f"Module {name} has no 'weight' attribute, skipping.")

            # For each layer group, use the first layer's weight for calibration
            # The trace will be shared across all layers in the group
            for group_key, modules in layer_groups.items():
                # Use the first module's weight for calibration
                first_module = modules[0]
                layer_states[first_module] = HutchPlusPlusState(
                    first_module.weight.numel(), num_sketch, num_query, self.device
                )
                layer_params[first_module] = first_module.weight
                # Store group info for trace sharing
                layer_states[first_module]._group_modules = modules

        elif granularity == "component":
            # Per-component: All layers of the same component type share one trace
            # e.g., all q_proj share one trace, all k_proj share another
            component_groups: Dict[str, List[nn.Module]] = {}
            for name, module in self.model.named_modules():
                if isinstance(module, tuple(target_modules)):
                    if hasattr(module, 'weight') and module.weight is not None:
                        if module.weight.requires_grad:
                            # Extract component name from full path
                            # e.g., "layer_0_model.layers.0.self_attn.q_proj" -> "q_proj"
                            if hasattr(module, 'layer_id') and module.layer_id:
                                parts = module.layer_id.split('_')
                                if len(parts) >= 3:
                                    full_path = '_'.join(parts[2:])
                                    component_name = full_path.split('.')[-1]  # Get last part (e.g., "q_proj")
                                    if component_name not in component_groups:
                                        component_groups[component_name] = []
                                    component_groups[component_name].append(module)
                            else:
                                warnings.warn(f"Module {name} has no layer_id, skipping.")
                        else:
                            warnings.warn(f"Module {name} weight requires_grad=False, skipping.")
                    else:
                        warnings.warn(f"Module {name} has no 'weight' attribute, skipping.")

            # For each component group, use the first layer's weight for calibration
            # The trace will be shared across all layers of this component type
            for component_name, modules in component_groups.items():
                # Use the first module's weight for calibration
                first_module = modules[0]
                layer_states[first_module] = HutchPlusPlusState(
                    first_module.weight.numel(), num_sketch, num_query, self.device
                )
                layer_params[first_module] = first_module.weight
                # Store group info for trace sharing
                layer_states[first_module]._group_modules = modules
                layer_states[first_module]._component_name = component_name

        else:
            raise ValueError(f"Unknown calibration granularity: {granularity}")

        print(f"Target layers: {len(layer_states)} (granularity={granularity}). Batches per phase: {num_batches}")
        if len(layer_states) == 0:
            return {"traces": {}, "scores": {}, "stats": {}}

        # stage1: randomized low-rank approximation
        current_vectors = {}
        for mod, state in layer_states.items():
            current_vectors[mod] = state.init_phase1_sketch()

        self._run_global_hvp_loop(layer_params, current_vectors, layer_states, "phase1", num_batches)

        for state in layer_states.values():
            state.finalize_phase1(num_batches)

        # stage2: subspace trace estimation
        current_vectors = {}
        for mod, state in layer_states.items():
            current_vectors[mod] = state.init_phase2_subspace()

        self._run_global_hvp_loop(layer_params, current_vectors, layer_states, "phase2", num_batches)

        finished_modules = []
        for mod, state in layer_states.items():
            is_done = state.finalize_phase2(num_batches)
            if is_done: finished_modules.append(mod)

        module_traces: Dict[nn.Module, float] = {}

        for mod in finished_modules:
            state = layer_states[mod]
            trace_value = state.final_trace

            # For per-layer and per-component: Share trace with all modules in the group
            if hasattr(state, '_group_modules'):
                for group_mod in state._group_modules:
                    module_traces[group_mod] = trace_value
            module_traces[mod] = trace_value

            del current_vectors[mod]
            del layer_params[mod]
            del layer_states[mod]

        if len(layer_states) > 0:
            # stage3: residual estimation
            current_vectors = {}
            for mod, state in layer_states.items():
                current_vectors[mod] = state.init_phase3_residual()

            self._run_global_hvp_loop(layer_params, current_vectors, layer_states, "phase3", num_batches)

            for mod, state in layer_states.items():
                state.finalize_phase3(num_batches)
                trace_value = state.final_trace

                # For per-layer and per-component: Share trace with all modules in the group
                if hasattr(state, '_group_modules'):
                    for group_mod in state._group_modules:
                        module_traces[group_mod] = trace_value
                module_traces[mod] = trace_value

        temp_scales, scores, stats = self._compute_temperature_scaling_factors(module_traces)

        traces_by_layer_id: Dict[str, float] = {}
        temp_scales_by_layer_id: Dict[str, float] = {}
        scores_by_layer_id: Dict[str, float] = {}
        for mod, trace_value in module_traces.items():
            layer_id = getattr(mod, "layer_id", None)
            if layer_id is None:
                continue
            traces_by_layer_id[layer_id] = trace_value

        for mod, temp_scale in temp_scales.items():
            mod.temp_scale = temp_scale
            layer_id = getattr(mod, "layer_id", None)
            if layer_id is None:
                continue
            temp_scales_by_layer_id[layer_id] = temp_scale

        for mod, score in scores.items():
            mod.hessian_score = score
            layer_id = getattr(mod, "layer_id", None)
            if layer_id is None:
                continue
            scores_by_layer_id[layer_id] = score

        return {
            "traces": traces_by_layer_id,
            "temp_scales": temp_scales_by_layer_id,
            "scores": scores_by_layer_id,
            "stats": stats,
        }

    def _compute_temperature_scaling_factors(
        self,
        traces: Dict[nn.Module, float],
        eps: float = 1e-8,
    ) -> Tuple[Dict[nn.Module, float], Dict[nn.Module, float], Dict[str, float]]:
        """
        Compute temperature scaling factors from Hessian traces.

        Formula:
            s_i = Sigmoid(κ * (log(χ_i) - μ_χ) / (σ_χ + ε))
            temp_scale_i = exp(-α * s_i)

        where:
            - χ_i = Tr(H_i) (Hessian trace)
            - κ: Gain factor (from qconfig)
            - α: Asynchrony coefficient (from qconfig)
            - μ_χ, σ_χ: Mean and std of log(traces)

        Returns:
            Tuple of (temp_scales, scores, stats)
        """
        if not traces:
            return {}, {}, {}

        log_traces = [math.log(max(value, eps)) for value in traces.values()]
        log_mean = sum(log_traces) / len(log_traces)
        log_var = sum((x - log_mean) ** 2 for x in log_traces) / len(log_traces)
        log_std = math.sqrt(log_var) if log_var > 0 else 0.0

        temp_scales: Dict[nn.Module, float] = {}
        scores: Dict[nn.Module, float] = {}
        for mod, trace_value in traces.items():
            kappa = getattr(getattr(mod, "qconfig", None), "kappa", 1.0)
            alpha = getattr(getattr(mod, "qconfig", None), "alpha", 0.5)
            z_i = (math.log(max(trace_value, eps)) - log_mean) / (log_std + eps)
            sensitivity_score = 1.0 / (1.0 + math.exp(-kappa * z_i))
            temp_scale = math.exp(-alpha * sensitivity_score)
            temp_scales[mod] = temp_scale
            scores[mod] = sensitivity_score

        stats = {"log_mean": log_mean, "log_std": log_std}
        return temp_scales, scores, stats

    def _run_global_hvp_loop(
        self,
        params_map: Dict[nn.Module, torch.Tensor],
        vectors_map: Dict[nn.Module, torch.Tensor],
        states_map: Dict[nn.Module, HutchPlusPlusState],
        phase: str,
        num_batches: int
    ) -> None:
        """
        Core engine: Iterate through DataLoader and compute HVP in batches.

        Memory-optimized version using PER-VECTOR HVP computation:
        - Avoids vmap (is_grads_batched=True) which has huge memory overhead
        - Computes HVP for each probe vector separately
        - Releases computation graph after each vector
        
        This is slower but uses much less memory (critical for large models).
        """
        if not params_map: return

        modules_list = list(params_map.keys())
        num_layers = len(modules_list)
        
        # Cache batch data (on CPU to save GPU memory)
        print(f"  Phase {phase}: Caching {num_batches} batches...")
        cached_batches = []
        for batch_idx, batch in enumerate(self.dataloader):
            if batch_idx >= num_batches: break
            # Keep on CPU, move to GPU only when needed
            cached_batches.append(batch)
        
        actual_batches = len(cached_batches)
        
        # Get number of vectors for this phase
        sample_vectors = list(vectors_map.values())[0]
        k = sample_vectors.shape[1]  # [numel, k]
        
        print(f"  Phase {phase}: {num_layers} layers × {actual_batches} batches × {k} vectors")
        
        # Process each layer
        for layer_idx, module in enumerate(modules_list):
            param = params_map[module]
            vectors_cpu = vectors_map[module]  # [numel, k]
            numel = vectors_cpu.shape[0]
            
            # Accumulator for this layer's HVP results
            hvp_accum = torch.zeros((numel, k), device='cpu', dtype=torch.float32)
            
            for batch_idx, batch in enumerate(cached_batches):
                # Move batch to GPU
                batch_gpu = self._move_to_device(batch)
                inputs, targets = self._unpack_batch(batch_gpu)
                
                # Truncate sequence if max_seq_len is set (saves memory)
                if self.max_seq_len is not None:
                    inputs, targets = self._truncate_sequence(inputs, targets)
                
                # Forward pass
                outputs = self.model(**inputs) if isinstance(inputs, dict) else self.model(inputs)
                loss = self.loss_func(outputs, targets)
                
                # Compute HVP for EACH vector separately (memory efficient)
                for vec_idx in range(k):
                    # Get single probe vector and move to GPU
                    vec = vectors_cpu[:, vec_idx].to(self.device)
                    
                    # Compute HVP for this single vector
                    hvp = _compute_hvp_single_vector(loss, param, vec)
                    
                    # Accumulate on CPU
                    hvp_accum[:, vec_idx] += hvp.cpu()
                    
                    # Clean up this vector's computation
                    del vec, hvp
                
                # Clean up batch computation graph
                del outputs, loss, batch_gpu, inputs, targets
                torch.cuda.empty_cache()
            
            # Pass accumulated HVP to state (shape: [numel, k])
            if phase == "phase1": states_map[module].accumulate_phase1(hvp_accum)
            elif phase == "phase2": states_map[module].accumulate_phase2(hvp_accum)
            elif phase == "phase3": states_map[module].accumulate_phase3(hvp_accum)
            
            del hvp_accum
            torch.cuda.empty_cache()
            print(f"    Layer {layer_idx + 1}/{num_layers} done", end='\r')

        # Clean up
        del cached_batches
        torch.cuda.empty_cache()
        print(f"    Phase {phase} complete{' ' * 30}")
    
    def _truncate_sequence(
        self,
        inputs: Dict[str, torch.Tensor],
        targets: Optional[torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], Optional[torch.Tensor]]:
        """Truncate input sequences to max_seq_len to reduce memory usage."""
        if self.max_seq_len is None:
            return inputs, targets
            
        truncated_inputs = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor) and v.dim() >= 2:
                truncated_inputs[k] = v[:, :self.max_seq_len]
            else:
                truncated_inputs[k] = v
                
        if targets is not None and isinstance(targets, torch.Tensor) and targets.dim() >= 1:
            targets = targets[:, :self.max_seq_len] if targets.dim() >= 2 else targets
            
        return truncated_inputs, targets

    def _unpack_batch(
        self,
        data_batch: Any,
    ) -> Tuple[Any, Any]:
        """
        Unpack data batch, supporting multiple formats.

        Args:
            data_batch: Data batch, could be dict, BatchEncoding, or (input, target) tuple

        Returns:
            Tuple[Any, Any]: inputs, targets
        """
        # Handle dict-like objects (including BatchEncoding)
        if hasattr(data_batch, 'keys') and hasattr(data_batch, 'get'):
            inputs = {k: v for k, v in data_batch.items() if k != 'labels'}
            targets = data_batch.get('labels')
        elif isinstance(data_batch, (list, tuple)):
            inputs, targets = data_batch
        else:
            raise ValueError(f"Unsupported batch type: {type(data_batch)}")
        return inputs, targets

    def _move_to_device(
        self,
        data_batch: Any,
    ) -> Any:
        """
        Move data batch to target device.

        Args:
            data_batch: Data batch

        Returns:
            Any: Moved data batch
        """
        if isinstance(data_batch, torch.Tensor): return data_batch.to(self.device)
        elif isinstance(data_batch, dict): return {k: self._move_to_device(v) for k, v in data_batch.items()}
        elif isinstance(data_batch, (list, tuple)): return type(data_batch)(self._move_to_device(v) for v in data_batch)
        elif hasattr(data_batch, 'keys') and hasattr(data_batch, 'to'):
            # Handle BatchEncoding and similar dict-like objects with .to() method
            return data_batch.to(self.device)
        return data_batch
