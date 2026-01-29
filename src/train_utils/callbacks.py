import swanlab
from transformers.trainer_callback import TrainerCallback


class HestiaStatsCallback(TrainerCallback):
    """
    Update Hestia state each step and push stats to SwanLab.
    """
    def __init__(self, quant_layers):
        self.quant_layers = quant_layers

    def on_step_begin(self, args, state, control, **kwargs):
        for layer in self.quant_layers:
            layer.update_training_state(state.global_step)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not self.quant_layers:
            return

        if not state.is_world_process_zero:
            return

        log_dict = {}
        for quant_layer in self.quant_layers:
            stats = quant_layer.get_quantization_stats()
            temp = stats.get("temperature")
            layer_id = stats.get("layer_id")
            
            log_dict[f"hestia/{layer_id}_temp"] = temp

        # log pressure
        pressure = self.quant_layers[0].get_quantization_stats().get("pressure")
        log_dict[f"process/pressure"] = pressure


        # log only at the first step
        if state.global_step == 1:
            temp_scale_distribution = {}
            for i, layer in enumerate(self.quant_layers):
                if layer.temp_scale is not None:
                    temp_scale_distribution[f"layer_{i}_temp_scale"] = layer.temp_scale
            if temp_scale_distribution:
                log_dict.update(temp_scale_distribution)

        swanlab.log(log_dict, step=state.global_step)
