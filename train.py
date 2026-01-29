import math
import os
import pickle
from accelerate import Accelerator

import torch
from transformers import AutoModelForCausalLM
from torch.utils.data import DataLoader, Subset

from hestia.quant_config import get_quant_config
from hestia.model_utils import convert_model
from train_utils.data_loader import data_util

from train_utils.arguments import parse_arguments
from train_utils.callbacks import HestiaStatsCallback
from train_utils.training import get_train_args, train
from utils.logging import (
    setup_logger,
    log_progress,
    init_swanlab,
)
from utils.seed import setup_seed

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


def get_exe_mode(args) -> str:
    if args.dummy_data:
        return "dry_run_fast"
    if args.dry_run:
        return "dry_run"
    return "train"


def load_and_quantize_model(
    load_dir,
    qconfig,
    quant_type,
    skip_layers,
    accelerator,
    logger,
    temp_scales_dict=None,
):
    log_progress(accelerator, logger, "===== Model Loading ⏳ =====")

    model = AutoModelForCausalLM.from_pretrained(
        load_dir,
        trust_remote_code=True,
    )
    log_progress(accelerator, logger, f"Base Model Loaded: {type(model)}")

    if quant_type == "hestia" or quant_type == "fairy_hestia":
        log_progress(
            accelerator,
            logger,
            f"Quant Config:\n{qconfig.to_str()}",
        )
        # Log additional Hestia-specific parameters
        log_progress(
            accelerator,
            logger,
            f"Hestia Parameters:\n"
            f"  - kappa (sensitivity gain): {qconfig.kappa}\n"
            f"  - alpha (asynchrony coeff): {qconfig.alpha}\n"
            f"  - group_size: {qconfig.group_size}\n"
            f"  - codebook: {qconfig.codebook}\n"
            f"  - enable_hestia: {qconfig.enable_hestia}\n"
            f"  - calibration_granularity: {qconfig.calibration_granularity}\n"
            f"  - num_sketch: {qconfig.num_sketch}\n"
            f"  - num_query: {qconfig.num_query}\n"
            f"  - num_batches: {qconfig.num_batches}",
        )
    log_progress(accelerator, logger, f"Quant Mode: {quant_type}")

    quant_layers = convert_model(
        model=model,
        qconfig=qconfig,
        quant_type=quant_type,
        skip_layers=skip_layers,
        temp_scales_dict=temp_scales_dict,
    )

    log_progress(
        accelerator,
        logger,
        "===== Model Preparation Complete ✅ =====",
    )
    return model, quant_layers


def compute_training_scale(args, world_size):
    """
    Compute gradient_accumulation_steps and max_steps from
    global_batch_size / seq_len / max_tokens.
    """

    per_device_bs = args.per_device_train_batch_size
    global_bs = args.global_batch_size
    seq_len = args.seq_len

    effective_per_step = world_size * per_device_bs

    if global_bs % effective_per_step != 0:
        raise ValueError(
            f"GLOBAL_BATCH_SIZE ({global_bs}) must be divisible by "
            f"WORLD_SIZE ({world_size}) * PER_DEVICE_BS ({per_device_bs})"
        )

    grad_accum_steps = global_bs // effective_per_step
    tokens_per_step = global_bs * seq_len

    max_steps = None
    if args.max_tokens is not None and args.max_tokens > 0:
        max_steps = args.max_tokens // tokens_per_step

    return grad_accum_steps, max_steps




def load_temp_scales(traces_path: str):
    """
    Load pre-computed temperature scaling factors from a pickle file.

    Args:
        traces_path: Path to the pickle file containing calibration results

    Returns:
        Dictionary of {layer_id: temp_scale}
    """
    with open(traces_path, 'rb') as f:
        payload = pickle.load(f)

    if isinstance(payload, dict) and "temp_scales" in payload:
        return payload.get("temp_scales", {})

    raise ValueError(f"Unsupported Hessian payload type: {type(payload)}")


def main():
    # init args and env
    args = parse_arguments()
    accelerator = Accelerator()
    world_size = accelerator.num_processes

    # set logger
    logger = setup_logger(
        log_path=args.logging_path,
        rank=os.environ.get("RANK", "0")
    )
    log_progress(accelerator, logger, f"Accelerator initialized (world_size={world_size})")
    exe_mode = get_exe_mode(args)

    # set seed
    setup_seed(args.random_seed)

    # load tokenizer and data
    tokenizer, data_collator, train_dataset = data_util(
        accelerator=accelerator,
        logger=logger,
        load_dir=args.load_dir,
        data_path=args.data_dir,
        random_seed=args.random_seed,
        tokenizer_dir=args.tokenizer_dir,
    )

    # train
    grad_accum_steps, max_steps = compute_training_scale(
        args=args,
        world_size=world_size
    )
    
    # prepare model
    quant_config = get_quant_config(args.quant_config_path)

    # Load temp_scales before model conversion if hessian_traces_path is provided
    temp_scales_dict = {}
    if args.quant_type in {"hestia", "fairy_hestia"} and args.hessian_traces_path:
        temp_scales_dict = load_temp_scales(args.hessian_traces_path)
        log_progress(
            accelerator,
            logger,
            f"Loaded {len(temp_scales_dict)} pre-computed temperature scaling factors from {args.hessian_traces_path}",
        )

    model, quant_layers = load_and_quantize_model(
        load_dir=args.load_dir,
        qconfig=quant_config,
        quant_type=args.quant_type,
        skip_layers=args.skip_layers,
        accelerator=accelerator,
        logger=logger,
        temp_scales_dict=temp_scales_dict,
    )
    if getattr(args, "gradient_checkpointing", False):
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            log_progress(accelerator, logger, "Gradient checkpointing enabled.")
        else:
            log_progress(accelerator, logger, "Gradient checkpointing not supported by this model.")
    
    # only hestia has scheduler
    if args.quant_type in {"hestia", "fairy_hestia"}:
        for layer in quant_layers:
            layer.scheduler.bind_total_steps(max_steps)

    # prepare callbacks
    callbacks = []
    if args.quant_type in {"hestia", "fairy_hestia"}:
        callbacks.append(HestiaStatsCallback(quant_layers))
        if quant_config.enable_hestia and not args.hessian_traces_path:
            log_progress(
                accelerator,
                logger,
                "Warning: Hestia enabled but no hessian_traces_path provided. "
                "Using default temperature scaling (temp_scale=1.0). "
                "Run offline_calibration.py to pre-compute temperature scaling factors.",
            )

    # get training args
    training_args = get_train_args(
        args,
        grad_accum_steps,
        max_steps,
    )

    # init swanlab in main process
    if args.report_to == "swanlab" and accelerator.is_main_process:
        init_swanlab(
            args,
            training_args,
            quant_layers
        )

    log_progress(accelerator, logger, "Starting Training Loop...")
    log_progress(
        accelerator,
        logger,
        (   
            f"Run Mode: {exe_mode}\n"
            f"World size: {world_size}\n"
            f"Max Steps: {max_steps}\n"
            f"Max Tokens: {args.max_tokens}\n"
            f"Training Epochs: {args.num_train_epochs}\n"
            f"GBS: {args.global_batch_size}\n"
            f"Seq Len: {args.seq_len}\n"
            f"Per Device BS: {args.per_device_train_batch_size}\n"
            f"Accu Steps: {grad_accum_steps}\n"
            f"Num Parameters: {model.num_parameters()/1e6}M\n"
            f"Train Dataset: {train_dataset}"
        ),
    )

    train(
        model=model,
        training_args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        output_dir=args.output_dir,
        callbacks=callbacks,
        resume=args.resume,
        exe_mode=exe_mode,
    )


if __name__ == '__main__':
    main()
