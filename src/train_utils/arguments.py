import argparse
import yaml
from typing import Dict, Any


def load_yaml_config(
    yaml_path: str
) -> Dict[str, Any]:
    try:
        with open(yaml_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as exc:
        raise ValueError(f"Cannot load {yaml_path}") from exc


def merge_args_with_config(
    args: argparse.Namespace,
    config: Dict[str, Any]
) -> argparse.Namespace:
    """
    Merge yaml and arguments (arguments is prior)
    """
    for key, value in config.items():
        if not hasattr(args, key) or getattr(args, key) is None:
            setattr(args, key, value)
    return args


def add_basic_arguments(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("Basic Arguments")
    group.add_argument("--random-seed", type=int, default=42, help="Random seed")
    group.add_argument("--dry-run", action="store_true", help="Test the pipeline without fwd and bwd pass")
    group.add_argument("--dummy-data", action="store_true", help="Use dummy data for test")


def add_quant_arguments(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("Quant Type")
    group.add_argument(
        "--quant-type",
        type=str,
        choices=[
            "hestia",
            "fairy_hestia",
            "ternary",
            "w2_sym",
            "w2_asym",
            "w4_sym",
            "w4_asym",
            "none",
        ],
        default="none", help="Quantization mode applied to linear layers",
    )
    group.add_argument("--quant-config-path", type=str, help="Path to quantization config yaml file")
    group.add_argument("--skip-layers", nargs="*", default=None, help="Names of modules to skip during quantization")
    group.add_argument("--hessian-traces-path", type=str, help="Path to pre-computed Hessian traces (pickle file)")


def add_path_arguments(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("Path Config")
    group.add_argument("--load-dir", type=str, help="Path of base model checkpoint to load for finetuning")
    group.add_argument("--tokenizer-dir", type=str, help="Optional tokenizer path (defaults to load_dir)")
    group.add_argument("--data-dir", type=str, help="Path to training dataset (HF load_from_disk)")
    group.add_argument("--config-path", type=str, help="Path to config yaml file")
    group.add_argument("--output-dir", type=str, help="Root path of output")


def add_training_arguments(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("Training Hyperparameter")
    group.add_argument("--global-batch-size", type=int, help="Global batch size across all devices")
    group.add_argument("--per-device-train-batch-size", type=int, help="Batch size per device/GPU")

    group.add_argument("--seq-len", type=int)
    group.add_argument("--max-tokens", type=int, help="Token budget")

    group.add_argument("--num-train-epochs", type=float, help="Total number of training epochs")
    group.add_argument("--max-steps", type=int, help="Set total number of training steps to perform")
    group.add_argument("--gradient-accumulation-steps", type=int, help="Number of update steps to accumulate before backward")

    group.add_argument("--learning-rate", type=float)
    group.add_argument("--weight-decay", type=float)
    group.add_argument("--max-grad-norm", type=float)
    group.add_argument("--adam-beta1", type=float)
    group.add_argument("--adam-beta2", type=float)

    group.add_argument("--bf16", action="store_true")
    group.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to reduce activation memory",
    )

    group.add_argument("--resume", action="store_true", help="Resume training from latest ckpt")


def add_scheduler_arguments(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("Scheduler Arguments")
    group.add_argument(
        "--lr-decay-style",
        type=str,
        choices=["wsd", "two_stage", "custom", "none"],
        default="wsd",
        help="Custom LR schedule to apply on top of optimizer",
    )
    group.add_argument("--warmup-ratio", type=float)
    group.add_argument("--wsd-ratio", type=float)
    group.add_argument("--min-lr-ratio", type=float)
    group.add_argument("--stage-ratio", type=float)


def add_log_arguments(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("Log Arguments")
    group.add_argument("--save-steps", type=int, help="Save interval")
    group.add_argument("--logging-steps", type=int, help="Log interval")
    group.add_argument("--save-total-limit", type=int, help="Max number of saved ckpts")
    group.add_argument("--save-strategy", type=str, default="steps", choices=["no", "epoch", "steps"])
    group.add_argument("--logging-path", type=str, help="Tensorboard log dir")
    group.add_argument("--report-to", type=str, default="swanlab", help="To use wandb or tensorboard, set this")
    group.add_argument("--model-name", type=str, help="Model name for swanlab logging")
    group.add_argument("--dataset-name", type=str, help="Dataset name for swanlab logging")
    group.add_argument("--date-str", type=str, help="Date string for swanlab logging")
    group.add_argument("--time-str", type=str, help="Time string for swanlab logging")
    group.add_argument("--swanlab-workspace", type=str, default="ComplexTrain", help="SwanLab workspace name")
    group.add_argument("--swanlab-project", type=str, default="Hestia-QAT", help="SwanLab project name")
    group.add_argument(
        "--swanlab-api-key",
        type=str,
        default=None,
        help="SwanLab API key (or set env SWANLAB_API_KEY)",
    )
    group.add_argument(
        "--swanlab-mode",
        type=str,
        default="cloud",
        choices=["cloud", "local"],
        help="SwanLab run mode; use local for offline environments",
    )
    group.add_argument(
        "--swanlab-logdir",
        type=str,
        default=None,
        help="SwanLab local log directory (use local disk like /tmp to avoid sqlite locking issues)",
    )


def add_distributed_arguments(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("Distributed Training Arguments")
    group.add_argument(
        "--main-process-ip",
        type=str,
        default="127.0.0.1",
        help="IP address of the main process (master) for multi-node training",
    )
    group.add_argument(
        "--main-process-port",
        type=int,
        default=29500,
        help="Port of the main process (master) for multi-node training",
    )
    group.add_argument(
        "--machine-rank",
        type=int,
        default=0,
        help="Rank of the current machine (node) in multi-node training (0 for master)",
    )
    group.add_argument(
        "--num-machines",
        type=int,
        default=1,
        help="Number of machines (nodes) for multi-node training",
    )


def validate_arguments(args):
    if args.dry_run and not args.dummy_data:
        import warnings
        warnings.warn("It is suggested to use --dummy-data in Dry-run mode")


def parse_arguments():
    parser = argparse.ArgumentParser()

    add_basic_arguments(parser)
    add_quant_arguments(parser)
    add_path_arguments(parser)
    add_training_arguments(parser)
    add_scheduler_arguments(parser)
    add_log_arguments(parser)
    add_distributed_arguments(parser)

    args, unknown = parser.parse_known_args()

    if args.config_path:
        config = load_yaml_config(args.config_path)
        args = merge_args_with_config(args, config)

    validate_arguments(args)
    return args


def format_arguments(args):
    return "\n".join([f"{k:25s} = {v}" for k, v in vars(args).items()])


if __name__ == '__main__':
    args = parse_arguments()
    print(format_arguments(args))
