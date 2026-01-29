import argparse
import json
import logging
from pathlib import Path

from eval_utils import default_results_dir, load_hestia_model, sanitize_filename, timestamp_tag


logger = logging.getLogger(__name__)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Load a Hestia-trained model and evaluate with lm-evaluation-harness.",
    )
    parser.add_argument("--model-dir", required=True, help="Path to trained model checkpoint.")
    parser.add_argument("--tokenizer-dir", default=None, help="Tokenizer path (defaults to --model-dir).")
    parser.add_argument(
        "--quant-type",
        choices=["hestia", "fairy_hestia", "ternary", "w2_sym", "w2_asym", "w4_sym", "w4_asym", "none"],
        default="none",
        help="Quantization type used during training.",
    )
    parser.add_argument("--quant-config-path", default=None, help="Quant config YAML/JSON path.")
    parser.add_argument(
        "--skip-layers",
        nargs="*",
        default=None,
        help="Module names to skip during quantization (e.g. lm_head).",
    )
    parser.add_argument(
        "--no-materialize-quantized",
        action="store_true",
        help="Skip materializing quantized weights and keep dynamic quantization in forward.",
    )
    parser.add_argument(
        "--tasks",
        default="arc_easy,arc_challenge,hellaswag,piqa,winogrande,gpqa_diamond_zeroshot",
        help="Comma-separated lm_eval task names, e.g. wikitext,piqa",
    )
    parser.add_argument("--batch-size", type=str, default="1", help="lm_eval batch size or 'auto' (defaults to 1).")
    parser.add_argument("--device", type=str, default="cuda", help="Device for lm_eval (e.g. cuda, cpu).")
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "fp32", "fp16", "bf16"],
        help="Model dtype for loading.",
    )
    parser.add_argument("--limit", type=float, default=None, help="Limit eval samples (int or float). Only use this for testing purposes.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for python's random module.")
    parser.add_argument("--numpy-seed", type=int, default=1234, help="Random seed for numpy.")
    parser.add_argument("--torch-seed", type=int, default=1234, help="Random seed for torch.")
    parser.add_argument("--fewshot-seed", type=int, default=1234, help="Random seed for fewshot sampler.")
    parser.add_argument("--output-filename", default=None, help="Optional path to write JSON results. Defaults to eval_results/<auto-name>.")
    return parser.parse_args()


def _build_default_output_path(args) -> Path:
    results_dir = default_results_dir()
    model_name = sanitize_filename(Path(args.model_dir).name)
    task_list = [t.strip() for t in args.tasks.split(",") if t.strip()]
    tasks = sanitize_filename("-".join(task_list))
    quant_type = sanitize_filename(args.quant_type)
    dtype = sanitize_filename(args.dtype)
    parts = [model_name, quant_type, dtype, timestamp_tag()]

    if args.quant_config_path:
        parts.append(f"qcfg-{sanitize_filename(Path(args.quant_config_path).stem)}")
    if args.no_materialize_quantized:
        parts.append("dyn")
    if tasks:
        parts.append(f"tasks-{tasks}")

    parts.append(f"bs{sanitize_filename(str(args.batch_size))}")
    parts.append(sanitize_filename(args.dtype))
    parts.append(sanitize_filename(args.device))

    if args.limit is not None:
        parts.append(f"limit{sanitize_filename(str(args.limit))}")

    if args.seed != 0 or args.numpy_seed != 1234 or args.torch_seed != 1234 or args.fewshot_seed != 1234:
        parts.append(f"seed{args.seed}_n{args.numpy_seed}_t{args.torch_seed}_f{args.fewshot_seed}")

    filename = "__".join(parts)
    return results_dir / f"{filename}.json"


def main():
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    try:
        from lm_eval import evaluator
        from lm_eval.models.huggingface import HFLM
        from lm_eval.utils import make_table
    except Exception as exc:
        raise RuntimeError(
            "lm-evaluation-harness is required. Install lm_eval before running."
        ) from exc

    model, tokenizer = load_hestia_model(
        model_dir=args.model_dir,
        tokenizer_dir=args.tokenizer_dir,
        quant_type=args.quant_type,
        quant_config_path=args.quant_config_path,
        skip_layers=args.skip_layers,
        dtype=args.dtype,
        device=args.device,
        materialize_quantized=not args.no_materialize_quantized,
    )

    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        device=args.device,
        batch_size=args.batch_size,
        trust_remote_code=True,
    )

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    results = evaluator.simple_evaluate(
        model=lm,
        tasks=tasks,
        batch_size=args.batch_size,
        device=args.device,
        limit=args.limit,
        random_seed=args.seed,
        numpy_random_seed=args.numpy_seed,
        torch_random_seed=args.torch_seed,
        fewshot_random_seed=args.fewshot_seed,
    )
    print("Evaluation Results:")
    print(make_table(results))

    payload = {
        "config": {
            "model_dir": args.model_dir,
            "tokenizer_dir": args.tokenizer_dir or args.model_dir,
            "quant_type": args.quant_type,
            "quant_config_path": args.quant_config_path,
            "tasks": tasks,
            "batch_size": args.batch_size,
            "device": args.device,
            "limit": args.limit,
            "dtype": args.dtype,
            "seed": args.seed,
            "numpy_seed": args.numpy_seed,
            "torch_seed": args.torch_seed,
            "fewshot_seed": args.fewshot_seed,
        },
        "results": results.get("results", {}),
        "versions": results.get("versions", {}),
    }

    print(json.dumps(payload, indent=2, sort_keys=True))
    output_path = Path(args.output_filename) if args.output_filename else _build_default_output_path(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    logger.info("Saved results to %s", output_path)


if __name__ == "__main__":
    main()
