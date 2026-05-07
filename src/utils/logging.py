# utils/logging.py
import logging
import swanlab
import os


def setup_logger(
    log_path: str ="logs",
    rank: str = "0",
):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()]
    )
    logger = logging.getLogger(__name__)

    if rank == "0" and not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        # Create log directory if it doesn't exist
        log_dir = os.path.dirname(log_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(file_handler)

    return logger


def log_progress(
    accelerator,
    logger,
    msg: str,
    *args,
    main_only: bool = True
):
    if main_only and not accelerator.is_main_process:
        return None

    worker_id = os.environ.get("RANK", "0")
    logger.info(f"[worker {worker_id}] {msg}", *args)
    return None


def init_swanlab(
    args,
    training_args,
    quant_layers,
):
    cfg = training_args.to_dict()
    cfg["dataset"] = args.dataset_name
    
    if args.quant_type in {"hestia", "fairy_hestia"}:
        cfg["model_type"] = f"Hestia-{args.model_name}"
        cfg["num_hestia_layers"] = len(quant_layers)
        qcfg = quant_layers[0].qconfig
        if hasattr(qcfg, "to_dict"):
            cfg["hestia_qconfig"] = qcfg.to_dict()
        else:
            # Copy and convert string values to numeric for SwanLab
            qcfg_dict = qcfg.__dict__.copy()
            # Convert temp_decay_style to numeric
            if "temp_decay_style" in qcfg_dict:
                style_map = {"cosine": 0, "linear": 1, "constant": 2}
                qcfg_dict["temp_decay_style_numeric"] = style_map.get(qcfg_dict["temp_decay_style"], -1)
                del qcfg_dict["temp_decay_style"]
            cfg["hestia_qconfig"] = qcfg_dict
    elif args.quant_type == "ternary":
        cfg["model_type"] = f"Ternary-{args.model_name}"
        cfg["num_quant_layers"] = len(quant_layers)
    elif args.quant_type in {"w2_sym", "w2_asym", "w4_sym", "w4_asym"}:
        cfg["model_type"] = f"{args.quant_type}-{args.model_name}"
        cfg["num_quant_layers"] = len(quant_layers)
        if quant_layers:
            qcfg = quant_layers[0].qconfig
            cfg["bitwidth_qconfig"] = (
                qcfg.to_dict() if hasattr(qcfg, "to_dict") else qcfg.__dict__.copy()
            )
    else:
        cfg["model_type"] = f"Baseline-{args.model_name}"

    run_name = (
        f"{args.date_str}{args.time_str}_"
        f"{args.quant_type}_"
        f"lr-{args.learning_rate}_"
        f"{args.model_name}_"
        f"globalbs{args.global_batch_size}"
    )

    api_key = args.swanlab_api_key or os.environ.get("SWANLAB_API_KEY")
    mode = (args.swanlab_mode or os.environ.get("SWANLAB_MODE") or "cloud").lower()
    logdir = (
        args.swanlab_logdir
        or os.environ.get("SWANLAB_LOGDIR")
        or "/tmp/swanlab"
    )

    init_kwargs = dict(
        workspace=args.swanlab_workspace,
        project=args.swanlab_project,
        name=run_name,
        config=cfg,
        logdir=logdir,
    )

    if mode == "local":
        init_kwargs["mode"] = "local"
        swanlab.init(**init_kwargs)
        print(f"[WARNING] SwanLab running in local mode. logdir={logdir}")
        return

    try:
        if api_key:
            # Explicit login for no-tty environments
            swanlab.login(api_key=api_key)
        swanlab.init(**init_kwargs)
    except Exception as exc:
        err_msg = str(exc).lower()
        # Fallback to local mode when api key is missing or cloud is unreachable
        if (
            ("api key not configured" in err_msg and not api_key)
            or ("connect timeout" in err_msg or "connection to" in err_msg)
        ):
            init_kwargs["mode"] = "local"
            swanlab.init(**init_kwargs)
            print("[WARNING] SwanLab cloud unavailable, switched to local mode.")
        else:
            raise
