# Hestia 🔥

**Hestia** is an anonymized implementation of a Hessian-guided differentiable
quantization-aware training (QAT) framework for extremely low-bit large language
models. Instead of using a hard round-and-clip quantizer in the forward pass and
an STE surrogate in the backward pass, Hestia trains with a temperature-controlled
Softmax expectation over the target codebook and anneals the relaxation back to
the final hard quantizer.

The framework is designed for low-bit LLM QAT, with offline Hutch++ Hessian-trace
calibration, tensor-wise adaptive annealing, distributed training through
`accelerate`, and optional SwanLab logging.

## ✨ Highlights

- 🔁 **Differentiable low-bit QAT.** Replaces hard STE quantization during training
  with a smooth expectation over the same discrete codebook.
- 🧭 **Dead-zone mitigation.** Boundary-adjacent latent weights receive nonzero
  update signals because the relaxed forward value changes continuously before
  annealing to the hard quantizer.
- 📐 **Hessian-guided annealing.** Offline Hessian traces provide lightweight
  tensor-wise sensitivity signals for adaptive temperature schedules.
- ⚙️ **Inference-compatible output.** The relaxed operator anneals to the target
  hard quantizer, so Hestia adds no inference-time overhead.
- 🚀 **LLM-ready training stack.** Supports grouped quantization, multi-GPU or
  multi-node training, checkpointing, evaluation scripts, and experiment logging.

## 🧩 Method Positioning

| Method | Differentiable relaxation | Curvature-aware | Tensor-wise adaptive annealing | Designed for extremely low-bit LLM QAT |
| --- | --- | --- | --- | --- |
| DSQ | ✓ | ✗ | ✗ | ✗ |
| HAWQ | ✗ | ✓ | ✗ | ✗ |
| CAGE | ✗ | ✓ | ✗ | △ |
| LOTION | ✓ | △ | ✗ | △ |
| Ours | ✓ | ✓ | ✓ | ✓ |

## 📁 Repository Layout

```text
Hestia/
├── configs/
│   ├── quant_example.yaml          # Example Hestia quantization config
│   └── hessian_calibration.yaml    # Example calibration config
├── env/
│   ├── requirements.txt            # Python dependencies
│   └── hestia.dockerfile           # Docker environment
├── eval/
│   └── eval_model.py               # Evaluation entry point
├── examples/
│   └── train_example.sh            # End-to-end training example
├── src/
│   ├── hestia/                     # Quantizer, scheduler, Hessian utilities
│   ├── train_utils/                # Data, arguments, callbacks, training loop
│   └── utils/                      # Logging and reproducibility utilities
├── offline_calibration.py          # Offline Hessian-trace calibration
└── train.py                        # Main QAT training script
```

## 🛠️ Installation

The codebase expects a recent CUDA-enabled PyTorch environment.

```bash
cd Hestia
pip install -r env/requirements.txt
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"
```

For distributed runs, configure `accelerate` once:

```bash
accelerate config
```

## 🚦 Quick Start

Hestia uses a two-stage workflow. First compute offline Hessian-trace sensitivity
statistics, then launch QAT using the saved temperature scales.

### 1. 📊 Offline Hessian Calibration

```bash
python offline_calibration.py \
  --model-path PATH_TO_MODEL \
  --data-dir PATH_TO_HF_DATASET \
  --quant-config-path configs/quant_example.yaml \
  --output-path hessian_traces.pkl \
  --calibrate-batch-size 1 \
  --max-seq-len 512
```

The output pickle contains Hessian traces, normalized sensitivity scores, and
temperature scales. A JSON companion file is also saved for inspection.

Useful options:

- `--num-sketch`: Hutch++ sketch dimension.
- `--num-query`: number of Hutchinson query vectors.
- `--num-batches`: number of calibration batches.
- `--calibration-granularity`: sensitivity granularity, such as tensor, layer, or component.
- `--skip-layers`: modules excluded from quantization, for example `lm_head`.

### 2. 🏋️ Quantization-Aware Training

Use the provided script as the recommended starting point. The script is written
for the parent directory layout, so run it from the directory that contains
`Hestia/`:

```bash
bash Hestia/examples/train_example.sh
```

Or launch directly with `accelerate` from inside `Hestia/`:

```bash
accelerate launch --config-file PATH_TO_ACCELERATE_CONFIG.yaml \
  train.py \
  --bf16 \
  --load-dir PATH_TO_MODEL \
  --tokenizer-dir PATH_TO_MODEL \
  --data-dir PATH_TO_HF_DATASET \
  --quant-type hestia \
  --quant-config-path configs/quant_example.yaml \
  --hessian-traces-path hessian_traces.pkl \
  --skip-layers lm_head \
  --global-batch-size 256 \
  --per-device-train-batch-size 16 \
  --seq-len 1024 \
  --max-tokens 8317664256 \
  --learning-rate 5e-5 \
  --output-dir PATH_TO_OUTPUT \
  --logging-path PATH_TO_LOG
```

If `--hessian-traces-path` is omitted, Hestia falls back to uniform temperature
scales. This is useful for debugging, but Hessian-guided annealing is the intended
setting.

## 🧪 Quantization Configuration

`configs/quant_example.yaml` controls the codebook, grouping, calibration budget,
and annealing schedule:

```yaml
num_sketch: 10
num_query: 20
num_batches: 5

codebook: [-1.0, 0.0, 1.0]
group_size: 0

compress_ratio: 0.2
anneal_ratio: 0.8
temp_decay_style: "cosine"
end_temp: 0.0

enable_hestia: True
```

Common fields:

- `codebook`: target quantization support, such as `[-1, 0, 1]` for ternary QAT.
- `group_size`: quantization group size. Use `0` for per-tensor, `-1` for per-channel,
  or a positive integer for block-wise grouping.
- `compress_ratio`: fraction of training used to introduce quantization pressure.
- `anneal_ratio`: fraction of training used for temperature annealing.
- `temp_decay_style`: temperature decay schedule.
- `enable_hestia`: whether to use Hessian-derived temperature scaling.

## 📈 Evaluation

```bash
python eval/eval_model.py \
  --model-dir PATH_TO_QUANTIZED_MODEL \
  --quant-type hestia \
  --quant-config-path configs/quant_example.yaml \
  --tasks arc_easy,arc_challenge,hellaswag,piqa,winogrande \
  --output-filename results.json
```

Task names depend on the local evaluation backend and configuration. See
`eval/eval_model.py` for the currently supported argument interface.

## 🌐 Distributed Training

Single-node and multi-node runs are both handled through `accelerate`.
The example script reads the following environment variables for multi-node
training:

```bash
export PET_NNODES=2
export PET_NODE_RANK=0
export PET_MASTER_ADDR=MASTER_NODE_IP
export PET_MASTER_PORT=23456
```

Then launch the same training script on each node with the appropriate
`PET_NODE_RANK`.

## 📝 Experiment Logging

SwanLab logging is optional. To enable it:

```bash
export SWANLAB_API_KEY=YOUR_KEY
export SWANLAB_MODE=cloud   # or local
```

The training loop logs optimization metrics, quantization pressure, temperature
schedules, and Hestia-specific quantization statistics.

## 🕶️ Notes for Anonymous Review

This repository is prepared for anonymous review. Please keep repository URLs,
author identities, institution names, and non-anonymized checkpoints out of
public-facing files until the review process is complete.

## 📚 Citation

An anonymized citation will be added after the review period.
