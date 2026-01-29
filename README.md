# Hestia

Hestia is a quantization training framework based on Hessian trace information, implementing the **Hessian Trace Informed Softmax Annealing (Hestia)** method for efficient quantization training of large language models.

## Features

- 🔥 **Hestia Quantization Method**: Softmax annealing quantization based on Hessian trace information
- 📊 **Offline Calibration**: Supports offline Hessian trace calibration using the Hutch++ algorithm
- 🚀 **Distributed Training**: Multi-node multi-GPU training support based on the Accelerate framework
- 📈 **Three-Phase Training**: Compress, Anneal, and Solid phases
- 🎯 **Flexible Quantization Configuration**: Supports multiple quantization granularities (tensor, layer, component)
- 📝 **Experiment Tracking**: Integrated with SwanLab for training process visualization

## Directory Structure

```
Hestia/
├── configs/                    # Configuration files
│   ├── quant_example.yaml     # Quantization config example
│   └── hessian_calibration.yaml  # Hessian calibration config
├── env/                        # Environment configuration
│   ├── requirements.txt       # Python dependencies
│   └── hestia.dockerfile      # Docker image configuration
├── examples/                   # Example scripts
│   └── train_example.sh       # Training example script
├── eval/                       # Evaluation scripts
│   └── eval_model.py          # Model evaluation
├── src/                        # Source code
│   ├── hestia/                # Hestia core modules
│   │   ├── hessian/           # Hessian-related utilities
│   │   ├── quant_linear.py    # Quantized linear layer
│   │   ├── thermo_quantizer.py # Thermodynamic quantizer
│   │   ├── thermo_scheduler.py # Thermodynamic scheduler
│   │   └── quant_config.py    # Quantization configuration
│   ├── train_utils/           # Training utilities
│   └── utils/                 # Utility functions
├── offline_calibration.py     # Offline calibration script
└── train.py                   # Main training script
```

## Installation

### Requirements

- Python >= 3.8
- CUDA >= 11.8
- PyTorch >= 2.0

### Installation Steps

1. Clone the repository:
```bash
git clone <repository-url>
cd Hestia
```

2. Install dependencies:
```bash
pip install -r env/requirements.txt
```

3. Configure Accelerate:
```bash
accelerate config
```

## Quick Start

### 1. Offline Hessian Calibration (Optional but Recommended)

Before training, it is recommended to run offline calibration to compute temperature scaling factors:

```bash
python offline_calibration.py \
    --model-path PATH_TO_MODEL \
    --data-dir PATH_TO_DATA \
    --quant-config-path configs/quant_example.yaml \
    --output-path hessian_traces.pkl \
    --calibrate-batch-size 1 \
    --max-seq-len 512
```

**Parameter Description:**
- `--model-path`: Path to the pretrained model
- `--data-dir`: Path to calibration dataset (HuggingFace datasets format)
- `--quant-config-path`: Path to quantization configuration file
- `--output-path`: Output path for Hessian traces file
- `--calibrate-batch-size`: Batch size for calibration (default: 1)
- `--max-seq-len`: Maximum sequence length (default: 512, for memory efficiency)

### 2. Training

Use the provided example script for training:

```bash
cd examples
bash train_example.sh
```

Or use Python command directly:

```bash
accelerate launch --config-file PATH_TO_ACCELERATE_CONFIG.yaml \
    train.py \
    --bf16 \
    --load-dir PATH_TO_MODEL \
    --quant-type hestia \
    --quant-config-path configs/quant_example.yaml \
    --hessian-traces-path hessian_traces.pkl \
    --skip-layers lm_head \
    --global-batch-size 256 \
    --per-device-train-batch-size 16 \
    --seq-len 1024 \
    --max-tokens 8317664256 \
    --learning-rate 5e-5 \
    --data-dir PATH_TO_DATA \
    --output-dir PATH_TO_SAVE_DIR \
    --logging-path PATH_TO_LOG
```

## Configuration

### Quantization Configuration File

Quantization configuration is specified via YAML file with the following main parameters:

```yaml
# Hessian calibration parameters
num_sketch: 10              # Hutch++ sketch dimension
num_query: 20               # Number of random samples
num_batches: 5              # Number of batches for estimation
calibration_granularity: "tensor"  # Calibration granularity: tensor/layer/component

# Quantization codebook
codebook: [-1.0, 0.0, 1.0]  # Set of quantization values

# Group-wise quantization
group_size: 0               # 0=per-tensor, -1=per-channel, >0=block-wise

# Training phase parameters
compress_ratio: 0.2         # Compression phase ratio
anneal_ratio: 0.8           # Annealing phase ratio
temp_decay_style: "cosine"  # Temperature decay style: linear/cosine/hessian
end_temp: 0.0               # Final temperature

# Hestia parameters
enable_hestia: True         # Enable Hestia method
```

### Training Parameters

Main training parameters include:

- **Model Parameters**:
  - `--load-dir`: Model loading path
  - `--tokenizer-dir`: Tokenizer path
  - `--bf16`: Use bfloat16 precision

- **Quantization Parameters**:
  - `--quant-type`: Quantization type (`hestia` or `fairy_hestia`)
  - `--quant-config-path`: Path to quantization configuration file
  - `--hessian-traces-path`: Path to Hessian traces file (offline calibration results)
  - `--skip-layers`: Layers to skip quantization (e.g., `lm_head`)

- **Training Parameters**:
  - `--global-batch-size`: Global batch size
  - `--per-device-train-batch-size`: Batch size per device
  - `--seq-len`: Sequence length
  - `--max-tokens`: Maximum training tokens
  - `--learning-rate`: Learning rate

- **Scheduler Parameters**:
  - `--warmup-ratio`: Warmup ratio
  - `--lr-decay-style`: Learning rate decay style
  - `--min-lr-ratio`: Minimum learning rate ratio

## Core Concepts

### Hestia Method

Hestia (Hessian Trace Informed Softmax Annealing) is a quantization training method based on Hessian trace information:

1. **Hessian Trace Computation**: Efficiently estimate Hessian matrix trace using the Hutch++ algorithm
2. **Temperature Scaling**: Compute per-layer temperature scaling factors based on Hessian traces
3. **Asynchronous Annealing**: Different layers undergo asynchronous temperature annealing based on their sensitivity
4. **Three-Phase Training**:
   - **Compression Phase**: Gradually increase quantization pressure
   - **Annealing Phase**: Temperature annealing based on Hessian information
   - **Solidification Phase**: Stabilize quantization results

### Quantization Granularity

Three calibration granularities are supported:

- **tensor**: Each Linear layer computed independently (e.g., `q_proj`, `k_proj`, `v_proj` are independent)
- **layer**: All Linear layers within the same Transformer layer share the same trace
- **component**: Same component types share (all `q_proj` share, all `k_proj` share, etc.)

## Distributed Training

Hestia supports multi-node multi-GPU distributed training using the Accelerate framework:

```bash
# Single node multi-GPU
accelerate launch --config-file accelerate_config.yaml train.py ...

# Multi-node multi-GPU
# Set environment variables
export PET_NNODES=2
export PET_NODE_RANK=0  # or 1
export PET_MASTER_ADDR="master_node_ip"
export PET_MASTER_PORT=23456

accelerate launch --config-file accelerate_config.yaml \
    --main_process_ip ${PET_MASTER_ADDR} \
    --main_process_port ${PET_MASTER_PORT} \
    --machine_rank ${PET_NODE_RANK} \
    --num_machines ${PET_NNODES} \
    train.py ...
```

## Experiment Tracking

Hestia integrates with SwanLab for experiment tracking and visualization:

```bash
export SWANLAB_API_KEY="your_api_key"
export SWANLAB_MODE="cloud"  # or "local"
```

The following information is automatically logged during training:
- Training loss and metrics
- Quantization statistics (temperature, pressure, etc.)
- Model parameter changes

## Evaluation

Use the evaluation script to evaluate quantized models:

```bash
python eval/eval_model.py \
    --model-path PATH_TO_QUANTIZED_MODEL \
    --task-list task1 task2 \
    --output-path results.json
```
