#!/usr/bin/env bash
set -euo pipefail

CODE_BASE="Hestia/"
cd "${CODE_BASE}"

export PYTHONPATH="${CODE_BASE}/src"

RUN_MODE=${RUN_MODE:-train}
QUANT_TYPE="hestia"
QUANT_CONFIG_PATH="${CODE_BASE}/configs/quant_example.yaml"

DATE_TAG=${DATE_TAG:-$(date +%Y-%m-%d)}
TIME_TAG=${TIME_TAG:-$(date +%H-%M-%S)}

MODEL_NAME="Llama-3.2-1B"
LOAD_DIR="PATH_TO_LOAD_DIR"
TOKENIZER_DIR="${LOAD_DIR}"
DATA_DIR="PATH_TO_DATA_DIR"
DATASET_NAME="Ultra-FineWeb"

# Hessian traces path
HESSIAN_TRACES_PATH="PATH_TO_HESSIAN_TRACES.pkl"

EXP_NAME="EXP_NAME"
SAVE_DIR="PATH_TO_SAVE_DIR"
LOG_PATH="PATH_TO_LOG_PATH"
mkdir -p "${SAVE_DIR}" 
SWANLAB_API_KEY="${SWANLAB_API_KEY:-}"
SWANLAB_MODE="${SWANLAB_MODE:-local}"
SWANLAB_LOGDIR="PATH_TO_SWANLAB_LOG_DIR"
mkdir -p "${SWANLAB_LOGDIR}"

GPUS_PER_NODE=8
NUM_NODES="${PET_NNODES:-1}"
NODE_RANK="${PET_NODE_RANK:-0}"
MASTER_ADDR="${PET_MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${PET_MASTER_PORT:-23456}"
NPROC_PER_NODE=$((${GPUS_PER_NODE} * ${NUM_NODES}))

ACCELERATE_CONFIG_PATH="PATH_TO_ACCELERATE_CONFIG.yaml"
if [[ ! -f "${ACCELERATE_CONFIG_PATH}" ]]; then
    echo "[ERROR] Accelerate config not found: ${ACCELERATE_CONFIG_PATH}" >&2
    exit 1
fi

[[ -f "${QUANT_CONFIG_PATH}" ]] || {
    echo "[ERROR] Quant config not found: ${QUANT_CONFIG_PATH}" >&2
    exit 1
}


MODEL_ARGS=(
    --bf16
    --load-dir "${LOAD_DIR}"
)


TRAIN_ARGS=(
    --global-batch-size 256
    --per-device-train-batch-size 16
    --seq-len 1024
    --max-tokens 8317664256
    --num-train-epochs 1
    --learning-rate 5e-5
    --adam-beta1 0.9
    --adam-beta2 0.95
    --weight-decay 0.01
    --max-grad-norm 1.0
    --resume
)

DISTRIBUTED_ARGS=(
    --main_process_ip "${MASTER_ADDR}"
    --main_process_port "${MASTER_PORT}"
    --machine_rank "${NODE_RANK}"
    --num_machines "${NUM_NODES}"
    --num_processes "${NPROC_PER_NODE}"
)



QUANT_ARGS=(
    --quant-type "${QUANT_TYPE}"
    --skip-layers lm_head
    --quant-config-path "${QUANT_CONFIG_PATH}"
    --hessian-traces-path "${HESSIAN_TRACES_PATH}"
)


SCHED_ARGS=(
    --warmup-ratio 0.05
    --lr-decay-style "wsd"
    --wsd-ratio 0.1
    --min-lr-ratio 0.1
    --stage-ratio 0.5
)


DATA_ARGS=(
    --random-seed 42
    --data-dir "${DATA_DIR}"
    --tokenizer-dir "${TOKENIZER_DIR}"
)


LOG_ARGS=(
    --output-dir "${SAVE_DIR}"
    --logging-path "${LOG_PATH}"
    --report-to swanlab
    --swanlab-api-key "${SWANLAB_API_KEY}"
    --swanlab-mode "${SWANLAB_MODE}"
    --swanlab-logdir "${SWANLAB_LOGDIR}"
    --model-name "${MODEL_NAME}"
    --dataset-name "${DATASET_NAME}"
    --date-str "${DATE_TAG}"
    --time-str "${TIME_TAG}"
    --logging-steps 1
    --save-steps 1000
    --save-strategy steps
    --save-total-limit 50
)


accelerate launch \
    --config-file "${ACCELERATE_CONFIG_PATH}" \
    "${DISTRIBUTED_ARGS[@]}" \
    train.py \
    "${MODEL_ARGS[@]}" \
    "${QUANT_ARGS[@]}" \
    "${TRAIN_ARGS[@]}" \
    "${SCHED_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${LOG_ARGS[@]}"
