#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT=${DATA_ROOT:-/data/lizhe/SH_data/Mic8_2s_gpurir}
MODEL_PATH=${MODEL_PATH:-model_grouping_inter_adfs_mse_sisdr_stft_8mic}
MODEL_NAME=${MODEL_NAME:-model_best.pth}
TEST_NAME=${TEST_NAME:-mic_8_snr3_t603_1000}
GPUS=${GPUS:-0}
PREDICTION_PATH=${PREDICTION_PATH:-${DATA_ROOT}/predictions_${MODEL_PATH}_test_${TEST_NAME}}
SAVE_DIR=${SAVE_DIR:-${MODEL_PATH}/eval_${TEST_NAME}}

python evaluation_fixed.py \
  --gpus "${GPUS}" \
  --dataset_root "${DATA_ROOT}" \
  --modelpath "${MODEL_PATH}" \
  --model_name "${MODEL_NAME}" \
  --test_name "${TEST_NAME}" \
  --prediction_path "${PREDICTION_PATH}" \
  --save_dir "${SAVE_DIR}"
