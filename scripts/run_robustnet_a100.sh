#!/usr/bin/env bash
set -euo pipefail

# Run from the ADA_Pro repository root on a CUDA Linux host.
# Paper method: RobustNet (CVPR 2021), pinned official implementation.
# TARGET DATA MUST NOT be used for training, source validation, or checkpoint selection.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
DATA_ROOT="${DGSEG_DATA_ROOT:?Set DGSEG_DATA_ROOT to the directory containing downloads/}"
WORK_ROOT="${ROBUSTNET_WORK_ROOT:-$(dirname "$REPO_ROOT")/DGSEG_REPRO}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
METHODS="${METHODS:-baseline isw}"
MODE="${MODE:-full}"   # smoke or full
BS="${BS:-4}"          # official per-GPU batch; original paper ran on two GPUs

if [[ "$MODE" != full && "$MODE" != smoke ]]; then
  echo "MODE must be full or smoke" >&2; exit 2
fi
"$PYTHON_BIN" -m pip install 'tensorboardX>=2.1' 'thop>=0.0.31' 'kmeans1d>=0.3.1' 'scikit-image>=0.16.2'
"$PYTHON_BIN" -m scripts.prepare_robustnet --data-root "$DATA_ROOT" --work-root "$WORK_ROOT" --extract

VENDOR="$WORK_ROOT/RobustNet"
export ROBUSTNET_CITYSCAPES_ROOT="$WORK_ROOT/cityscapes"
export DGSEG_DATA_ROOT="$DATA_ROOT"

for method in $METHODS; do
  case "$method" in
    baseline) extra=(--wt_reg_weight 0 --cov_stat_epoch 0 --wt_layer 0 0 0 0 0 0 0) ;;
    isw) extra=(--wt_reg_weight 0.6 --relax_denom 0.0 --clusters 2 --cov_stat_epoch 5 --trials 10 --wt_layer 0 0 2 2 2 0 0) ;;
    *) echo "Unknown method $method" >&2; exit 2 ;;
  esac
  out="$REPO_ROOT/runs/robustnet_${method}_${MODE}"
  mkdir -p "$out"
  if [[ "$MODE" == smoke ]]; then
    iterations=7; crop=256; batch=2; smoke_flag=(--test_mode)
  else
    iterations=40000; crop=768; batch="$BS"; smoke_flag=()
  fi
  echo "Training $method, mode=$MODE, iterations=$iterations, crop=$crop, batch=$batch"
  (
    cd "$VENDOR"
    "$PYTHON_BIN" train.py \
      --dataset cityscapes \
      --covstat_val_dataset cityscapes \
      --val_dataset \
      --arch network.deepv3.DeepR50V3PlusD \
      --city_mode train --lr_schedule poly --lr 0.01 --poly_exp 0.9 \
      --max_cu_epoch 10000 --class_uniform_pct 0.5 --class_uniform_tile 1024 \
      --crop_size "$crop" --scale_min 0.5 --scale_max 2.0 --rrotate 0 \
      --max_iter "$iterations" --bs_mult "$batch" --gblur --color_aug 0.5 \
      --date dgseg --exp "robustnet_${method}_${MODE}" --ckpt "$out" --tb_path "$out" \
      "${extra[@]}" "${smoke_flag[@]}" \
      2>&1 | tee "$out/train.log"
  )
  if [[ "$MODE" == smoke ]]; then
    "$PYTHON_BIN" -m scripts.evaluate_robustnet --vendor "$VENDOR" --data-root "$DATA_ROOT" \
      --method "$method" --smoke --output "$out/evaluation_smoke.json"
  else
    shopt -s nullglob
    checkpoints=("$out"/dgseg/robustnet_${method}_${MODE}/*/best_cityscapes_*.pth)
    shopt -u nullglob
    if [[ ${#checkpoints[@]} -eq 0 ]]; then
      echo "No source-validated official checkpoint found under $out" >&2; exit 1
    fi
    checkpoint="${checkpoints[0]}"
    "$PYTHON_BIN" -m scripts.evaluate_robustnet --vendor "$VENDOR" --data-root "$DATA_ROOT" \
      --method "$method" --checkpoint "$checkpoint" \
      --output "$REPO_ROOT/reports/evaluations/r19_robustnet_${method}_mapillary_v1.2.json"
  fi
done
