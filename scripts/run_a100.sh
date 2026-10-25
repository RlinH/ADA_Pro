#!/usr/bin/env bash
set -euo pipefail

# Run from the project root on one A100. Set DATA_ROOT and choose tracks/methods/seeds.
DATA_ROOT="${DGSEG_DATA_ROOT:?Set DGSEG_DATA_ROOT to the directory containing downloads/}"
TRACKS="${TRACKS:-r19}"
METHODS="${METHODS:-erm mixstyle randconv}"
SEEDS="${SEEDS:-0 1 2}"
EPOCHS="${EPOCHS:-80}"
BATCH_SIZE="${BATCH_SIZE:-4}"
WORKERS="${WORKERS:-4}"

for track in $TRACKS; do
  python3 -m scripts.build_manifests --data-root "$DATA_ROOT" --track "$track"
  case "$track" in
    r19) target=mapillary_v1.2; height=512; width=1024 ;;
    rs7_u2r) target=rural; height=512; width=512 ;;
    rs7_r2u) target=urban; height=512; width=512 ;;
    *) echo "Unsupported track: $track" >&2; exit 2 ;;
  esac
  for method in $METHODS; do
    for seed in $SEEDS; do
      run_dir="runs/${track}_${method}_seed${seed}"
      mkdir -p "$run_dir"
      resume=()
      if [[ -f "$run_dir/last.pt" ]]; then
        resume=(--resume "$run_dir/last.pt")
      fi
      python3 -m scripts.train \
        --data-root "$DATA_ROOT" --track "$track" --method "$method" \
        --run-dir "$run_dir" --seed "$seed" --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" --workers "$WORKERS" --height "$height" --width "$width" \
        --pretrained-backbone "${resume[@]}"
      python3 -m scripts.evaluate \
        --data-root "$DATA_ROOT" --track "$track" --target "$target" \
        --height "$height" --width "$width" \
        --checkpoint "$run_dir/best_source_val.pt" \
        --output "reports/evaluations/${track}_${method}_seed${seed}_${target}.json"
    done
  done
done

python3 -m scripts.summarize
