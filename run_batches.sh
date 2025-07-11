#!/bin/bash

SEEDS=($1)
KS=($2)
MODES=($3)

for k in "${KS[@]}"; do
  if [[ "$k" -eq 0 ]]; then
    # For k=0, only run baseline
    run_modes=(baseline)
  else
    # For k!=0, run all user-specified modes except baseline
    run_modes=()
    for mode in "${MODES[@]}"; do
      [[ "$mode" == "baseline" ]] && continue
      run_modes+=("$mode")
    done
  fi

  for mode in "${run_modes[@]}"; do
    for seed in "${SEEDS[@]}"; do
      echo "Running: k=$k, mode=$mode, seed=$seed"
      python -m scripts.run --k "$k" --mode "$mode" --seed "$seed"
    done
  done
done
