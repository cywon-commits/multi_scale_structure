#!/bin/bash
# P0, P5, P6 (single noise seed; g=0 points have no field calibration)
cd "$(dirname "$0")/sim"
for f in ../configs/P0_c*.json ../configs/P5_c*.json ../configs/P6_g*.json; do
  python -u run_point.py $f --nmax 12000 --tol 1e-6 2>&1 | grep --line-buffered -E "==|->|WINNER|calib"
done
