#!/bin/bash
cd "$(dirname "$0")/sim"
for f in ../configs/P5_c*.json ../configs/P6_g*.json; do
  python -u run_point.py $f --nmax 8000 --tol 1e-6 2>&1 | grep --line-buffered -E "==|->|WINNER|calib"
done
for f in ../configs/P0_c0.5.json ../configs/P0_c0.8.json ../configs/P5_c0.85.json ../configs/P5_c0.93.json; do
  python -u run_point.py $f --seeds sp1,sp2,sp2c --tag _sp --nmax 8000 --tol 1e-6 2>&1 | grep --line-buffered -E "==|->|WINNER|seed"
done
