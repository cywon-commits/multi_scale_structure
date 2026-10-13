#!/bin/bash
# g=0 benchmark seeds (single spiral, orthogonal pair, collinear pair) and P6 (1q vs hex/SkX) at fixed box
cd "$(dirname "$0")/sim"
for f in ../configs/P0_c0.2.json ../configs/P0_c0.5.json ../configs/P0_c0.8.json ../configs/P0_c0.95.json ../configs/P0_c1.1.json ../configs/P5_c0.85.json ../configs/P5_c0.93.json; do
  python -u run_point.py $f --seeds sp1,sp2,sp2c --tag _sp --nmax 12000 --tol 1e-6 2>&1 | grep --line-buffered -E "==|->|WINNER|seed"
done
for f in ../configs/P6_g0.5.json ../configs/P6_g0.8.json; do
  python -u run_point.py $f --seeds sp1,hex --tag _sp --nmax 12000 --tol 1e-6 2>&1 | grep --line-buffered -E "==|->|WINNER|seed|calib"
done
