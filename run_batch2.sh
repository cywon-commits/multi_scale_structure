#!/bin/bash
cd "$(dirname "$0")/sim"
for f in ../configs/P3.json ../configs/P4.json; do
  python -u run_point.py $f --seeds noise,noise1,qc,hex --nmax 20000 --tol 1e-7 2>&1 | grep --line-buffered -E "==|->|WINNER|calib|seed"
done
