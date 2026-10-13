#!/bin/bash
# P1..P4 : 3 seeds (noise, qc, hex) each, field-calibrated
cd "$(dirname "$0")/sim"
for f in ../configs/P1.json ../configs/P2.json ../configs/P3.json ../configs/P4.json; do
  python -u run_point.py $f --seeds noise,noise1,noise2,qc,hex,qcr,hexr --nmax 20000 --tol 1e-7 2>&1 | grep --line-buffered -E "==|->|WINNER|calib|seed"
done
