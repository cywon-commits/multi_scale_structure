#!/bin/bash
# Step 4: box-size dependence of P1, P2 (L = 384, 512 (done in batch2), 768); 2 noise seeds + qc seed
cd "$(dirname "$0")/sim"
for L in 384 768; do
 for f in ../configs/P1.json ../configs/P2.json; do
  python -u run_point.py $f --L $L --seeds noise,noise1,qc --nmax 8000 --tol 1e-7 2>&1 | grep --line-buffered -E "==|->|WINNER|seed"
 done
done
