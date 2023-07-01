#!/bin/bash

vals=(0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1)
#vals=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1)
#vals=(0.2 0.4 0.6 0.8 1)
for t in {0..19};
do
    echo "Collecting data for time-step ${vals[t]}"
    python collectdata.py --time ${vals[t]}
    echo "Now Training for time-step ${vals[t]}"
    python experiment_scripts/train_comparison.py --start $((t+1))
done