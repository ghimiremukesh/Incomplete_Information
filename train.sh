#!/bin/bash

vals=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1)

for t in {0..9};
do
    echo "Collecting data for time-step ${vals[t]}"
    python collectdata_4d_follow_zero_relative_state.py --time ${vals[t]}
    echo "Now Training for time-step ${vals[t]}"
    python experiment_scripts/train_toy_soccer_4d.py --start $((t+1))
done