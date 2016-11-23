#!/usr/bin/env bash
if [[ $# -ne 1 ]]; then
  echo "usage: $0 glob_to_run_outs"
  exit 1
fi

set -ex

./p/emit_rewards.py $1 > /tmp/r.tsv &
./p/emit_stats.py $1 > /tmp/mt.tsv &
wait
Rscript --vanilla ./p/emit_plots.R
eog /tmp/plots.png
