#!/bin/sh
# Five-seed retraining (seeds 0-4) of the leading direct and decomposed net-load models of
# the sweep with weather trajectories (--leads). The leaders are read from
# netload_stats_v030_leads.json; the reported run is seed 42.
cd "$(dirname "$0")"
DIRECT=$("${PYTHON:-python3}" -c "import json; print(json.load(open('netload_stats_v030_leads.json'))['summary']['best_direct'])")
DECOMP=$("${PYTHON:-python3}" -c "import json; print(json.load(open('netload_stats_v030_leads.json'))['summary']['best_decomposed'])")
echo "direct $DIRECT, decomposed $DECOMP"
for s in 0 1 2 3 4; do
  PYTHONPATH=../../.. "${PYTHON:-python3}" netload_sweep_v030.py --leads --seed $s --only direct --convs "$DIRECT"
  PYTHONPATH=../../.. "${PYTHON:-python3}" netload_sweep_v030.py --leads --seed $s --only decomposed --convs "$DECOMP"
done
echo ALL_SEEDS_DONE
