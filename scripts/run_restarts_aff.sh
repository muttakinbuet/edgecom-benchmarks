#!/usr/bin/env bash
# Restarts on the institution ground truth (CS-Areas graph, title and
# title + abstract features, 5 seeds), as in run_restarts_main.sh. Resumable.
set -u
cd "$(dirname "$0")/.."
python src/dblp_markbp.py --subset data/subsets/cs_areas_le2006_aff \
    --features edge_feats_title edge_feats_title_abstract \
    --methods markbp_autoin nocd_eagg dynares_eagg --seeds 0 1 2 3 4 \
    --restarts 0 1 2 --workers 5 --out results/restarts/cs_areas_aff.jsonl \
    >> results/restarts/cs_areas_aff.log 2>&1
