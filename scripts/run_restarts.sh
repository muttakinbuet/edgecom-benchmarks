#!/usr/bin/env bash
# Restart experiment for the joint fit (appendix "Restarts of the joint fit").
# Ours (markbp_autoin) and the averaged-feature baseline (nocd_eagg) are fitted
# with restarts 0..2; restart 0 reproduces the main runs. Selection among
# restarts uses the held-out loss only (src/restarts.py). Resumable.
set -u
cd "$(dirname "$0")/.."
FEATS="edge_feats_title edge_feats_title_abstract edge_feats_title_outside edge_feats_title_abstract_outside"
run() {  # subset out workers seeds...
  local subset=$1 out=$2 workers=$3; shift 3
  python src/dblp_markbp.py --subset "$subset" --features $FEATS --methods markbp_autoin \
      --seeds "$@" --restarts 0 1 2 --workers "$workers" --out "results/restarts/$out.jsonl" \
      >> "results/restarts/$out.log" 2>&1
  python src/dblp_markbp.py --subset "$subset" --features edge_feats_title_abstract \
      --methods nocd_eagg --seeds "$@" --restarts 0 1 2 --workers "$workers" \
      --out "results/restarts/$out.jsonl" >> "results/restarts/$out.log" 2>&1
}
run data/subsets/cs_areas_le2006_main cs_areas 9 0 1 2 3 4 5 6 7 8 9
run data/subsets/patents_g_2006_2010 patents_g 7 0 1 2 3 4
run data/subsets/patents_g_2006_2010_assignee patents_g_assignee 7 0 1 2 3 4
