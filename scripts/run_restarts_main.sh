#!/usr/bin/env bash
# Restarts for the main tables: best of 3 restarts by held-out loss, for ours
# (markbp_autoin) and the two strongest baselines (nocd_eagg, dynares_eagg), on
# all five ground truths and the four MiniLM feature sets. Extends
# run_restarts.sh, whose result files it appends to. Five workers on the larger
# graphs: nine exhaust 16 GB of memory with DynaResGCN and the runs swap. Restart 0 reproduces the
# single runs. Resumable.
set -u
cd "$(dirname "$0")/.."
FEATS="edge_feats_title edge_feats_title_abstract edge_feats_title_outside edge_feats_title_abstract_outside"
run() {  # subset out workers methods seeds...
  local subset=$1 out=$2 workers=$3 methods=$4; shift 4
  python src/dblp_markbp.py --subset "$subset" --features $FEATS --methods $methods \
      --seeds "$@" --restarts 0 1 2 --workers "$workers" --out "results/restarts/$out.jsonl" \
      >> "results/restarts/$out.log" 2>&1
}
run data/subsets/cs_systems_le2006_main cs_systems 9 "markbp_autoin nocd_eagg dynares_eagg" 0 1 2 3 4
run data/subsets/rand30_s1 rand30 9 "markbp_autoin nocd_eagg dynares_eagg" 0 1 2 3 4
run data/subsets/cs_areas_le2006_main cs_areas 5 "nocd_eagg dynares_eagg" 0 1 2 3 4 5 6 7 8 9
run data/subsets/patents_g_2006_2010 patents_g 5 "nocd_eagg dynares_eagg" 0 1 2 3 4
run data/subsets/patents_g_2006_2010_assignee patents_g_assignee 5 "nocd_eagg dynares_eagg" 0 1 2 3 4
