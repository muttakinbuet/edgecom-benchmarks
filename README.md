# Does Edge Text Help Overlapping Community Detection?

Code for the TMLR submission *Does Edge Text Help Overlapping Community
Detection? Two Benchmarks with Textual Edges*. It covers:

- the two benchmarks: **DBLP-EF**, a rebuild of the DBLP overlapping-community
  benchmark that keeps the papers behind every co-authorship edge, and
  **Pat-G**, a co-inventor graph from US patents;
- **MarkBP-auto**, a marked Bernoulli–Poisson model that uses the text of each
  edge, with two label-free permutation tests that decide whether to use it;
- the baselines, the experiments, the result files and the scripts that
  produce every table of the paper.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

All experiments run on CPU with one thread per run (`torch.set_num_threads(1)`),
which makes them deterministic for a given seed. Several runs execute in
parallel with `--workers`.

## Building the benchmarks

### DBLP-EF

```bash
# 1. Graph and venue communities from the raw DBLP XML (conference papers up
#    to 2006, main volumes only), keeping the papers behind every edge
python src/com_dblp.py --download --xml data/dblp.xml \
    --out data/com_dblp_ef_conf_le2006_venue_main \
    --pub-types inproceedings --max-year 2006 --community-mode venue \
    --main-volume-only

# 2. Abstracts and affiliations from OpenAlex (by DOI; cached and resumable)
python src/openalex_enrich.py --dataset data/com_dblp_ef_conf_le2006_venue_main

# 3. Evaluation subsets
python src/subset.py --dataset data/com_dblp_ef_conf_le2006_venue_main \
    --preset cs-areas --out data/subsets/cs_areas_le2006_main
python src/subset.py --dataset data/com_dblp_ef_conf_le2006_venue_main \
    --preset cs-systems --out data/subsets/cs_systems_le2006_main
python src/subset.py --dataset data/com_dblp_ef_conf_le2006_venue_main \
    --random-k 30 --min-size 300 --max-size 3000 --seed 1 \
    --out data/subsets/rand30_s1

# 4. Edge features (MiniLM sentence embeddings of the papers behind each edge);
#    --with-outside also writes the leakage-controlled *_outside variant
python src/text_edge_features.py --subset data/subsets/cs_areas_le2006_main \
    --text title --with-outside
python src/text_edge_features.py --subset data/subsets/cs_areas_le2006_main \
    --text title_abstract --cache data/openalex_cache.jsonl --with-outside

# 5. Institution ground truth (second, content-independent ground truth)
python src/affiliation_truth.py --subset data/subsets/cs_areas_le2006_main \
    --dataset data/com_dblp_ef_conf_le2006_venue_main \
    --out data/subsets/cs_areas_le2006_aff
```

Each subset folder holds `nodes.tsv`, `edges.tsv`, `communities.txt` (one
community per line), `stats.json` and one `edge_feats_<name>.npy` per feature
set, aligned with the rows of `edges.tsv`. The `stats.json` of every subset
records how it was cut.

### Pat-G

Download the PatentsView bulk tables `g_patent`, `g_cpc_current`,
`g_inventor_disambiguated`, `g_assignee_disambiguated` and `g_patent_abstract`
(`.tsv.zip`) from https://patentsview.org/download/data-download-tables into
`data/raw/patents`, then:

```bash
python src/patents.py --stage graph
python src/patents.py --stage features
```

This writes `data/subsets/patents_g_2006_2010` (CPC subclass ground truth) and
`data/subsets/patents_g_2006_2010_assignee` (assignee ground truth).

## Running the methods

`src/dblp_markbp.py` runs every method on one subset and appends one JSON line
per (seed, method, features, readout) to the output file. Finished jobs are
skipped on rerun.

```bash
python src/dblp_markbp.py --subset data/subsets/cs_areas_le2006_main \
    --features edge_feats_title edge_feats_title_abstract \
               edge_feats_title_outside edge_feats_title_abstract_outside \
    --shuffle-features edge_feats_title_abstract \
    --methods nocd edge_kmeans markbp_autoin markbp_autoin_shuffle \
              nocd_eagg dynares dynares_eagg nocd_gine cesna link_comm \
    --seeds 0 1 2 3 4 5 6 7 8 9 --workers 8 --out results/my_run.jsonl
```

The method names are listed at the top of `src/dblp_markbp.py`. Our method is
`markbp_autoin` (MarkBP-auto + input features); it is implemented in
`src/markbp_auto.py` and `src/ocd_models.py`. Node inputs averaged from edge
features (the `*_eagg` baselines and `markbp_autoin`) are built from training
edges only, so the held-out edges used for early stopping never reach the
encoder.

## Reproducing the paper

| Paper | Runs | Table script |
|---|---|---|
| Controlled graphs | `src/poc.py`, `src/poc_permutation.py`, `src/poc_homophily.py` | `src/poc_tables.py`, `src/auto_tables.py`, `src/plot_poc.py` |
| DBLP-EF baselines and subsets | `src/dblp_markbp.py` | `src/baseline_tables.py --out paper/tables` |
| Institutions as ground truth | `src/dblp_markbp.py` on `cs_areas_le2006_aff` | in `src/baseline_tables.py` |
| Pat-G | `src/dblp_markbp.py` on the two patent subsets | in `src/baseline_tables.py` |
| Alignment dial, placebo | `src/poc.py --align`, `src/placebo_features.py` + `src/dblp_markbp.py` | `src/plot_dial.py` |
| Significance tests | | `src/significance.py` |
| Variant screen, sensitivity | `src/variant_screen.py`, `src/sensitivity.py` | same scripts |
| Case study, per-venue results | `src/case_study.py`, `src/per_community.py` | same scripts |
| Protein complexes (appendix) | `src/yeast.py` | same script |

The result files behind every table are in `results/` (JSON lines), so the
tables can be regenerated without rerunning the experiments.

## Data and licences

- Code: MIT licence (`LICENSE`).
- DBLP-EF is derived from DBLP (CC0) and OpenAlex (CC0). Pat-G is derived from
  PatentsView (CC BY 4.0; attribution: USPTO PatentsView).
- We do not redistribute abstract texts, whose copyright may lie with
  publishers. The released benchmark contains their embeddings and the paper
  identifiers; `src/openalex_enrich.py` refetches the texts.

## DynaResGCN baseline

The DynaResGCN baselines use the official implementation released with the
DynaResGCN paper. Clone it into `external/dynaresgcn` (we used commit `51bfd6a`); see
`src/dynaresgcn.py`.

## Repository layout

```
src/            benchmark builders, models, baselines, experiments, table scripts
results/        JSON-lines results behind every table of the paper
paper/tables/   the generated LaTeX tables (output of the table scripts)
paper/figures/  the generated figures
external/       official DynaResGCN code (not included, see above)
```

The table scripts write to `paper/tables` by default, so regenerating a table
and comparing it with the committed one (`git diff`) checks the result.
