# RayLabTF-RBP

This project analyzes exact alternative-splicing event overlap between transcription factors (TFs) and RNA-binding proteins / RNA-associated regulators (RBPs). It also matches the TF–RBP overlap events against age/sex splicing datasets to summarize which candidate events are male-higher, female-higher, or directionally mixed across ages.

The main entry point is [script.py](script.py), which reads the combined CSV tables in [data](data) and generates summary tables plus publication-style figures.

## What the analysis produces

Running the script creates:

- a TF × RBP exact-event overlap matrix
- a Jaccard-similarity heatmap for TF–RBP overlaps
- a site-level CSV listing the important shared events behind the Jaccard matrix
- age/sex candidate-event associations
- age persistence summaries
- male-higher, female-higher, and combined direction plots

The generated outputs are written to [tf_rbp_analysis_outputs](tf_rbp_analysis_outputs), and selected files are also available in [graphs](graphs).

## Input data

The repository includes these combined input tables:

- [data/01_all_tf_splicing_events.csv](data/01_all_tf_splicing_events.csv)
- [data/02_all_rbp_splicing_events.csv](data/02_all_rbp_splicing_events.csv)
- [data/03_all_age_sex_splicing_events.csv](data/03_all_age_sex_splicing_events.csv)
- [data/04_dataset_manifest.csv](data/04_dataset_manifest.csv)

The script can also work with the original per-dataset files if they are present, but the combined tables above are the expected inputs for this workspace.

## Key outputs

Important generated files include:

- [tf_rbp_analysis_outputs/tf_rbp_exact_event_overlap_heatmap.png](tf_rbp_analysis_outputs/tf_rbp_exact_event_overlap_heatmap.png)
- [tf_rbp_analysis_outputs/tf_rbp_jaccard_matrix.csv](tf_rbp_analysis_outputs/tf_rbp_jaccard_matrix.csv)
- [tf_rbp_analysis_outputs/tf_rbp_jaccard_important_sites.csv](tf_rbp_analysis_outputs/tf_rbp_jaccard_important_sites.csv)
- [tf_rbp_analysis_outputs/tf_rbp_candidate_age_sex_associations.csv](tf_rbp_analysis_outputs/tf_rbp_candidate_age_sex_associations.csv)
- [tf_rbp_analysis_outputs/tf_rbp_candidate_age_persistence_summary.png](tf_rbp_analysis_outputs/tf_rbp_candidate_age_persistence_summary.png)
- [tf_rbp_analysis_outputs/tf_rbp_candidate_age_persistence_male_higher.png](tf_rbp_analysis_outputs/tf_rbp_candidate_age_persistence_male_higher.png)
- [tf_rbp_analysis_outputs/tf_rbp_candidate_age_persistence_female_higher.png](tf_rbp_analysis_outputs/tf_rbp_candidate_age_persistence_female_higher.png)
- [tf_rbp_analysis_outputs/tf_rbp_candidate_age_persistence_male_female_combined.png](tf_rbp_analysis_outputs/tf_rbp_candidate_age_persistence_male_female_combined.png)

## Requirements

The script uses:

- Python 3.14+
- pandas
- numpy
- matplotlib

If you are starting from a clean environment, install the Python packages with:

```bash
pip install pandas numpy matplotlib
```

## How to run

From the repository root:

```bash
python script.py --data-dir data --output-dir tf_rbp_analysis_outputs
```

If you want the figures and CSVs written somewhere else, change `--output-dir` to that folder.

## Notes

- The Niranjan count-colored heatmap and its [exploratory exact-test table](graphs/niranjan/tables/tf_rbp_count_exact_test_exploratory.csv) can be regenerated with `.venv/bin/python overlap_count_significance.py`. The heatmap marks pairs with Benjamini–Hochberg adjusted p < 0.05 with an asterisk. For each pair, the table reports the observed and expected overlap, fold enrichment, one-sided exact p-value, and adjusted p-value; negative log10 values preserve very small probabilities when floating-point p-values round to zero. The pairwise test calls the R package SuperExactTest directly (`MSET`, upper tail); install it first with `Rscript -e 'install.packages("SuperExactTest", repos="https://cloud.r-project.org")'`. The generated results used SuperExactTest 1.1.0.
- These tests use the union of observed TF and RBP events as the background because a common universe of all events tested in every experiment is not available. This selected background and differences in experimental coverage can bias results; the null also assumes independent random selection of events. The results are exploratory, not validated evidence of enrichment. Same-named TF and RBP sets, such as mod–mod, need particular care because they may share source data. A defensible confirmatory analysis requires a common set of detectable/tested events and consideration of study dependence.
- The script removes exact duplicate TF/RBP input files by content before calculating overlaps.
- The Jaccard matrix is based on exact shared `event_id` matches.
- The site-level CSV expands each nonzero TF–RBP overlap into one row per shared event, with gene IDs, event IDs, event types, experiment summaries, and directionality metadata.
