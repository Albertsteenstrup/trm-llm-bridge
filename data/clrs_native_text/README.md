# CLRS Native-Text Data

Isolated data area for the new paired CLRS native/text experiment.

- `raw/`: paired rows or native CLRS rows generated directly from `dm-clrs`.
- `prepared/`: derived native specialist splits and hard-NL bridge splits.
- `test_splits/`: fixed subsets for GPT, bridge, and specialist comparisons.

Do not mix these files with earlier CLRS attempts under `data/benchmarks/clrs`,
`data/raw/clrs_text_official`, or `data/prepared/clrs_text_bridge_v2_paper`.
