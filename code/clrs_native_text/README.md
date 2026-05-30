# CLRS Native-Text

This folder is the isolated code area for the new paired CLRS experiment.

The goal is to generate CLRS-Text from the same `dm-clrs` samples that provide
native CLRS tensors, so each row can carry both:

- official CLRS-Text `question` / `answer`
- native CLRS `inputs` / `hints` / `outputs` for the specialist

The B200 experiment extends this with hard natural-language descriptions whose
target is the exact native input row:

- `build_hard_nl_native_bridge_dataset.py`: native CLRS -> hard NL bridge rows.
- `split_native_raw_by_split.py`: combined native raw JSONL -> train/val/test files.
- `benchmark_openai_text_to_native.py`: GPT baseline for text -> native-input translation.

Keep this folder separate from older `stage2_bridge_v2` and `initial`
experiments.
