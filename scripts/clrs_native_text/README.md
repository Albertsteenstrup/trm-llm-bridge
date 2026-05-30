# CLRS Native-Text Scripts

Run scripts for the isolated paired CLRS native/text experiment.

These scripts should write into `data/clrs_native_text` and should call code in
`code/clrs_native_text`.

Main B200 scripts:

- `make_ucloud_clrs_native_text_bundle.sh`: create `clrs_native_text_ucloud_minimal.tar.gz`.
- `install_ucloud_deps.sh`: install dm-clrs and bridge/specialist Python deps in the UCloud PyTorch app.
- `prepare_b200_clrs_bridge_specialist.sh`: generate native CLRS splits and hard-NL bridge splits.
- `prepare_b200_no_algo_curriculum.sh`: generate native CLRS splits plus no-algorithm-name bridge curriculum levels 0-4 and held-out OOD.
- `train_b200_specialist.sh`: train the native CLRS specialist.
- `train_b200_bridge.sh`: train the hard-NL -> native-input bridge.
- `train_b200_dynamic_curriculum_bridge.sh`: train the no-algorithm bridge with validation-gated curriculum advancement.
- `eval_bridge_specialist.sh`: run hard-NL -> bridge -> specialist inference.
- `run_openai_hard_nl_text_to_native.sh`: GPT-5.4 Code Interpreter baseline on the same hard-NL split, capped by `MAX_SAMPLES`.
