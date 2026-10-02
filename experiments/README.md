# Experiment packages

`rank8/` is the reference executable package. It contains the actual sequential driver, launchers, objective-separation tests, configuration, and the frozen rank-8 Bridge.

`rank32/` and `rank64/` contain the rank-specific method/training configurations and their frozen Bridge tensors. They use the same driver logic as rank 8 with the rank, experiment path, and device selected at deployment time.

The configuration files retain `{root}` and `{experiment}` placeholders. Launchers resolve these against the local checkout. Public model identifiers replace machine-local model paths.

Before running:

1. Supply the 7,500-example prepared training parquet at `artifacts/math_eopd_author_train7500.parquet`.
2. Supply the three evaluation parquet files under `third_party/OPRD/datasets/test_data/`.
3. Overlay `code/verl_extensions/` into the OPRD VERL package used for the EMD stage.
4. Verify `bridge.pt` against `bridge_sha256.json`.
5. Run a one-step full-model smoke test before the 174-step stage.

The included pipeline has conservative hardware gates from the shared-server deployment. Adapt device selection and capacity gates to an isolated machine without changing the method or training hyperparameters.
