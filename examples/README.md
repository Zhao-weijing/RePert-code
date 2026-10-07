# Synthetic estimator example

The supplied NPZ is generated with NumPy seed 3407. It contains 32 fitting compounds, four distinct synthetic acquisition plates per fitting compound, 12 disjoint calibration compounds with three toy candidate predictions, and five disjoint query compounds. There are 16 features. These identities and profiles are artificial. The example calls the unchanged copied EB fitting/prediction and convex-weight functions. It does not train IMR or reproduce a manuscript experiment.

Run `python scripts/run_estimator_example.py`. Expected structural fields are in `expected_structure.json`. Predictions, weights and a reloadable EB model are written to `outputs/example/`.
