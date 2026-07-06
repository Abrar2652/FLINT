# REPRODUCE.md — one command per number (target state, Gate 6)

> Filled in as gates pass. Goal: a stranger reruns every paper number from here.

## Environment
```bash
pip install -e ".[dev,llm]"
pytest -q                     # infrastructure tests (pass today)
```

## Gate 1 — baseline reproduction
```bash
# 1. run GRAMS submodule on 250WT (see baselines/grams_wrapper.py)
# 2. score + tolerance check:
python scripts/run_gate1_reproduce.py --grams-output <preds.json>
```

## Main results (per table, after Gate 6)
```bash
# example shape; make_tables.py reads experiments/results/ and emits LaTeX
for s in 0 1 2 3 4; do
  python scripts/run_experiment.py --config configs/default.yaml seed=$s dataset=250wt
done
python scripts/make_tables.py --table main
```

Every run writes: resolved config, git SHA, library versions, per-table scores,
and mean±std. No number appears in the paper that isn't traceable to a run dir.
