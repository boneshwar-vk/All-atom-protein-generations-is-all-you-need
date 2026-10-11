# Adding a scorer (or reward, or stage)

The system is registry-driven: new evaluators are added by implementing an interface and
registering under a name, then referenced **by name from config** — no call sites change.

## 1. Implement the interface

```python
# e.g. packages/toolkit/src/aapg/toolkit/my_scorer.py
from aapg.core.scoring import SCORERS, Scorer, Score


@SCORERS.register("my_scorer")
class MyScorer(Scorer):
    name = "my_scorer"

    def __init__(self, threshold: float = 1.0) -> None:
        self.threshold = threshold

    def score(self, candidate) -> Score:
        value = ...  # compute from candidate (structure / complex / sequence)
        return Score(values={"my_metric": value})
```

## 2. Add a config

```yaml
# configs/scorer/my_scorer.yaml
name: my_scorer
threshold: 2.0
```

## 3. Reference it

Add it to a reward bundle (`configs/reward/*.yaml`), a validation stage
(`configs/pipeline/*.yaml`), or the iterative-MPNN scorer list — by name. Consumers resolve it
via `SCORERS.create("my_scorer", **cfg)`.

## Notes

- Rewards use the same pattern with `REWARDS` and the `Reward` ABC.
- Validation stages use the `Stage` ABC in `aapg.validation.stages`.
- Keep anything that shells out to an external tool in `aapg.toolkit`, not `aapg.core`.
- Make sure the module defining the scorer is imported so its `@register` runs (entry-point or
  explicit import in the package `__init__`).
