# Shared contracts: how to use them

`hedgelab.contracts` is the boundary between the calibration and the pricing/hedging
code. Each part is written against the contracts and never imports the other's modules, so
people can work in parallel and swap implementations (e.g. NIG for VG) without breaking
anyone.

## What is in the package

| Type | Kind | Purpose |
|------|------|---------|
| `CollarSpec` | dataclass | Terms of the collar: call and put strikes, averaging window, futures legs |
| `CalibratedModel` | dataclass | Result of a calibration: model name, `as_of`, parameters, implied-vol RMSE |
| `HedgePosition` | dataclass | Target hedge on one date: signed lots per instrument ticker |
| `Calibrator` | protocol | `calibrate(chain, as_of) -> CalibratedModel` |
| `Pricer` | protocol | `price(model, spec, as_of) -> float` |
| `Hedger` | protocol | `hedge(model, spec, as_of) -> HedgePosition` |

Data flow:

```
option chain --Calibrator--> CalibratedModel --Pricer / Hedger--> price / HedgePosition
```

## Rules

1. Import shared types from the package root: `from hedgelab.contracts import CalibratedModel`.
2. Contracts never import calibration, pricing or hedging code. They depend only on the
   standard library and pandas.
3. Every method takes an explicit `as_of: date`. The first step of an implementation is
   `filter_as_of(...)`, so no data after `as_of` is ever used.
4. Raise the domain exceptions from `hedgelab.exceptions` (`CalibrationError`,
   `PricingError`), never bare `Exception`.
5. Dataclasses are immutable. Build a new object instead of editing one.

## Implementing a protocol

Protocols are structural: a class satisfies one by having the right method, with no
inheritance needed.

```python
from datetime import date

import pandas as pd

from hedgelab.contracts import CalibratedModel
from hedgelab.loaders import filter_as_of


class NigCalibrator:
    """Calibrate a NIG model on the option chain."""

    def calibrate(self, chain: pd.DataFrame, as_of: date) -> CalibratedModel:
        quotes = filter_as_of(chain, as_of)
        params, rmse = self._fit(quotes)
        return CalibratedModel("NIG", as_of, params, rmse)
```

A consumer depends only on the protocol:

```python
from hedgelab.contracts import Calibrator, Hedger


def run_day(calibrator: Calibrator, hedger: Hedger, chain, spec, as_of):
    model = calibrator.calibrate(chain, as_of)
    return hedger.hedge(model, spec, as_of)
```

## Working in parallel

Until the other side exists, test against a small fake that satisfies the protocol. For
example, a hedger can be developed with a `Calibrator` that returns a fixed
`CalibratedModel`. `tests/test_contracts.py` shows the pattern. The check
`isinstance(obj, Calibrator)` confirms an implementation exposes the expected method.

## Changing a contract

A change to a file in `hedgelab/contracts/` affects everyone, so:

1. Discuss it with the whole team first, in an issue or in the group.
2. Open a PR from a branch named `contracts/<short-description>`.
3. Merge only after all contributors who use the changed type have approved.
4. Adding a field with a default or a new type is safe. Removing or renaming a public name
   breaks other people's code and needs an agreed migration.

## Checks

```bash
uv run ruff format . && uv run ruff check .
uv run mypy
uv run pytest -q -m "not slow"
```
