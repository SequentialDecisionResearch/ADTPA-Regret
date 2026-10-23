# IYW holdings accounting repair

Fixed `choose` to accept existing IYW shares and include their marked-to-market value and scenario payoff in both the wealth denominator and numerator. Updated `run` call and test callers. This prevents the false `nonpositive wealth` exception when a portfolio holds IYW but no cash or individual stocks.

Run: `python -m pytest -q tests` and `python -m v3.cli --n 700 --d 6 --scenarios 100 --out results/experiment_01_repaired`.

**Scope:** This is a targeted accounting repair, not a certification of the backtester. Remaining issues include fee-matching of the IYW reference, pending-order consistency, and validity of risk/regret comparisons. Synthetic results are illustrative only.
