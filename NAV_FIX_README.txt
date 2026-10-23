NAV-only patch (2026-10-22)

Change: v3/engine.py now updates NAV after the day's cash contributions,
trades and fees, using W_t / (W_(t-1) + C_t). First observation uses
initial capital + C_t as the denominator. Same formula for IYW benchmark.
No strategy, gate, seed, cashflow, price or execution logic was changed.

Historical original results are preserved in:
  real_market_oos_outputs_before_nav_fix/
The new real_market_oos_outputs/ has the exact frozen price snapshot and
its data manifest; it is ready for a rerun without downloading.

From Anaconda Prompt in this project folder:
  python real_market_oos_validation.py --run
  python nav_regression_test.py --old real_market_oos_outputs_before_nav_fix --new real_market_oos_outputs

Expected: PASS lines for four policies and final PASS.
If a test fails, preserve both output folders and share the complete error.
Note: historic output summaries are not regenerated until --run is executed.
