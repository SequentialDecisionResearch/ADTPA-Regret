# ADTPA decision audit

import inspect
import pandas as pd
import v3.engine as engine
from legacy_adtpa.data import synthetic

# Modify choose() only in memory, not on disk.
source = inspect.getsource(engine.choose)
old = "    return best\n"
new = "    best['all_candidates'] = diagnostics\n    return best\n"

if source.count(old) != 1:
    raise RuntimeError("Cannot locate the expected return statement")

namespace = dict(vars(engine))
exec(source.replace(old, new), namespace)
audit_choose = namespace["choose"]

captured = []

def wrapped_choose(prices, t, shares, iyw_shares, cash,
                   config, policy, decision_number):
    result = audit_choose(
        prices, t, shares, iyw_shares, cash,
        config, policy, decision_number
    )
    if str(prices.index[t].date()) == "2020-02-17":
        captured.extend(result.get("all_candidates", []))
    return result

engine.choose = wrapped_choose

prices = synthetic(n=700, d=6, seed=22, regime="mixed")
prices = prices[
    [c for c in prices.columns if c != "IYW"] + ["IYW"]
]

cfg = engine.Config(seed=22, scenarios=500, start=233)
engine.run(prices, "combined_dynamic", cfg)

if not captured:
    raise RuntimeError("No diagnostics captured")

df = pd.DataFrame([
    {
        "horizon": x["horizon"],
        "action": x["action"],
        "score": x["score"],
        "feasible": x["feasible"],
        **x["metrics"],
    }
    for x in captured
])

print("\nBEST CANDIDATE BY ACTION:")
best = (
    df.sort_values("score", ascending=False)
      .groupby("action", sort=False)
      .head(1)
)
print(best.sort_values("score", ascending=False).to_string(index=False))

print("\nTOP 10 CANDIDATES:")
print(
    df.sort_values("score", ascending=False)
      .head(10)
      .to_string(index=False)
)


# Bootstrap stability audit at the same portfolio state
print("\nBOOTSTRAP STABILITY AUDIT")

target = "2020-02-17"
original_scenarios = engine.scenarios_at
original_wrapped_choose = engine.choose

results = []
current_repetition = None

def resampled_scenarios(prices, t, h, config, seed):
    if (current_repetition is not None
            and str(prices.index[t].date()) == target):
        seed = seed + 100000 * current_repetition
    return original_scenarios(prices, t, h, config, seed)

def stability_choose(prices, t, shares, iyw_shares, cash,
                     config, policy, decision_number):
    global current_repetition

    if str(prices.index[t].date()) != target:
        return original_wrapped_choose(
            prices, t, shares, iyw_shares, cash,
            config, policy, decision_number
        )

    # Repeat evaluation at exactly the same portfolio state.
    for rep in range(100):
        current_repetition = rep
        result = audit_choose(
            prices, t, shares, iyw_shares, cash,
            config, policy, decision_number
        )

        candidates = result["all_candidates"]
        best_by_action = {}

        for action in ("stock_buy", "iyw_buy", "cash_wait"):
            feasible = [
                x for x in candidates
                if x["action"] == action and x["feasible"]
            ]
            if feasible:
                best_by_action[action] = max(
                    feasible, key=lambda x: x["score"]
                )

        row = {
            "rep": rep,
            "winner": result["action"],
            "horizon": result["horizon"],
        }

        for action, candidate in best_by_action.items():
            row[action] = candidate["score"]

        results.append(row)

    current_repetition = None

    # Execute the original decision, not a resampled decision.
    return original_wrapped_choose(
        prices, t, shares, iyw_shares, cash,
        config, policy, decision_number
    )

engine.scenarios_at = resampled_scenarios
namespace["scenarios_at"] = resampled_scenarios
engine.choose = stability_choose

engine.run(prices, "combined_dynamic", cfg)

stability = pd.DataFrame(results)

print("\nWinner frequencies:")
print(stability["winner"].value_counts(dropna=False))

print("\nScore advantage distributions:")
for baseline in ("iyw_buy", "cash_wait"):
    difference = stability["stock_buy"] - stability[baseline]
    print(
        baseline,
        "mean=", difference.mean(),
        "median=", difference.median(),
        "positive_fraction=", (difference > 0).mean(),
        "5%=", difference.quantile(0.05),
        "95%=", difference.quantile(0.95),
    )

stability.to_csv("seed22_bootstrap_stability_500.csv", index=False)
print("\nSaved: seed22_bootstrap_stability_500.csv")
