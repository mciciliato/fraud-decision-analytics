# Decisions and assumptions

Every non-obvious choice, the alternative considered, and why it was rejected. All parameter values are invented for the synthetic programme, and none is calibrated to a real system. The mechanic is deliberately different in kind from the source study's: a time-based payout chance rising over a cycle, with the pool shared pro rata among the spend of the paying step. There is no single winner, no level-triggered pool, and no tiers. No window, cap or pool size is taken from the source.

Confidence: **H** high, **M** medium, **L** low. Impact: what changes if the assumption is wrong.

## Method

| # | Decision | Alternative considered | Why rejected | Confidence | Impact if wrong |
|---|---|---|---|---|---|
| D1 | Late-entry baseline = share of *time* the programme spends late in its cycle | Share of all entries that are late; share among unflagged accounts | Both depend on the accounts under test (exploiters raise the entry share; "unflagged" depends on the flag): selection circularity | H | Biased p-values; FDR no longer controlled |
| D2 | Cohort defined on behaviour only (timing + breadth); never on wins or returns | Include winners or high returners | Outcome-based selection inflates the measured effect: section 1 shows 1.27 in-sample vs 1.02 on holdout. A test enforces that shuffling rewards leaves the flags unchanged | H | Edge and cost overstated |
| D3 | "Late" = cycle step at least 60% of the maximum cycle | 50%, 75% | 50% dilutes with ordinary activity; 75% is reached in too few cycles (payouts usually happen before it) | M | Lower precision at 50%, lower recall at 75% |
| D4 | One-sided exact binomial test, Benjamini-Hochberg at 1% | Normal approximation; Bonferroni; fixed share cut-off | Small accounts and a ~9% baseline break the normal approximation; Bonferroni over ~3,800 tests is too strict for a review queue; a fixed cut-off ignores sample size | M | More false positives at a looser FDR |
| D5 | Exposure breadth: late entries in at least 3 distinct cycles | No breadth requirement | One busy late week would otherwise qualify an account; breadth is what separates a strategy from a coincidence | M | Lower recall if set higher |
| D6 | Minimum 8 entries to be assessed | 20 | Exploiters have little activity outside their window; the exact test stays valid at small n | M | Recall falls as it rises |
| D7 | Flag on even cycles, measure on odd cycles | Same cycles | Same-sample measurement is the circularity this study demonstrates | H | Optimistic edge |
| D8 | Theoretical return = base cashback + time-average of (pool paid / step spend) | Analytic design return; return of unflagged accounts | Analytic return ignores how real spend moves over the cycle; unflagged accounts depend on the flag. The trajectory-based value is label-free and matches regular accounts (test) | H | Benchmark biased |
| D9 | z-test on a pooled delta-method standard error with accounts as the unit | z-test over entries | Entries of one account share a policy and spend level; treating them as independent understates the error | H | Overstated significance |
| D10 | BCa bootstrap over accounts, 2,000 resamples | Percentile bootstrap | Returns are skewed (pool payouts); BCa corrects bias and skew | M | Intervals miscentred |
| D11 | Results reported for cohorts only | Per-account edge | Per-account estimates were unstable for low-volume accounts in the source study; cohort ratios are what the programme pays | H | Individual verdicts unsupported by the data |
| D12 | Time-to-edge = first cycle from which P(return >= 1 + x) stays above the confidence level for all later cycles, plus the share of paths that never fall back | First crossing | A curve can touch the level and fall back; only a sustained crossing describes when the edge is reliable | M | Edge looks faster than it is |
| D13 | Monte Carlo engine summarises the population by its expected spend per cycle step and vectorises over paths x cycles | Per-entry simulation | The source study took ~20 minutes with per-entry loops; this runs in under a second. The cost (the engine misses burstiness and crowding) is stated as a limitation | M | Prediction differs from observed cohort (it does) |
| D14 | Cost figures use observed holdout extraction, not the Monte Carlo | Use the simulation | The simulation under-predicts extraction per cycle (1.35 vs 2.74 credits) | H | Cost understated by about half |
| D15 | Detection-threshold rule: at least 3 late entries, a late majority, late entries in at least 2 cycles, evaluated after N cycles and averaged over windows starting every 3 cycles | Re-use the FDR test | A team acting after a few cycles needs a simple rule; the FDR test has no power over 1-3 cycles. Rolling windows stop the answer hinging on how long the first cycles happened to be (the first six included only two long cycles) | M | N moves |
| D16 | Expected cost(N) = extraction x N x caught + extraction x 40 cycles x missed + 60 credits x false positives | Ignore false positives; infinite horizon | Both are needed for a trade-off; the horizon and the false-positive cost are the main assumptions behind N | L | N moves; the curve is flat between 12 and 16 |
| D17 | Pool health compared per account per cycle (contributed vs received) for the flagged cohort against everyone else, plus the planted segments | Shares of the pool only | Shares hide that the cohort is small; per-account flows show the size of the transfer (6.5x vs 1.6x received per credit contributed) | H | - |
| D18 | Charts follow one spec (`src/fraud_analytics/charts.py`): validated palette in fixed order, ordinal blue ramp for the ordered +1/+2/+3% levels, dumbbell for discovery vs holdout, emphasis on the total in the cost trade-off, solid hairline references, legend plus selective direct labels, a table next to every chart, plain-text fallbacks for every rich output | matplotlib defaults; `display(Markdown(...))` | Defaults read as unpolished; `Markdown` objects render as `<IPython.core.display.Markdown object>` in viewers without Markdown support (PyCharm) | H | - |
| D19 | Pool-health counterfactual re-simulates the programme without the exploiting segment | Recompute shares on the same run | Removing exploiters changes step spend and therefore every pro rata share; only a re-simulation captures that | M | Effect on regular accounts mis-sized |

## Synthetic programme

| # | Assumption | Value | Rationale | Confidence |
|---|---|---|---|---|
| A1 | Step length and maximum cycle | 4 hours; 42 steps (7 days) | Coarse resolution keeps the simulation vectorised; a week is a natural cycle | L (invented) |
| A2 | Payout chance at cycle step a | (a / 42)^4, so it always pays by step 42 | Rises slowly then sharply; most cycles pay between steps 20 and 35 | L (invented) |
| A3 | Pool | 900 credits seed per cycle + 2% of spend | Sized so the late strategy has a moderate long-run edge (1.07). A larger pool (tried: 1,500 + 3%, edge 1.19) puts +1/+2/+3% so far below the expected return that their time-to-edge curves are indistinguishable; at 1.07 the three levels separate clearly (90% reached after 206 / 295 / 457 cycles) | L (invented) |
| A4 | Pool sharing | Pro rata to the spend of the paying step | Cashback-style; removes single-winner noise so returns reflect strategy, not luck | M |
| A5 | Base cashback | Mean 0.85 per credit, gamma shape 2 | Below 1 so the pool is the only source of excess | L (invented) |
| A6 | Population | 4,000 accounts: 3% exploiting, 5% heavy legitimate, 92% regular | Small positive class, realistic for a review queue | L (invented) |
| A7 | Regular activity | Gamma-distributed rate, mean 0.06 entries per step, shape 1.5; log-normal spend (log-mean 2.0, log-sd 0.6) | Heterogeneous, right-skewed | L (invented) |
| A8 | Exploiting activity | 0.004 per step before a personal start phase (uniform 0.55-0.80 of the max cycle), 0.35 after; 2.5x spend | Strong but imperfect timing | L (invented) |
| A9 | Heavy legitimate activity | 4x rate, rising mildly with cycle phase (sensitivity 1.0) | The main false-positive risk | L (invented) |
| A10 | Horizon | One year (2,190 steps, 84 complete cycles) | Enough cycles for a discovery/holdout split and N up to 30 | M |

## Not done (and why)

- **Hazard curves of entry timing** (used in the source study): replaced by the binomial late-share test, which is simpler to audit. A hazard model is the natural extension if exploiters' windows drift over time.
- **Seed sweep**: one seed is reported. A sweep belongs in the shared generator (Repo 0) once it exists.
- **Dashboard**: Module B of Repo 2, out of scope here.
