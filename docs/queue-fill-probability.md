# Queue-fill probability

Directional residual imbalance is not enough to make a signal tradable. A passive order can be correct about the next mid move and still fail to execute, or it can execute only when the queue is being depleted against it. This module adds a transparent snapshot-level execution proxy to separate signal quality from tradability.

The first implementation is deliberately conservative. The lab currently consumes L2 snapshots, not event-level add, cancel, and trade messages. Queue position is therefore estimated from visible top-of-book depth:

```text
bid_queue_ahead = queue_position_fraction * bid_sz_1
ask_queue_ahead = queue_position_fraction * ask_sz_1
```

The queue-ahead values are normalized by same-side visible depth to produce queue-share penalties. A deeper position in the visible queue reduces passive fill probability even when residual pressure points toward depletion.

The execution proxy can now make that penalty child-order-size-aware with `add_queue_position_order_size_features`:

```text
bid_child_order_size = order_size_fraction * bid_sz_1  # or explicit bid order size
ask_child_order_size = order_size_fraction * ask_sz_1  # or explicit ask order size
bid_queue_clear_size = bid_queue_ahead + bid_child_order_size
ask_queue_clear_size = ask_queue_ahead + ask_child_order_size
```

Those clearance sizes are normalized as `bid_queue_clear_share` and `ask_queue_clear_share`. When the clearance-share columns are present, `add_passive_fill_probabilities` uses them instead of the raw queue-ahead share, so larger child orders receive lower passive full-fill probability even at the same queue position and pressure state. The realized-fill labelers also honor the clearance sizes: `add_queue_position_realized_fill_proxy` and `add_event_level_realized_fill_proxy` require visible or event-level queue depletion to clear queue-ahead plus child size before marking a full passive fill. This is a snapshot-level capacity guardrail: it distinguishes “can get a token lot filled” from “can clear a realistic child order before the book moves.”

## Passive fill logic

For a passive buy resting at the bid, fill probability increases when sell pressure is likely to deplete the bid:

```text
bid_depletion_pressure = -LCRI + spread stress
```

For a passive sell resting at the ask, fill probability increases when buy pressure is likely to deplete the ask:

```text
ask_depletion_pressure = LCRI + spread stress
```

Both sides include volatility stress, replenishment state, and a queue-share penalty. The output columns are:

- `bid_fill_probability`
- `ask_fill_probability`
- `bid_adverse_fill_probability`
- `ask_adverse_fill_probability`
- `fill_probability_imbalance`
- `passive_fill_regime`

## Execution-adjusted edge

The execution layer combines transaction-cost-aware net returns with fill and adverse-fill probabilities:

```text
long_fill_adjusted_edge = bid_fill_probability * long_net_return_ticks
                          - bid_adverse_fill_probability * abs(long_net_return_ticks)

short_fill_adjusted_edge = ask_fill_probability * short_net_return_ticks
                           - ask_adverse_fill_probability * abs(short_net_return_ticks)
```

The best positive side becomes `best_execution_side`. If both fill-adjusted edges are negative, the row abstains even when the directional LCRI probability is confident.

`execution_adjusted_edge_summary` turns the row-level edge into a compact evaluation artifact with tradable/abstain share, mean and median execution-adjusted edge, side dominance, average fill/adverse-fill probabilities, and optional conflicts against the pre-execution `publishable_side` gate. The conflict rate is the key execution-aware publishability diagnostic: it identifies cases where a signal cleared probability and transaction-cost gates but becomes non-tradable, or flips side, once visible queue position and adverse-fill selection are applied.

`execution_publishability_review_packet` expands that conflict rate into a publishable-vs-execution cross-tab for release review. It prioritizes rows where pre-execution `long`/`short` signals become `abstain` or flip side after queue/adverse-fill adjustment, and reports average best fill probability, adverse-fill probability, pre-execution side fill probability, execution-adjusted edge, and edge drag. The demo writes this packet for both full-sample and heldout slices as:

- `execution_publishability_review_packet.csv`
- `heldout_execution_publishability_review_packet.csv`

These artifacts make the publishability gate execution-aware instead of only probability/cost-aware: reviewers can see whether nominal alpha survives visible queue position and adverse selection, not just whether it predicts direction.

## Passive fill edge curve

`passive_fill_edge_curve` provides a small calibration surface for API users and research dashboards. It keeps only rows where `best_execution_side` is `long` or `short`, selects the side-appropriate predicted fill probability, adverse-fill probability, and realized net return, then bins tradable rows by predicted fill quality.

The returned table has stable columns:

- `bin`
- `rows`
- `long_rows`
- `short_rows`
- `mean_predicted_fill_probability`
- `mean_adverse_fill_probability`
- `mean_realized_edge_ticks`
- `positive_edge_rate`
- `mean_execution_adjusted_edge_ticks`

A healthy prototype should usually show higher predicted-fill buckets with non-deteriorating realized edge and an adverse-fill rate that does not swamp the fill benefit. This is not a formal matching-engine validation; it is a reviewable snapshot-level diagnostic that can later be calibrated against event-level fills.

```python
curve = passive_fill_edge_curve(execution_frame, bins=5)
```

## Queue-position toxicity surface

`queue_position_toxicity_surface` adds an adverse-selection check to queue-capacity claims. It keeps only tradable rows, selects side-appropriate bid/ask queue share, fill probability, adverse-fill probability, realized fill, and realized return, then bins each regime/side by queue depth. Each cell reports:

- `mean_predicted_fill_probability`
- `mean_adverse_fill_probability`
- `adverse_to_fill_ratio`
- `realized_fill_rate`
- `realized_loss_rate`
- `mean_realized_edge_ticks`
- `mean_execution_adjusted_edge_ticks`
- `queue_toxicity_label`

A cell is labeled `toxic_queue_fill` when adverse-fill probability is large relative to fill probability, the realized loss rate is high, or execution-adjusted edge is negative. This is the review artifact for the failure mode where a strategy appears to have enough passive capacity only because it gets filled mostly during adverse selection.

```python
toxicity = queue_position_toxicity_surface(execution_frame, queue_bins=5)
```

`queue_position_execution_readiness_scorecard` can now consume this surface via
`toxicity_surface=toxicity`. The readiness gate reports `queue_toxicity_label`,
`toxic_queue_row_share`, `toxic_queue_regimes`, `worst_toxicity_regime`, worst
adverse-to-fill/loss-rate severity, and row-weighted toxic-cell execution edge,
and blocks publication when the toxic-row share exceeds
`max_toxic_queue_row_share`. That keeps calibrated queue-capacity evidence from
passing review if its fills are concentrated in adverse-selection cells, while
also showing reviewers whether the block is driven by a broad mild toxicity band
or a small severe loss pocket.

## Latency edge survival by event regime

`queue_position_latency_edge_regime_surface` extends the aggregate latency edge-survival check into decision-time event-window regimes. It keeps the selected passive side and decision-time execution-adjusted edge fixed, then replays realized selected-side fills from later snapshot latencies inside each symbol/session group. Each regime/latency row reports candidate counts, realized fill rate, mean decision edge, realized edge ticks, edge gap versus immediate queue state, survival ratio, and an `edge_latency_regime_label`.

This catches a publishability failure mode that a fill-rate surface can miss: an event window may retain enough fills after one or two snapshots of latency, but lose the high-edge fills specifically. Reviewers can therefore isolate whether execution-adjusted LCRI survives queue-state staleness in the regimes where the signal claims to be tradable.

```python
edge_surface = queue_position_latency_edge_regime_surface(
    execution_frame,
    regime_col="passive_fill_event_window_regime",
    group_cols="symbol",
    latencies=(0, 1, 2, 5),
)
```

This changes the research question from:

```text
Does residual imbalance predict next-mid direction?
```

into:

```text
Does residual imbalance remain tradable after visible queue position, passive fill selection, adverse fills, transaction costs, and queue-state latency?
```

## Limitations

This is a snapshot proxy, not a full event-level matching-engine simulator. It does not observe order IDs, cancellations, hidden liquidity, queue priority loss, or trade prints. The purpose is to provide an execution-aware research surface that is compatible with the current normalized snapshot schema. A later tranche can replace or calibrate the proxy with event-level fill labels when real message data is available.
