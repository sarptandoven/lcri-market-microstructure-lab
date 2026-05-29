# Queue-fill probability

Directional residual imbalance is not enough to make a signal tradable. A passive order can be correct about the next mid move and still fail to execute, or it can execute only when the queue is being depleted against it. This module adds a transparent snapshot-level execution proxy to separate signal quality from tradability.

The first implementation is deliberately conservative. The lab currently consumes L2 snapshots, not event-level add, cancel, and trade messages. Queue position is therefore estimated from visible top-of-book depth:

```text
bid_queue_ahead = queue_position_fraction * bid_sz_1
ask_queue_ahead = queue_position_fraction * ask_sz_1
```

The queue-ahead values are normalized by same-side visible depth to produce queue-share penalties. A deeper position in the visible queue reduces passive fill probability even when residual pressure points toward depletion.

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

This changes the research question from:

```text
Does residual imbalance predict next-mid direction?
```

into:

```text
Does residual imbalance remain tradable after visible queue position, passive fill selection, adverse fills, and transaction costs?
```

## Limitations

This is a snapshot proxy, not a full event-level matching-engine simulator. It does not observe order IDs, cancellations, hidden liquidity, queue priority loss, or trade prints. The purpose is to provide an execution-aware research surface that is compatible with the current normalized snapshot schema. A later tranche can replace or calibrate the proxy with event-level fill labels when real message data is available.
