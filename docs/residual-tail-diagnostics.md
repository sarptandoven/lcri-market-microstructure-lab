# Residual Tail Diagnostics

LCRI is most interesting in the tails.

Small residuals are useful for calibration, but the research question for signal delivery is whether extreme residual pressure behaves differently from normal book noise. Tail diagnostics summarize that behavior without turning the metric into a trading rule.

For each threshold, the diagnostic reports positive and negative tails:

```text
positive tail: lcri >= threshold
negative tail: lcri <= -threshold
```

Each bucket records:

- row count
- directional hit rate
- mean future return in ticks when available

This makes it easier to inspect whether large residuals are symmetric, whether one side has better transmission, and whether extreme pressure is actually useful after liquidity conditioning.

Research uses:

- compare raw LCRI tails against transmission-pressure tails
- inspect whether shadow absorption removes bad tail events
- track whether queue reversal risk concentrates inside failed tails
