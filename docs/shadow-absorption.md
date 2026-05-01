# Shadow Absorption

Residual imbalance is not automatically transmitted pressure.

A large positive LCRI can mean buyers are overwhelming the local liquidity baseline. It can also mean passive sellers are absorbing the displayed pressure before it becomes a price move. Shadow absorption is a research layer for separating those states.

The mechanism combines four pieces of state:

```text
pressure              = lcri
memory_alignment      = sign(lcri) * pressure_memory
fracture_drag         = max(-(sign(lcri) * fracture_memory), 0)
shadow_absorption     = decay_risk + fracture_drag + max(abs(lcri) - memory_alignment, 0)
transmission_pressure = lcri / (1 + shadow_absorption)
```

Interpretation:

- low absorption means residual pressure is persistent and aligned with book structure
- high absorption means displayed pressure is decaying, opposed by fracture, or not supported by memory
- transmission pressure is the pressure left after the absorption haircut

This is designed for publishability, not just prediction. A signal with high raw LCRI but high absorption should be harder to publish because the book is showing evidence that the pressure is being absorbed.

The first implementation labels each row as:

```text
transmitted
contested
absorbed
```

The next research step is to compare publishable-edge hit rate by absorption regime.
