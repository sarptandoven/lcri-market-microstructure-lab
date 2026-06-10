# Contributing

Preview Ladder DiT contributions should improve falsifiability, reproducibility, or product usability.

## Good first contributions

- Add a synthetic fixture with one isolated failure mode.
- Add a metric with a documented unit and lower-is-better direction.
- Add a benchmark adapter that writes the public report schema without requiring private services.
- Add a failure demo with source, mask, preview, final, and report JSON.

## Benchmark submission rules

1. Include the exact command, seed, model identifiers, model hashes when available, and hardware.
2. Submit the full metric vector, not only a scalar rank.
3. Include at least one failure grid per method.
4. Do not use private videos unless you can redistribute them with masks.
5. Label hand-picked demos as demos, not benchmark results.

## Review principle

A method that is slower but makes previews meaningfully more predictive can be a valid contribution. A method that only improves final visual quality while breaking preview-final consistency is out of scope unless it is clearly labeled as a baseline.
