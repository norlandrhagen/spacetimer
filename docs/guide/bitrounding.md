---
hide:
  - navigation
---

[< Level select](../usage.md){ .st-back }

# Bitrounding { data-stage="2-2" }

Most float32 data stores more precision than the measurement has. Bitrounding sets the extra bits to zero, which often cuts the compressed size in half. It is opt-in and separate from `recommend`. Nothing is rounded unless you call it:

```python
ds = ds.spacetimer.bitround()  # recommended bits, where there is a recommendation
ds.to_zarr(store, encoding=encoding)
```

The values are rounded before writing, not by a codec, so any Zarr reader can open the result, including JavaScript ones. Dask-backed data stays lazy.

## Recommended bits

`keepbits` is the number of mantissa bits to keep. The mantissa is the part of a float that holds its significant digits.

`bitround()` with no arguments uses dynamical.org's table. `convention_keepbits(ds)` returns the same table as a dict:

| Variable | Matched on | Mantissa bits |
| --- | --- | --- |
| Wind speed and components | "wind" in the name, `standard_name`, or `long_name` (not "direction") | 6 |
| Precipitation | "precip" in the same fields | 8 |
| Pressure | `units` of `Pa` | 11 |

Other float variables are not rounded unless you add them yourself. A warning lists them.

## Per variable

Pass your own dict, or extend the recommended one:

```python
from spacetimer import convention_keepbits

ds = ds.spacetimer.bitround(keepbits={"t2m": 10})
ds = ds.spacetimer.bitround(keepbits={**convention_keepbits(ds), "t2m": 10})
```

`apply_keepbits(ds, keepbits=...)` does the same thing as a function.

## Choosing a value

Keeping `n` bits stores a value to within 1 part in 2^(n+1). The error grows with the value, so this works well for values near zero and poorly for values with a large offset. At 7 bits, 15 °C is stored to within 0.03 °C, but 288.15 K becomes 288.0, an error of up to 1 K.

dynamical.org's default of 7 bits assumes their units (°C, and Pa at 11 bits). This is why `spacetimer` does not round variables it can't identify.

To choose a value, decide how much error you can accept at the variable's typical magnitude. Then check the rounded data before writing.
