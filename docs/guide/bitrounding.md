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

`bitround()` with no arguments uses the bits dynamical.org chose for each variable in their [reformatters](https://github.com/dynamical-org/reformatters) datasets. `convention_keepbits(ds)` returns the same choices as a dict. Variables are matched on CF `standard_name` first:

| Mantissa bits | `standard_name` |
| --- | --- |
| 6 | Wind components, `wind_speed`, `wind_speed_of_gust` |
| 7 | Temperatures (in °C only), precipitation, humidity, cloud cover, radiation fluxes, snow, soil moisture, runoff, CAPE |
| 8 | `convective_precipitation_flux`, `snowfall_flux`, NDVI |
| 10 | `air_pressure` |
| 11 | `surface_air_pressure`, `air_pressure_at_mean_sea_level`, `geopotential_height` |
| 12 | `wind_from_direction` |

The full table is `spacetimer.bitround.KEEPBITS`. Where dynamical.org used different values across datasets, it takes the most common one, and the higher one on a tie.

Without a known `standard_name`, "wind" (6), "precip" (7), or "direction" (12) in the name or `long_name` matches, as do pressure `units` (`Pa`, `hPa`, `kPa`, `mbar`: 11).

Other float variables, and temperatures in kelvin, are not rounded unless you add them yourself. A warning lists them.

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

dynamical.org's 7 bits for temperature assume °C. This is why `spacetimer` skips temperatures in kelvin and does not round variables it can't identify.

To choose a value, decide how much error you can accept at the variable's typical magnitude. Then check the rounded data before writing.
