---
hide:
  - navigation
---

[< Level select](../usage.md){ .st-back }

# Reading the report { data-stage="1-3" }

`report` prints one row per variable. It works on any encoding, including one you wrote by hand.

| Column | Meaning |
| --- | --- |
| `chunks` | Chunk shape |
| `shards` | Shard shape |
| `objects` | Number of stored objects (shards, or chunks without sharding) |
| `slice chunks` | Chunks read to load one full map at one time step |
| `series chunks` | Chunks read to load the full time series at one point |

## In a notebook

End a cell with the report, or call `ds.spacetimer.report(encoding)`. In a terminal you get the Markdown table above instead.

```python
spacetimer.report(ds, spacetimer.recommend(ds, "time"))
```

The report is one collapsible row per variable, like the Xarray repr; the first row starts open. A row shows array, chunk, and shard sizes beside a cube of the chunk grid, and what one map or one time series reads. `size` is uncompressed, like the Xarray repr; read sizes apply `compression_ratio`. `show shards` redraws the cube with shard boundaries, and `encoding` shows what to pass to `to_zarr`. Axes with many chunks draw every n-th line; shard lines always fall on chunk lines, and shards draw fewer lines than chunks.

The "one time series" line needs to know which axis is time. It looks for a dim with `time` in its name; if yours is called something else, pass `time_dim="..."`.

=== "time"

    --8<-- "docs/assets/report-time.html"

=== "space"

    --8<-- "docs/assets/report-space.html"

=== "balanced"

    --8<-- "docs/assets/report-balanced.html"

=== "web"

    --8<-- "docs/assets/report-web.html"

## In a script

Use `.costs` for the numbers, or `.to_pandas()` for a DataFrame. The notebook view shows one report; to compare use cases side by side, use a table:

```python
import pandas as pd

pd.concat(
    {
        uc: spacetimer.report(ds, spacetimer.recommend(ds, uc)).to_pandas()
        for uc in ("time", "space", "balanced", "web")
    },
    names=["use_case"],
)
```
