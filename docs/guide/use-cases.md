---
hide:
  - navigation
---

[< Level select](../usage.md){ .st-back }

# Use cases { data-stage="1-2" }

A use case is the access pattern you expect readers to use most. It decides how the data is chunked.

<div class="st-presets" markdown>

| | Use case | Best for | Chunk size | Chunks per shard |
| --- | --- | --- | --- | --- |
| <img src="../../assets/sprites/icon-time.svg" alt="" height="32"> | `"time"` | Time series at a point | ~3.5 MiB compressed, 32x32 spatial tiles | 8 |
| <img src="../../assets/sprites/icon-space.svg" alt="" height="32"> | `"space"` | Full maps at one time step | up to ~3.5 MiB compressed, one time step | 8 |
| <img src="../../assets/sprites/icon-balanced.svg" alt="" height="32"> | `"balanced"` | A mix of maps and time series | ~3.5 MiB compressed, tiles sized per dataset | 8 |
| <img src="../../assets/sprites/icon-web.svg" alt="" height="32"> | `"web"` | Map tiles in a browser (e.g. [topozarr](https://github.com/carbonplan/topozarr)) | 512 KiB uncompressed | 4 |

</div>

You can also pass a [`UseCase`](../api.md#spacetimer.UseCase) enum value. All other options must be passed by name.

`"balanced"` picks the tile edge for each variable so that one map and one time series read about the same number of chunks. At a fixed chunk size, the two counts multiply to a constant, so making them equal also makes the slower of the two as fast as it can be. ERA5 hourly (73,000 x 721 x 1440) gets 81x85 tiles: 153 chunks per map and 110 per series. A 720-step dataset gets 241x240 tiles. The edge is never smaller than `"time"`'s 32.

With a Dataset, `recommend` warns when a time chunk spans more than a year, the limit in [dynamical.org's layout tool](https://github.com/dynamical-org/reformatters/blob/cb7d15bfe5dfba501ed383041d141444d6ee7009/src/scripts/chunk_shard_size.py#L330-L339). It does not cap them: for daily data a one-year chunk is well under the 3.5 MiB target. A `zarr.Group` gets no warning, since its time step needs a data read.
