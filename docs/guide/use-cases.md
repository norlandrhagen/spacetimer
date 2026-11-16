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
| <img src="../../assets/sprites/icon-balanced.svg" alt="" height="32"> | `"balanced"` | A mix of maps and time series | ~3.5 MiB compressed, 128x128 spatial tiles | 8 |
| <img src="../../assets/sprites/icon-web.svg" alt="" height="32"> | `"web"` | Map tiles in a browser (e.g. [topozarr](https://github.com/carbonplan/topozarr)) | 512 KiB uncompressed | 4 |

</div>

You can also pass a [`UseCase`](../api.md#spacetimer.UseCase) enum value. All other options must be passed by name.

The 128x128 tiles for `"balanced"` are a first guess between `"time"` and `"space"`. They have not been benchmarked yet.
