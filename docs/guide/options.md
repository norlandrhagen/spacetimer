---
hide:
  - navigation
---

[< Level select](../usage.md){ .st-back }

# Options { data-stage="2-1" }

## Sharding

A shard is one stored object that holds many chunks. `chunks_per_shard` sets how many chunks a shard holds along each spatial dimension. If the shard has room left, `spacetimer` fills it along the other dimensions, such as time. For example, a `"space"` layout with 721x1440 maps puts 282 time steps in one object instead of writing one object per time step.

```python
spacetimer.recommend(ds, "web", chunks_per_shard=8)  # bigger shards
spacetimer.recommend(ds, "web", sharding=False)  # one object per chunk
```

A shard never holds more than 512 chunks. If `chunks_per_shard` would go over that, `spacetimer` lowers it and warns you.

The last shard along a dimension can extend past the edge of the array by less than one chunk. A 721-pixel dimension in 361-pixel chunks becomes one 722-pixel shard, not two.

## Chunk size

The `"time"`, `"space"`, and `"balanced"` chunk sizes are compressed sizes. `compression_ratio` converts them to an uncompressed size. The default is `0.2`, which means the data compresses to 20% of its size. If your data only compresses to half its size, pass `0.5` and the chunks get smaller.

`"web"` is sized in uncompressed bytes, so the ratio does not change it.

To set the size yourself, pass `raw_chunk_bytes`, the uncompressed size in bytes. Chunks never go over it.

## Dimension names

`spacetimer` finds spatial dimensions by name: `x`, `y`, `lat`, `lon`, `latitude`, or `longitude`. For a Dataset or Zarr group, it also reads the CF metadata on coordinates. A coordinate with `axis` X or Y, a latitude, longitude, or projection `standard_name`, or `units` of degrees north or east marks its dimensions as spatial. This finds WRF's `south_north` and `west_east` through their 2-D `XLAT`/`XLONG` coordinates. The time dimension is found the same way: `axis` T, `standard_name` time, `units` like "hours since 2000-01-01", or a datetime coordinate. Otherwise any dimension with "time" in its name is used.

Every variable with a spatial dimension gets an entry in the encoding. Variables without one keep Xarray's defaults, and a warning lists them. `"time"` and `"balanced"` also warn if they can't find a time dimension.

If detection misses, pass `spatial_dims=("row", "col")`, which replaces both the names and the metadata detection. For time, pass `time_dim="valid_time"`.

## Compression

By default, every use case compresses with Blosc (zstd, level 3, byte shuffle). To use something else, pass `compressors`:

```python
from zarr.codecs import ZstdCodec

encoding = spacetimer.recommend(ds, "time", compressors=[ZstdCodec(level=5)])
```

Changing the codec does not change the chunk sizes. If your codec compresses much more or less than 20%, pass a matching `compression_ratio` too.

The result is a plain dict, so you can also edit any entry before writing, for example `encoding["t2m"]["compressors"] = [ZstdCodec(level=5)]`.
