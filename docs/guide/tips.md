---
hide:
  - navigation
---

[< Level select](../usage.md){ .st-back }

# Tips { data-stage="2-3" }

!!! warning "Dask-backed datasets"
    A dataset from `xr.open_zarr` is backed by Dask, and `to_zarr` refuses to write when the Dask chunks don't line up with the shards. Rechunk to the shards first:

    ```python
    enc = encoding["t2m"]
    ds = ds.chunk(dict(zip(ds["t2m"].dims, enc.get("shards", enc["chunks"]))))
    ds.to_zarr(store, encoding=encoding)
    ```

## Where the numbers come from

### Small chunks are cheap now

Earthmover's [I/O-maxing tensors in the cloud](https://www.earthmover.io/blog/i-o-maxing-tensors-in-the-cloud/) benchmark found that Icechunk and obstore read about as fast with 800 KB chunks as with 50 MB chunks. It suggested 3-15 MB as a good range. Python-based fsspec backends were 2-10x slower, and slowest with many small chunks. `spacetimer` assumes you read with a Rust-based backend (obstore, Icechunk, or zarrs).

### Shards keep object counts down

Small chunks used to mean millions of objects, which are slow to list and hard on HPC file systems. A Zarr v3 shard holds many chunks in one object. Each shard has an index, so a reader can still fetch a single chunk. The index takes 16 bytes per chunk, so `spacetimer` limits a shard to 512 chunks (an 8 KiB index).

### `"time"`, `"space"`, and `"balanced"`

The sizes follow dynamical.org's [chunk and shard layout tool](https://github.com/dynamical-org/reformatters/blob/main/docs/chunk_shard_layout_tool.md): chunks of about 3.5 MiB compressed (2.5-6 MiB is fine), shards of 100-600 MiB, and square spatial tiles. Their tool assumes data compresses to 20% of its size, and `spacetimer` does the same by default.

`"time"` uses the 32x32 tiles from their analysis mode. `"space"` (one time step per chunk) and `"balanced"` (tiles sized so maps and series read alike) are `spacetimer`'s own shapes at the same chunk size. dynamical.org also has a forecast mode, which keeps all lead times of one `init_time` together. `spacetimer` does not implement it.

### `"web"`

Browser tiles are sized by the pixels they cover, not by compressed bytes. topozarr uses 512 KiB uncompressed tiles, about 362x362 float32 pixels, with 4x4 tiles per shard.

### Codec

Every use case compresses with Blosc (zstd, level 3, byte shuffle), the same as dynamical.org's production datasets.

## Credits

The chunk and shard sizes, the `"time"` use case, the cost table, and the bitrounding table come from [dynamical.org's reformatters](https://github.com/dynamical-org/reformatters). The `"web"` use case comes from [topozarr](https://github.com/carbonplan/topozarr). The case for small chunks comes from Earthmover's [I/O-maxing tensors in the cloud](https://www.earthmover.io/blog/i-o-maxing-tensors-in-the-cloud/).
