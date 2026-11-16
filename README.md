# spacetimer

Chunk and shard recommendations for Zarr, based on how people will read your data: time series, maps, a mix of both, or web tiles. It reads only metadata and returns an `encoding` dict for `to_zarr`.

## Install

```bash
uv add git+https://github.com/norlandrhagen/spacetimer
```

## Usage

```python
import spacetimer

encoding = spacetimer.recommend(ds, "time")  # or "space", "balanced", "web"
spacetimer.report(ds, encoding)  # chunks read per map and per time series
ds.to_zarr("out.zarr", encoding=encoding)
```

See the [docs](https://norlandrhagen.github.io/spacetimer/) for options, bitrounding, and how to read the report.

## License

[MIT](LICENSE)
