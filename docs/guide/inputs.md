---
hide:
  - navigation
---

[< Level select](../usage.md){ .st-back }

# Inputs { data-stage="1-1" }

[`recommend`](../api.md#spacetimer.recommend) and [`report`](../api.md#spacetimer.report) accept any of these inputs, along with a use case (`"time"`, `"space"`, `"web"`, or `"balanced"`).

=== "Xarray"

    ```python
    import spacetimer

    encoding = spacetimer.recommend(ds, "space")
    ds.to_zarr(store, encoding=encoding)
    ```

    `spacetimer` only reads metadata (dimension names, shapes, dtypes, and attributes), so no data is loaded. For a single `DataArray`, pass `da.to_dataset()`.

    Importing `spacetimer` also adds a `ds.spacetimer` accessor with the same calls: `ds.spacetimer.recommend("space")`, `ds.spacetimer.report(encoding)`, and `ds.spacetimer.bitround()`.

=== "Zarr"

    Pass a `zarr.Group` to compare an existing store with a use case:

    ```python
    import zarr
    import spacetimer

    group = zarr.open_group("data.zarr", mode="r")
    spacetimer.report(group, spacetimer.recommend(group, "time"))
    ```

=== "Icechunk"

    An Icechunk session's store opens as a `zarr.Group`, so the call is the same:

    ```python
    import zarr
    import spacetimer

    group = zarr.open_group(session.store, mode="r")
    spacetimer.report(group, spacetimer.recommend(group, "time"))
    ```

=== "Planning"

    To pick a layout before the data exists, describe each variable's name, dims, shape, and dtype with a [`VarSpec`](../api.md#spacetimer.VarSpec):

    ```python
    import numpy as np
    import spacetimer
    from spacetimer import VarSpec

    spec = VarSpec(
        "t2m", ("time", "latitude", "longitude"), (73000, 721, 1440), np.dtype("f4")
    )
    spacetimer.report([spec], spacetimer.recommend([spec], "time"))
    ```

    If you plan to append to a dataset, describe its final shape here instead of passing the first few time steps.
