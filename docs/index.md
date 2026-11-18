---
template: home.html
title: Home
hide:
  - navigation
  - toc
---

Chunk, shard, and codec recommendations for Zarr
{ #start .st-stage-line }

<div class="st-console" markdown>
<div class="st-console__elbow"></div>
<div class="st-console__top"><span>Mission select</span><i></i><i></i></div>
<div class="st-console__rail"><span>Strat</span><span>Grid</span><span>Zarr</span></div>
<div class="st-console__screen" markdown>

=== "Space"

    <span class="st-sprite st-sprite--space" aria-hidden="true"></span>
    Whole maps per chunk. Fast for "all space, one time step."

    ```python
    import spacetimer

    encoding = spacetimer.recommend(ds, "space")
    ds.to_zarr(store, encoding=encoding)
    spacetimer.report(ds, encoding)
    ```

    ```text
    | var | chunks      | shards       | objects | slice chunks | series chunks |
    | --- | ----------- | ------------ | ------- | ------------ | ------------- |
    | t2m | 1x721x1440  | 282x721x1440 | 259     | 1            | 73,000        |
    ```

=== "Time"

    <span class="st-sprite st-sprite--time" aria-hidden="true"></span>
    Long time series per chunk. Fast for "this point, all time."

    ```python
    import spacetimer

    encoding = spacetimer.recommend(ds, "time")
    ds.to_zarr(store, encoding=encoding)
    spacetimer.report(ds, encoding)
    ```

    ```text
    | var | chunks      | shards       | objects | slice chunks | series chunks |
    | --- | ----------- | ------------ | ------- | ------------ | ------------- |
    | t2m | 4480x32x32  | 4480x256x256 | 306     | 1,035        | 17            |
    ```

=== "Balanced"

    <span class="st-sprite st-sprite--balanced" aria-hidden="true"></span>
    Halfway between Time and Space: a map and a time series read about the same number of chunks.

    ```python
    import spacetimer

    encoding = spacetimer.recommend(ds, "balanced")
    ds.to_zarr(store, encoding=encoding)
    spacetimer.report(ds, encoding)
    ```

    ```text
    | var | chunks      | shards       | objects | slice chunks | series chunks |
    | --- | ----------- | ------------ | ------- | ------------ | ------------- |
    | t2m | 666x81x85   | 666x648x680  | 660     | 153          | 110           |
    ```

=== "Web"

    <span class="st-sprite st-sprite--web" aria-hidden="true"></span>
    Small tiles per chunk. Fast for "pan and zoom on a map."

    ```python
    import spacetimer

    encoding = spacetimer.recommend(ds, "web")
    ds.to_zarr(store, encoding=encoding)
    spacetimer.report(ds, encoding)
    ```

    ```text
    | var | chunks      | shards       | objects | slice chunks | series chunks |
    | --- | ----------- | ------------ | ------- | ------------ | ------------- |
    | t2m | 1x361x360   | 2x722x1440   | 36,500  | 8            | 73,000        |
    ```

</div>
</div>



## Install

=== "uv"

    ```bash
    uv add git+https://github.com/norlandrhagen/spacetimer
    ```

=== "pip"

    ```bash
    pip install git+https://github.com/norlandrhagen/spacetimer
    ```
