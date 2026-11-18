from __future__ import annotations

import warnings
from collections.abc import Mapping

import numpy as np
import xarray as xr

from spacetimer.core import _PKG, Data, VarSpec, _specs

PASCAL = frozenset({"Pa", "pascal", "pascals"})


def round_mantissa(values: np.ndarray, *, keepbits: int) -> np.ndarray:
    """Round floats to ``keepbits`` mantissa bits, half to even. NaN is kept.

    Same arithmetic as numcodecs BitRound, applied to the values before writing
    so the store needs no extra codec (JS readers can't decode BitRound).
    """
    nmant = _nmant(values.dtype)
    if not 0 <= keepbits <= nmant:
        raise ValueError(f"keepbits must be in [0, {nmant}], got {keepbits}")
    # the bit view below assumes native byte order (zarr v2 can return >f4)
    values = values.astype(values.dtype.newbyteorder("="), copy=False)
    if keepbits == nmant:
        return values.copy()
    uint = np.dtype(f"u{values.dtype.itemsize}").type
    drop = nmant - keepbits
    bits = np.ascontiguousarray(values).view(uint)
    half = uint((1 << (drop - 1)) - 1)
    mask = ~uint((1 << drop) - 1)
    rounded = ((bits + ((bits >> uint(drop)) & uint(1)) + half) & mask).view(
        values.dtype
    )
    return np.where(np.isnan(values), values, rounded)


def apply_keepbits(ds: xr.Dataset, *, keepbits: Mapping[str, int]) -> xr.Dataset:
    """Return ``ds`` with each named variable rounded; lazy for dask-backed data."""
    for name, k in keepbits.items():
        if name not in ds.data_vars:
            raise KeyError(f"keepbits names unknown variable {name!r}")
        nmant = _nmant(ds[name].dtype)
        if not 0 <= k <= nmant:
            raise ValueError(f"keepbits for {name!r} must be in [0, {nmant}], got {k}")
    return ds.assign(
        {
            name: xr.apply_ufunc(
                round_mantissa,
                ds[name],
                kwargs={"keepbits": k},
                dask="parallelized",
                output_dtypes=[ds[name].dtype],
                keep_attrs=True,
            )
            for name, k in keepbits.items()
        }
    )


def convention_keepbits(data: Data) -> dict[str, int]:
    """dynamical.org reformatters mantissa bits: wind 6, precipitation 8, Pa 11.

    Only variables the table names are returned. Mantissa bits are relative
    precision (7 bits moves 288.15 K to 288.0), so the rest are yours to choose.
    """
    out: dict[str, int] = {}
    unmatched: list[str] = []
    for s in _specs(data):
        if not np.issubdtype(s.dtype, np.floating):
            continue
        bits = _convention(s)
        if bits is None:
            unmatched.append(s.name)
        else:
            out[s.name] = min(bits, _nmant(s.dtype))
    if unmatched:
        warnings.warn(
            f"no keepbits convention for {unmatched}; they are left unrounded "
            "unless you add them to the dict",
            skip_file_prefixes=_PKG,
        )
    return out


def _convention(spec: VarSpec) -> int | None:
    text = " ".join(
        str(s).lower()
        for s in (
            spec.name,
            spec.attrs.get("standard_name"),
            spec.attrs.get("long_name"),
        )
        if s
    )
    if "direction" in text:  # an angle, not a speed
        return None
    if "wind" in text:
        return 6
    if "precip" in text:
        return 8
    if spec.attrs.get("units") in PASCAL:
        return 11
    return None


def _nmant(dtype: np.dtype) -> int:
    if not np.issubdtype(dtype, np.floating):
        raise TypeError(f"keepbits needs a float dtype, got {dtype}")
    return int(np.finfo(dtype).nmant)
