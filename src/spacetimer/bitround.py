from __future__ import annotations

import warnings
from collections.abc import Mapping

import numpy as np
import xarray as xr

from spacetimer.core import _PKG, Data, VarSpec, _specs

# keep_mantissa_bits by CF standard_name across dynamical.org reformatters'
# template configs (commit cb7d15b): the most common value, ties to the higher
KEEPBITS: dict[str, int] = {
    **dict.fromkeys(
        (
            "eastward_wind",
            "northward_wind",
            "x_wind",
            "y_wind",
            "wind_speed",
            "wind_speed_of_gust",
        ),
        6,
    ),
    **dict.fromkeys(
        (
            "air_temperature",
            "dew_point_temperature",
            "sea_surface_temperature",
            "soil_temperature",
            "surface_temperature",
            "precipitation_flux",
            "runoff_flux",
            "surface_runoff_flux",
            "relative_humidity",
            "specific_humidity",
            "cloud_area_fraction",
            "cloud_area_fraction_in_atmosphere_layer",
            "sea_ice_area_fraction",
            "surface_snow_area_fraction",
            "surface_snow_density",
            "surface_snow_thickness",
            "lwe_thickness_of_surface_snow_amount",
            "volume_fraction_of_condensed_water_in_soil",
            "atmosphere_mass_content_of_water",
            "atmosphere_mass_content_of_water_vapor",
            "atmosphere_convective_available_potential_energy",
            "lagrangian_tendency_of_air_pressure",
            "surface_downward_eastward_stress",
            "surface_downward_northward_stress",
            "surface_downwelling_shortwave_flux_in_air",
            "surface_downwelling_longwave_flux_in_air",
            "surface_diffuse_downwelling_shortwave_flux_in_air",
            "surface_direct_downwelling_shortwave_flux_in_air",
            "surface_downward_latent_heat_flux",
            "surface_downward_sensible_heat_flux",
            "surface_net_downward_longwave_flux",
            "surface_net_downward_shortwave_flux",
            "toa_net_downward_longwave_flux",
            "equivalent_reflectivity_factor",
        ),
        7,
    ),
    **dict.fromkeys(
        (
            "convective_precipitation_flux",
            "snowfall_flux",
            "normalized_difference_vegetation_index",
        ),
        8,
    ),
    "air_pressure": 10,
    "surface_air_pressure": 11,
    "air_pressure_at_mean_sea_level": 11,
    "geopotential_height": 11,
    "wind_from_direction": 12,
}
# their temperatures are in degC; bits are relative, so K needs about 5 more
CELSIUS_ONLY = frozenset(n for n in KEEPBITS if n.endswith("temperature"))
CELSIUS = frozenset({"degree_Celsius", "degrees_Celsius", "degC", "celsius", "°C"})
# bits are relative precision, so any pressure unit takes the same value
PRESSURE = frozenset({"Pa", "pascal", "pascals", "hPa", "kPa", "mbar", "millibar"})


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
    """Mantissa bits from dynamical.org's per-variable choices (``KEEPBITS``).

    Matched on CF ``standard_name``, then on "wind", "precip", or "direction" in
    the name or ``long_name``, or pressure ``units``. Temperatures match only in
    degC: mantissa bits are relative precision, so 7 bits moves 288.15 K to
    288.0. Unmatched floats are left out and listed in a warning.
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
    # str(): zarr attrs can hold any JSON value
    units, sn = (str(spec.attrs.get(k, "")) for k in ("units", "standard_name"))
    if sn in KEEPBITS:
        if sn in CELSIUS_ONLY and units not in CELSIUS:
            return None
        return KEEPBITS[sn]
    # no known standard_name: match on names, at the table's values
    text = " ".join(
        str(s).lower() for s in (spec.name, sn, spec.attrs.get("long_name")) if s
    )
    if "direction" in text:
        return KEEPBITS["wind_from_direction"]
    if "wind" in text:
        return KEEPBITS["eastward_wind"]
    if "precip" in text:
        return KEEPBITS["precipitation_flux"]
    if units in PRESSURE:
        return KEEPBITS["surface_air_pressure"]
    return None


def _nmant(dtype: np.dtype) -> int:
    if not np.issubdtype(dtype, np.floating):
        raise TypeError(f"keepbits needs a float dtype, got {dtype}")
    return int(np.finfo(dtype).nmant)
