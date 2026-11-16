import hypothesis.extra.numpy as hnp
import numcodecs
import numpy as np
import pytest
import xarray as xr
from hypothesis import given
from hypothesis import strategies as st

from spacetimer import apply_keepbits, round_mantissa


@given(
    st.sampled_from(["f2", "f4", "f8"]).flatmap(
        lambda d: st.tuples(
            hnp.arrays(d, st.integers(0, 50)),
            st.integers(0, np.finfo(d).nmant - 1),
        )
    )
)
def test_round_mantissa_matches_numcodecs(case):
    values, keepbits = case
    out = round_mantissa(values, keepbits=keepbits)
    nan = np.isnan(values)
    assert np.isnan(out[nan]).all()  # numcodecs can round a NaN to inf; we keep it
    want = np.frombuffer(
        numcodecs.BitRound(keepbits=keepbits).encode(values.copy()), values.dtype
    )
    np.testing.assert_array_equal(out[~nan], want[~nan])
    np.testing.assert_array_equal(round_mantissa(out, keepbits=keepbits), out)


def test_round_mantissa_identity_and_errors():
    values = np.float32([1.2345678])
    np.testing.assert_array_equal(round_mantissa(values, keepbits=23), values)
    with pytest.raises(ValueError):
        round_mantissa(values, keepbits=24)
    with pytest.raises(TypeError):
        round_mantissa(np.int32([1]), keepbits=3)
    native = np.float32([1.2345678, -3.14159])
    np.testing.assert_array_equal(
        round_mantissa(native.astype(">f4"), keepbits=7),
        round_mantissa(native, keepbits=7),
    )


def test_apply_keepbits_errors():
    ds = xr.Dataset({"v": ("x", np.float32([1.0])), "m": ("x", np.uint8([1]))})
    with pytest.raises(KeyError):
        apply_keepbits(ds, keepbits={"missing": 3})
    with pytest.raises(ValueError):
        apply_keepbits(ds, keepbits={"v": 30})
    with pytest.raises(TypeError):
        apply_keepbits(ds, keepbits={"m": 3})
