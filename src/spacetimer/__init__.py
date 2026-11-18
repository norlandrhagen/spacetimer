from importlib.metadata import PackageNotFoundError, version

from spacetimer.bitround import apply_keepbits, convention_keepbits, round_mantissa
from spacetimer.core import (
    PRESETS,
    Costs,
    Encoding,
    Preset,
    Report,
    SizeBasis,
    UseCase,
    VarSpec,
    recommend,
    report,
)

try:
    __version__ = version("spacetimer")
except PackageNotFoundError:
    __version__ = "0.0.0+dev"

__all__ = [
    "PRESETS",
    "Costs",
    "Encoding",
    "Preset",
    "Report",
    "SizeBasis",
    "UseCase",
    "VarSpec",
    "apply_keepbits",
    "convention_keepbits",
    "recommend",
    "report",
    "round_mantissa",
]
