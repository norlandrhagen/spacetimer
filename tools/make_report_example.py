"""Write the Report HTML shown in docs/guide/reading-the-report.md.

python tools/make_report_example.py
"""

from pathlib import Path

import spacetimer
from spacetimer import VarSpec

specs = [
    VarSpec("t2m", ("time", "lat", "lon"), (8760, 721, 1440), "float32"),
    VarSpec("tp", ("time", "lat", "lon"), (8760, 721, 1440), "float32"),
]
out = Path(__file__).parent.parent / "docs" / "assets"


def write(name: str, r: spacetimer.Report) -> None:
    (out / f"report-{name}.html").write_text(r._repr_html_().replace("\n", " ") + "\n")


for use_case in ("time", "space", "balanced", "web"):
    write(use_case, spacetimer.report(specs, spacetimer.recommend(specs, use_case)))
