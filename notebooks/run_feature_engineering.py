"""
Production runner for feature_engineering_pipeline.
Loads the pipeline source, skips interactive execution blocks by content
(not line numbers), then calls run_feature_pipeline().

Strategy per %%-delimited cell:
  1. Scan all lines for the first occurrence of a SKIP_PATTERN that is not
     inside a def/class definition.
  2. Walk backward from that line to find the nearest module-level
     (unindented) non-comment line — this is the true start of the block.
  3. Comment out that line and everything after it in the cell.
  This preserves any constant definitions (e.g. FEATURES_DIR, SPLIT_OUTPUT_NAMES)
  that appear before the interactive loop in the same cell.

Interactive blocks suppressed:
  - STAGE 5-7 loop (split_results dict init + build loop)
  - split_results inspection loop
  - write-to-disk loop (constants kept, loop suppressed)
  - feat_cols inspection (daily.columns reference)
  - p95 plot-per-split loop
  - histogram/zone-map loop (base_images_dir used inside loop body)
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

pipeline_path = HERE / "feature_engineering_pipeline.py"
source = pipeline_path.read_text(encoding="utf-8")

_SKIP_PATTERNS: tuple[str, ...] = (
    "split_results",
    "daily.columns",
    "base_images_dir",
)


def _sanitize_cell(cell: str) -> str:
    """Comment out the interactive block in a cell, keeping prior constants.

    Finds the first line (at any indentation) containing a skip pattern that
    is not a function/class definition. Walks backward to the nearest
    module-level non-comment line, then comments out from there to end of cell.

    Args:
        cell: Raw cell text (everything after a # %% delimiter).

    Returns:
        Sanitized cell text safe to exec in production.
    """
    lines = cell.splitlines(keepends=True)

    trigger_idx: int | None = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("def ") or stripped.startswith("class "):
            continue
        if any(pat in stripped for pat in _SKIP_PATTERNS):
            trigger_idx = i
            break

    if trigger_idx is None:
        return cell

    skip_from = trigger_idx
    for i in range(trigger_idx, -1, -1):
        line = lines[i]
        stripped = line.strip()
        is_module_level = not line.startswith((" ", "\t"))
        if is_module_level and stripped and not stripped.startswith("#"):
            if stripped.startswith("def ") or stripped.startswith("class "):
                return cell
            skip_from = i
            break

    result: list[str] = []
    for i, line in enumerate(lines):
        result.append(("# [skipped] " + line) if i >= skip_from else line)

    return "".join(result)


raw_cells = source.split("# %%")
patched = "# %%".join(_sanitize_cell(c) for c in raw_cells)

os.chdir(HERE)

ns: dict = {"__file__": str(pipeline_path), "__name__": "__main__"}
exec(compile(patched, str(pipeline_path), "exec"), ns)
