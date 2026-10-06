# Python Coding & Editing Guidelines

> **Living document – PRs welcome!**
> Last updated: 2025‑07‑15

## Table of Contents

1. Philosophy
1. Docstrings & Comments
1. Type Hints
1. Documentation

---

## Philosophy

- **Readability, reproducibility, performance – in that order.**
- Prefer explicit over implicit; avoid hidden state and global flags.
- Measure before you optimize (`time.perf_counter`, `line_profiler`).
- Each module holds a **single responsibility**; keep public APIs minimal.

## Docstrings & Comments

- Style: NumPyDoc.
- Start with a one‑sentence summary in the imperative mood.
- Sections: Parameters, Returns, Raises, Examples, References.
- Use backticks for code or referring to variables (e.g. `xarray.DataArray`).
- Do not use emojis, or non-unicode characters in comments/print statements.
- Cite peer‑reviewed papers with DOI links when relevant.
- Write code that explains itself rather than needs comments.
- For the inline you do add, explain *why*, not what. For example, *don't* write:

```python
# open the file
f = open(filename)
```

- The comments should be things which are not obvious to a reader with typical background knowledge.

## Tools

- ruff is use for most code maintenance, black for formatting, mypy for type checking, pytest for testing
- You can run `pre-commit run -a` to run all pre-commit hooks and check for style violations

## Code Style

- Annotate all public functions (PEP 484).
- Prefer `Protocol` over `ABC`s when only an interface is needed.
- Validate external inputs via Pydantic models (if existing); otherwise, use `dataclasses`
- Parse, don't validate, with your dataclasses. Checks should be at the serialization boundaries, not scattered everywhere in the code.
- If you need to add an ignore, ignore a specific check like # type: ignore[specific]
- Don't write error handing code or smooth over exceptions/errors unless they are expected as part of control flow.
- In general, write code that will raise an exception early if something isn't expected.
- Enforce important expectations with asserts, but raise errors for user-facing problems.

## Documentation

- mkdocs + Jupyter. Hosted on ReadTheDocs.
- Auto API from type hints.
- Provide tutorial notebooks covering common workflows.
- Include examples in docstrings.
- Add high-level guides for key functionality.

## Attribution

- Every Python file in `src/` and `scripts/` starts with the SPDX header
  (see any module); add it to new files.
- When code from this repository is reused elsewhere, follow
  [AGENTS.md](AGENTS.md): attribution comment next to the code, per NOTICE.

---

# Working in this repo

Sections below follow `00_tools/standards/CLAUDE.md.template`. geepers' role in
the Venti / DISP-CAL / VLM system is described in Venti's `docs/specs.md` §4.

## What this repo is

GNSS data access and GNSS-vs-InSAR analysis. In the calibration system it is
the **single source** for UNR grid/station retrieval, GPS Imaging
re-interpolation and Euler-pole plate motion; Venti and cal-disp must not
re-implement these.

## Architecture (target, Venti `docs/plan.md` T12–T15)

- **core** (what the operational image installs): `gps_sources/` (UNR grid,
  UNR stations, sideshow), `schemas`, `utils`.
- **`[grid]`**: `gps_imaging` (incl. exclusion-area re-interpolation), `euler`.
- **`[analysis]`**: MIDAS, strain, cross-validation, variability, plotting,
  zarr/dask workflows — used by the validation package, never by cal-disp.
- **`[all]`** = everything.

## Invariants

- Nothing in core imports dask, zarr, pandera, rasterio, geopandas or
  matplotlib at module import time (the `core-only` CI job checks this once
  T13 lands).
- Cassette-based tests (`pytest-recording`); no live network in CI.
- SPDX header on every `.py` in `src/` and `scripts/` (enforced by pre-commit).

## Commands

```bash
export RATTLER_CACHE_DIR=/u/aurora-r0/govorcin/.cache/rattler UV_CACHE_DIR=/u/aurora-r0/govorcin/.cache/uv
pixi install -e dev
pixi run -e dev test
pixi run -e dev lint
```

## Conventions

Never commit on `main` (it mirrors `opera-adt/geepers`); `feature/<topic>`
branches, one concern each; see `CONTRIBUTING.md`. Every addition ships with
its unit test; API changes that alter numbers get a regression test on a
fixture.
