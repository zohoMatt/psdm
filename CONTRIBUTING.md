# Contributing

Development uses [uv](https://docs.astral.sh/uv/).

```bash
uv sync                      # create .venv with the package and dev tools
uv run pytest                # tests
uv run ruff check .          # lint
uv run ruff format .         # format
uv run --group docs sphinx-build -W -b html docs docs/_build/html   # docs
```

## Reference data

`tests/data/reference/*.csv` are computed with the original US EPA code
(`Water_Treatment_Models` commit `3f514c2`). To regenerate them, check out that repository and
run the generator scripts:

| script | output | used by |
|---|---|---|
| `generate.py` | `addesigns_single.csv`, `multi_*.csv` | `tests/test_solver.py` |
| `generate_io.py` | `io_process_input_file.csv`, `../io_missing_t0.xlsx` | `tests/test_io_plotting.py` |
| `generate_tce.py` | `tce_curves.csv`, `tce_params.csv` | `tests/test_tce_epa.py` |

```bash
uv run --no-project --with numpy --with scipy --with "pandas<3" --with openpyxl --with matplotlib \
    python tests/data/reference/generate.py <Water_Treatment_Models>/PSDM tests/data/reference
```

(`generate_io.py` also takes the `examples/` folder as a third argument.)

`tests/data/addesigns_test.xlsx` holds AdDesignS output for the single-compound cases and is
the independent check.

## Releasing

1. Update `__version__` in `src/psdm/__init__.py` and the changelog.
2. Commit, tag `vX.Y.Z`, push, and publish a GitHub release for the tag.
3. The `Release` workflow tests, builds and uploads to PyPI via trusted publishing.
