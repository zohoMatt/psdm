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

`tests/data/reference/*.csv` are breakthrough curves computed with the original US EPA code
(`Water_Treatment_Models` commit `3f514c2`). To regenerate them, check out that repository and
run

```bash
uv run --no-project --with numpy --with scipy --with pandas --with openpyxl --with matplotlib \
    python tests/data/reference/generate.py <Water_Treatment_Models>/PSDM tests/data/reference
```

`tests/data/addesigns_test.xlsx` holds AdDesignS output for the same case and is the
independent check.

## Releasing

1. Update `__version__` in `src/psdm/__init__.py` and the changelog.
2. Commit, tag `vX.Y.Z`, push, and publish a GitHub release for the tag.
3. The `Release` workflow tests, builds and uploads to PyPI via trusted publishing.
