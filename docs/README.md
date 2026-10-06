# Building the docs locally

From the repository root:

```console
uv run --extra cpu --with sphinx --with myst-parser sphinx-build -b html docs docs/_build/html
```

Then open `docs/_build/html/index.html`.

`myst-parser` is required because the README and the contributing guide are
Markdown and are pulled in via `docs/overview.md` and `docs/contributing.md`.
Read the Docs installs it from `docs/requirements.txt`, together with a
CPU-only torch; the package itself is installed from `pyproject.toml`.
