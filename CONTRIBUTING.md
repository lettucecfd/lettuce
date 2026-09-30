# Contributing to lettuce

Contributions are welcome and appreciated — bug reports, documentation fixes,
new boundary conditions, collision models or flow setups alike. This is a
volunteer-driven research project, so please keep the scope of a single change
as narrow as it can reasonably be.

## Ways to contribute

**Report a bug** — open an issue using the
[bug report template](https://github.com/lettucecfd/lettuce/issues/new?template=BUG_REPORT.yml).
Please include your operating system, your GPU and CUDA version if the problem
is device-related, and the steps needed to reproduce the behaviour.

**Suggest a feature** — open an issue using the
[feature request template](https://github.com/lettucecfd/lettuce/issues/new?template=FEATURE_REQUEST.yml).

**Improve the documentation** — docstrings, the examples in `examples/`, and the
Sphinx sources under `docs/` all count. Documentation-only pull requests are
very welcome.

**Fix an issue** — issues labelled `bug` or `enhancement` that nobody is
assigned to are open to anyone.

## Setting up a development environment

lettuce uses [uv](https://docs.astral.sh/uv/) for dependency management. Install
it first, then:

```console
git clone https://github.com/lettucecfd/lettuce
cd lettuce
uv sync --extra cpu
```

`uv sync` creates `.venv/`, installs all dependencies and installs lettuce
itself in editable mode, so code changes take effect immediately.

If you plan to open a pull request, fork the repository on GitHub first and
clone your fork instead.

### Choosing the right extra

Exactly one hardware extra must be selected — they are mutually exclusive:

| Extra | Use for |
|---|---|
| `cpu` | No GPU available, or macOS |
| `cu124` | CUDA 12.4 |
| `cu126` | CUDA 12.6 |
| `cu128` | CUDA 12.8 |
| `cu130` | CUDA 13.0 |

The extra selects the PyTorch build that uv pulls from the corresponding
PyTorch index. CUDA 12.4 is the oldest version we test; older toolkits are not
maintained.

## Running the tests

The unit tests:

```console
uv run --extra cpu pytest tests
```

The integration checks that CI also runs — a convergence study and a
performance benchmark driven through the command-line interface:

```console
uv run --extra cpu lettuce --no-cuda convergence --use-no-cuda_native
uv run --extra cpu lettuce --no-cuda benchmark --use-no-cuda_native
```

To run a single directory or file:

```console
uv run --extra cpu pytest tests/collision
uv run --extra cpu pytest tests/reporter/test_HDF5Reporter.py
```

### A note on skipped tests

On a machine without a CUDA device the test suite reports a large number of
skips — currently around 880 of roughly 1280 collected tests. Almost all of them
carry the reason `CUDA is not available on this machine.` and come from the
`device`, `native` and `configuration` fixtures in `tests/conftest.py`.

This means **a green test run on a CPU-only machine does not exercise the CUDA
or `cuda_native` code paths at all.** If your change touches
`lettuce/cuda_native/`, device handling, or anything under `lettuce/ext/`, run
the suite on a CUDA machine before opening the pull request, or say so
explicitly in the PR description so a reviewer with suitable hardware can check
it.

Use `pytest -rs` to see the skip reasons for a run.

## Continuous integration

Every pull request is built against eight configurations: on Python 3.12, Ubuntu
with each of the five hardware extras plus macOS with `cpu`; and on Python 3.13,
Ubuntu and macOS with `cpu`. The two ends of the range declared in
`requires-python` are therefore both covered.

Note that the GitHub runners have no GPU, so the CUDA builds verify that the
correct PyTorch variant resolves and installs — they do not execute CUDA kernels
either.

## Opening a pull request

1. Branch off `master`:

   ```console
   git checkout -b my-bugfix-or-feature
   ```

2. Make your change, together with tests and documentation.
3. Push the branch and open a pull request.

The [pull request template](https://github.com/lettucecfd/lettuce/blob/master/.github/pull_request_template.md)
contains the checklist that reviewers go by:

- The pull request is associated with an issue.
- The pull request has a description.
- If you added a new method:
  - it has a description,
  - the class is mentioned in the corresponding `__init__` where appropriate,
  - there is an example using it in `examples/simple_flows/` or
    `examples/advanced_flows/`,
  - there is a test in `tests/`.
- Add someone else as reviewer and wait for approval before merging.

### Where things go

| What you added | Where it belongs |
|---|---|
| Boundary condition | `lettuce/ext/_boundary/` |
| Collision model | `lettuce/ext/_collision/` |
| Equilibrium | `lettuce/ext/_equilibrium/` |
| Flow setup | `lettuce/ext/_flows/` |
| Forcing scheme | `lettuce/ext/_force/` |
| Reporter / observable | `lettuce/ext/_reporter/` |
| Stencil | `lettuce/ext/_stencil/` |
| Native CUDA code generation | `lettuce/cuda_native/` |

Tests mirror this layout under `tests/` (`tests/boundary/`, `tests/collision/`,
`tests/flow/`, `tests/native/`, …).

## Code style

The project does not currently configure an automated formatter or linter, so
there is no command you need to run before submitting. Please match the style of
the code you are editing: the existing sources follow PEP 8 with four-space
indentation, and public classes and functions carry docstrings.

Type hints are used in the newer parts of the codebase — adding them to code you
touch is welcome, but not required.

## Building the documentation

```console
uv run --extra cpu --with sphinx --with myst-parser sphinx-build -b html docs docs/_build/html
```

Then open `docs/_build/html/index.html`.

The README is written in Markdown and pulled into the documentation through
`docs/overview.md`, which is why `myst-parser` is required. Read the Docs
installs it from `docs/requirements.txt`, together with a CPU-only torch;
the package itself is installed from `pyproject.toml`.

The API reference in `docs/modules.rst` lists the public classes and functions
by name (`lettuce.BGKCollision`, not the private module they are defined in).
When you add a public class, add an entry there as well.

## Versioning and releases

Versions are derived from git tags by
[setuptools-scm](https://github.com/pypa/setuptools-scm) — there is no version
number to edit by hand anywhere in the source tree. A build from an untagged
commit produces a development version such as `0.2.4.dev303+g6094d7a7e`.

The project is not yet published to a package index.
