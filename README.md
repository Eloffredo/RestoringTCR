# RestoringTCR

This repo supports the paper "Restoring data balance via generative models of T cell receptors for antigen-binding prediction" authored by Emanuele Loffredo, Mauro Pastore, Simona Cocco and Remi Monasson.

To use this repo and reproduce our findings, clone this repo with ``git clone`` and create the env using [uv](https://docs.astral.sh/uv/). Simply run ``uv lock && uv sync`` to build the env using the ``pyproject.toml`` and then activate the environment with ``source .venv/bin/activate``.

The set of figures found in the paper together with supporting data can be found in ``figures/`` with a notebook to reproduce the all the plots.
The underlying script are in the ``src/`` directory.
