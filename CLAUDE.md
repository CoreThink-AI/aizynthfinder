# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

AiZynthFinder is a retrosynthetic planning tool. Given a target molecule (as SMILES), it searches backwards through known reaction templates to find purchasable precursors. The default algorithm is Monte Carlo Tree Search (MCTS), but the framework supports swappable search algorithms, expansion policies, filter policies, stock sources, and scorers.

## Development setup

Use `uv` and `pyproject.toml` for installing all dependencies in a venv:

```bash
git clone git@github.com:CoreThink-AI/aizynthfinder
uv venv -p 3.10
source .venv/bin/activate
uv pip install -e .[all]
uv pip install fastapi "uvicorn[standard]"
```

## Commands

Use pytest from the activated venv to run unittests

```bash
pytest -v
```

Use the activated venv python to run all scripts:

```bash
python -m aizynthcli --config config.yml --smiles smiles.txt
python -m aizynthapp --config config.yml          # Jupyter notebook GUI
python -m download_public_data <folder>           # download pretrained models + stock
python -m smiles2stock <infile> <outfile>         # convert SMILES file to HDF5 stock
```

## Architecture

### Core data flow

`AiZynthFinder` (in `aizynthfinder/aizynthfinder.py`) is the main entry point. A typical run:

1. Load `Configuration` from YAML → populates `Stock`, `ExpansionPolicy`, `FilterPolicy`, `ScorerCollection`
2. Set `finder.target_smiles` → resets the search tree
3. Call `finder.tree_search()` → runs MCTS (or alternative) until time/iteration limit
4. Call `finder.build_routes()` → extracts `ReactionTree` objects via `TreeAnalysis`
5. Inspect `finder.routes` for ranked synthesis routes

### Configuration (`context/config.py`)

`Configuration` is a dataclass that owns all context objects. It reads YAML files that may contain `${ENV_VAR}` substitution. Top-level YAML keys:

- `expansion` → `ExpansionPolicy` (one or more named strategies)
- `filter` → `FilterPolicy`
- `stock` → `Stock`
- `scorer` → `ScorerCollection`
- `search` → `_SearchConfiguration` (algorithm, limits, bond constraints)
- `post_processing` → `_PostprocessingConfiguration` (route selection, scoring)

### Extensibility via `ContextCollection`

`Stock`, `ExpansionPolicy`, `FilterPolicy`, and `ScorerCollection` all inherit from `ContextCollection`. Each collection holds named items (loaded by key) and a selection (which items are active). New strategies are loaded dynamically via `load_dynamic_class()` in `utils/loading.py`, which resolves `"module.path.ClassName"` strings at runtime — this is the plugin mechanism used throughout.

### Search algorithms (`search/`)

- **`search/mcts/`** — default MCTS. `MctsSearchTree.one_iteration()` does select → expand → rollout → backprop. Supports single-objective, weighted-sum, and Pareto multi-objective modes via `_MODE2NODECLASS`.
- **`search/breadth_first/`** and **`search/dfpn/`** — alternative AND/OR tree algorithms, both extending `AndOrSearchTreeBase` from `search/andor_trees.py`.
- Custom search algorithms are loaded by setting `search.algorithm` to a fully-qualified class name.

### Expansion policies (`context/policy/expansion_strategies.py`)

`ExpansionStrategy` (ABC) exposes `get_actions(molecules) → (actions, priors)`. Built-in strategies use ONNX/TF models to predict reaction templates from molecular fingerprints. Custom strategies (e.g., Chemformer REST API, ModelZoo) live in `plugins/expansion_strategies.py` and are loaded via `PYTHONPATH`.

### Filter policies (`context/policy/filter_strategies.py`)

`FilterStrategy` (ABC) with `feasibility(reaction) → (bool, prob)`. Applied after expansion to remove low-quality reactions.

### Stock (`context/stock/`)

`Stock` holds `StockQueryMixin` implementations. Built-in: `InMemoryInchiKeyQuery` (HDF5/CSV), `MolbloomFilterQuery` (probabilistic). Membership test (`mol in stock`) checks all selected sub-stocks.

### Reaction trees (`reactiontree.py`)

`ReactionTree` is a bipartite `networkx.DiGraph` with `UniqueMolecule` and `FixedRetroReaction` nodes. `ReactionTreeFromExpansion` converts an MCTS node's expansion into a tree. Trees can be serialized to/from dict (JSON-compatible) and rendered as images.

### Chemistry primitives (`chem/`)

- `Molecule` / `TreeMolecule` — RDKit molecule wrappers; `TreeMolecule` carries parent/ancestor context for cycle detection
- `TemplatedRetroReaction` / `SmilesBasedRetroReaction` — retrosynthesis reaction types
- `chem/serialization.py` — `MoleculeSerializer`/`MoleculeDeserializer` for tree persistence

### Scorers (`context/scoring/`)

Scorers operate on either `MctsNode` or `ReactionTree` objects. `CombinedScorer` applies weighted combination. Key built-ins: `StateScorer`, `FractionInStockScorer`, `MaxTransformScorer`, `BrokenBondsScorer`. Add custom scorers by subclassing `Scorer` (from `scorers_base.py`) and registering via config.

### Interfaces

- `interfaces/aizynthcli.py` — batch CLI, reads SMILES from file, writes JSON results
- `interfaces/aizynthapp.py` — launches Jupyter notebook GUI
- `interfaces/gui/` — ipywidgets GUI components (clustering, Pareto fronts)

### Tools (`tools/`)

Standalone utilities: `download_public_data.py` (fetches pretrained models from figshare), `make_stock.py` (SMILES → HDF5 InChIKey stock), `cat_output.py` (merge multiple CLI output files).

## Plugins

The `plugins/` directory contains optional expansion strategies (Chemformer, ModelZoo) requiring extra dependencies. Add `plugins/` to `PYTHONPATH` and reference strategies as `expansion_strategies.ClassName` in config.

## Testing patterns

Tests use `pytest-datadir` for fixtures (per-test `data/` directories). Integration tests are marked with `@pytest.mark.integration` and skipped unless `--run_integration` is passed. Mock expansion/filter strategies in `conftest.py` are used throughout to avoid model loading in unit tests.
