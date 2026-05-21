"""FastAPI service for AiZynthFinder retrosynthesis using ONNX expansion policies.

Run locally:
    uvicorn service:app --port 8000

Example:
    curl -X POST http://localhost:8000/retrosynthesis \\
        -H "Content-Type: application/json" \\
        -d '{"smiles": "CC(=O)Oc1ccccc1C(=O)O"}'
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from rdkit import Chem

from aizynthfinder.aizynthfinder import AiZynthFinder


CONFIG_PATH = os.environ.get(
    "AIZYNTH_CONFIG",
    os.path.join(os.path.dirname(__file__), "config_production.yml"),
)


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class RetrosynthesisRequest(BaseModel):
    smiles: str = Field(
        ...,
        description="SMILES string of the target product molecule.",
        examples=["CC(=O)Oc1ccccc1C(=O)O"],
    )
    time_limit: int = Field(
        default=120,
        ge=10,
        le=3600,
        description="Search time limit in seconds (10–3600).",
    )
    iteration_limit: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Maximum MCTS iterations (1–1000).",
    )
    max_routes: int = Field(
        default=10,
        ge=1,
        le=500,
        description="Maximum number of routes to return per category.",
    )


class SearchStats(BaseModel):
    search_time_seconds: float
    iterations: int
    expansion_calls: int
    number_of_nodes: int
    number_of_routes: int
    number_of_solved_routes: int
    is_solved: bool
    top_score: float


class RetrosynthesisResponse(BaseModel):
    target_smiles: str
    stats: SearchStats
    stock_configured: bool
    purchasable_routes: List[Dict[str, Any]] = Field(
        description=(
            "Routes whose leaf molecules are all in stock (purchasable precursors). "
            "Empty if no stock is configured."
        )
    )
    non_purchasable_routes: List[Dict[str, Any]] = Field(
        description=(
            "Routes containing at least one leaf molecule not in stock. "
            "If no stock is configured, all routes are returned here."
        )
    )


# ── FastAPI app + AiZynthFinder bootstrap ────────────────────────────────────

app = FastAPI(
    title="AiZynthFinder ONNX Retrosynthesis",
    description=(
        "Retrosynthetic planning service using AiZynthFinder MCTS with ONNX-based "
        "USPTO and Ringbreaker expansion policies and a ZINC purchasability stock. "
        "Routes are separated by whether their leaf precursors are purchasable."
    ),
    version="1.0",
)

_finder: Optional[AiZynthFinder] = None


def get_finder() -> AiZynthFinder:
    """Lazily initialize AiZynthFinder and reuse it across requests."""
    global _finder
    if _finder is None:
        if not os.path.exists(CONFIG_PATH):
            raise RuntimeError(f"Config file not found: {CONFIG_PATH}")
        finder = AiZynthFinder(configfile=CONFIG_PATH)

        if not finder.expansion_policy.items:
            raise RuntimeError("No expansion policy configured in YAML")
        finder.expansion_policy.select_all()

        if finder.filter_policy.items:
            finder.filter_policy.select_all()

        if finder.stock.items:
            finder.stock.select_all()

        _finder = finder
    return _finder


# ── helpers ───────────────────────────────────────────────────────────────────

def _route_is_purchasable(route_dict: Dict[str, Any]) -> bool:
    """Return True if every leaf molecule in the route tree is in_stock."""
    def walk(node: Dict[str, Any]) -> bool:
        children = node.get("children", [])
        if not children:
            return bool(node.get("in_stock", False))
        return all(walk(child) for child in children)
    return walk(route_dict)


def _heavy_atom_count(smiles: str) -> int:
    mol = Chem.MolFromSmiles(smiles)
    return mol.GetNumHeavyAtoms() if mol is not None else 0


def _is_productive_reaction(reaction_smiles: str) -> bool:
    """Return True if the retrosynthesis step actually simplifies the product.

    A step is productive if it produces multiple fragments (real disconnection)
    or its single reactant has strictly fewer heavy atoms than the product.
    """
    if ">>" not in reaction_smiles:
        return True
    product, reactants = reaction_smiles.split(">>", 1)
    reactant_parts = [r for r in reactants.split(".") if r]
    if not reactant_parts:
        return False
    if len(reactant_parts) > 1:
        return True
    return _heavy_atom_count(reactant_parts[0]) < _heavy_atom_count(product)


def _route_is_meaningful(route_dict: Dict[str, Any]) -> bool:
    """Return True if the route has at least one reaction and all reactions are productive."""
    has_reaction = False

    def walk(node: Dict[str, Any]) -> bool:
        nonlocal has_reaction
        if node.get("type") == "reaction" or node.get("is_reaction"):
            has_reaction = True
            if not _is_productive_reaction(node.get("smiles", "")):
                return False
        for child in node.get("children", []):
            if not walk(child):
                return False
        return True

    return walk(route_dict) and has_reaction


# ── endpoints ─────────────────────────────────────────────────────────────────

@app.post("/retrosynthesis", response_model=RetrosynthesisResponse)
def retrosynthesize(request: RetrosynthesisRequest) -> RetrosynthesisResponse:
    """Run AiZynthFinder MCTS for the given target SMILES and return routes,
    split by whether leaf precursors are purchasable (in stock).
    """
    finder = get_finder()

    original_time = finder.config.search.time_limit
    original_iter = finder.config.search.iteration_limit
    finder.config.search.time_limit = request.time_limit
    finder.config.search.iteration_limit = request.iteration_limit

    try:
        finder.target_smiles = request.smiles
        finder.prepare_tree()
        finder.tree_search()
        finder.build_routes()
        finder.routes.compute_scores(*finder.scorers.objects())
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Search failed: {type(exc).__name__}: {exc}"
        )
    finally:
        finder.config.search.time_limit = original_time
        finder.config.search.iteration_limit = original_iter

    stock_configured = bool(finder.stock.selection)
    route_dicts = list(finder.routes.dict_with_extra(include_scores=True, include_metadata=True))
    route_dicts = [r for r in route_dicts if _route_is_meaningful(r)]

    purchasable: List[Dict[str, Any]] = []
    non_purchasable: List[Dict[str, Any]] = []
    for route_dict in route_dicts:
        if stock_configured and _route_is_purchasable(route_dict):
            purchasable.append(route_dict)
        else:
            non_purchasable.append(route_dict)

    purchasable = purchasable[: request.max_routes]
    non_purchasable = non_purchasable[: request.max_routes]

    raw_stats = finder.extract_statistics()
    profiling = getattr(finder.tree, "profiling", {}) or {}

    stats = SearchStats(
        search_time_seconds=float(raw_stats.get("search_time", 0.0)),
        iterations=int(profiling.get("iterations", 0)),
        expansion_calls=int(profiling.get("expansion_calls", 0)),
        number_of_nodes=int(raw_stats.get("number_of_nodes", 0)),
        number_of_routes=int(raw_stats.get("number_of_routes", len(route_dicts))),
        number_of_solved_routes=int(raw_stats.get("number_of_solved_routes", 0)),
        is_solved=bool(raw_stats.get("is_solved", False)),
        top_score=float(raw_stats.get("top_score") or 0.0),
    )

    return RetrosynthesisResponse(
        target_smiles=request.smiles,
        stats=stats,
        stock_configured=stock_configured,
        purchasable_routes=purchasable,
        non_purchasable_routes=non_purchasable,
    )


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/config")
def get_config() -> Dict[str, Any]:
    """Return information about the loaded AiZynthFinder configuration."""
    finder = get_finder()
    return {
        "config_path": CONFIG_PATH,
        "expansion_policies": list(finder.expansion_policy.items),
        "expansion_policies_selected": finder.expansion_policy.selection,
        "filter_policies": list(finder.filter_policy.items),
        "filter_policies_selected": finder.filter_policy.selection,
        "stocks": list(finder.stock.items),
        "stocks_selected": finder.stock.selection,
        "search_time_limit": finder.config.search.time_limit,
        "search_iteration_limit": finder.config.search.iteration_limit,
        "max_transforms": finder.config.search.max_transforms,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "service:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8000")),
        log_level="info",
        reload=False,
    )
