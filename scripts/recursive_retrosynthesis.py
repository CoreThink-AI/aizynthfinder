"""Recursive retrosynthesis tree expansion via the Chemformer API.

Repeatedly calls the retrosynthesis API on a product SMILES, then on each
predicted reactant that looks "viable" (log-likelihood above a threshold and
not yet a simple purchasable building block), building a retrosynthesis tree.

Usage:
    python scripts/recursive_retrosynthesis.py "CC(=O)Oc1ccccc1C(=O)O" [options]

Options:
    --model          Model name stem (default: uses API default)
    --url            API base URL
    --n-beams        Beam count per call (default: 5)
    --max-depth      Max recursion depth (default: 3)
    --min-ll         Min log-likelihood to expand a reaction (default: -5.0)
    --max-mw         Max molecular weight to treat a fragment as a leaf (default: 200)
    --out            Output YAML file (default: stdout)
"""

import argparse
import sys
import time
from typing import Optional

import requests
import yaml

API_URL = "https://chemformer-retrosynthesis-knq67derjq-uc.a.run.app"


# ── molecular weight estimate (no rdkit dependency) ──────────────────────────
# Rough atom-weight table sufficient for a leaf/expand heuristic.
_ATOMIC_WEIGHTS = {
    "C": 12, "N": 14, "O": 16, "S": 32, "F": 19, "Cl": 35.5,
    "Br": 80, "I": 127, "P": 31, "B": 11, "H": 1, "Si": 28,
}


def _estimate_mw(smiles: str) -> float:
    """Very rough MW from SMILES atom counts (ignores implicit H precisely)."""
    import re
    mw = 0.0
    for sym, w in sorted(_ATOMIC_WEIGHTS.items(), key=lambda x: -len(x[0])):
        # match uppercase symbol not inside brackets already consumed
        count = len(re.findall(r'(?<!\[)' + sym + r'(?![]a-z])', smiles))
        mw += count * w
    return mw


def _split_reactants(dot_smiles: str) -> list:
    return [s.strip() for s in dot_smiles.split(".") if s.strip()]


def _predict(smiles: str, model: Optional[str], n_beams: int, url: str) -> list:
    """Call the API and return the predictions list, or [] on failure."""
    endpoint = f"{url}/retrosynthesis/{model}/predict" if model else f"{url}/retrosynthesis/predict"
    try:
        r = requests.post(
            endpoint,
            json={"smiles": smiles, "n_beams": n_beams},
            timeout=180,
        )
        r.raise_for_status()
        return r.json().get("predictions", [])
    except Exception as exc:
        print(f"  [API error for {smiles[:50]}: {exc}]", file=sys.stderr)
        return []


def _is_leaf(smiles: str, max_mw: float) -> bool:
    """Treat a fragment as a leaf (building block) if MW is small."""
    return _estimate_mw(smiles) <= max_mw


# ── recursive expansion ───────────────────────────────────────────────────────

def expand(
    smiles: str,
    depth: int,
    max_depth: int,
    min_ll: float,
    max_mw: float,
    n_beams: int,
    model: Optional[str],
    url: str,
    visited: set,
) -> dict:
    """Return a retrosynthesis tree node for *smiles*."""
    node = {"smiles": smiles, "depth": depth}

    if smiles in visited:
        node["status"] = "already_expanded"
        return node

    if _is_leaf(smiles, max_mw):
        node["status"] = "leaf"
        node["estimated_mw"] = round(_estimate_mw(smiles), 1)
        return node

    if depth >= max_depth:
        node["status"] = "max_depth_reached"
        return node

    visited.add(smiles)
    indent = "  " * depth
    print(f"{indent}▶ depth={depth} {smiles[:70]}", file=sys.stderr)

    predictions = _predict(smiles, model, n_beams, url)
    viable = [p for p in predictions if p["log_likelihood"] >= min_ll]

    if not viable:
        node["status"] = "no_viable_reactions"
        node["best_ll"] = predictions[0]["log_likelihood"] if predictions else None
        return node

    node["status"] = "expanded"
    node["reactions"] = []

    for pred in viable:
        time.sleep(0.1)  # be polite to the API
        reactant_smiles = _split_reactants(pred["reactants_smiles"])
        reaction_node = {
            "log_likelihood": round(pred["log_likelihood"], 4),
            "reaction_smarts": pred["reaction_smarts"],
            "reactants": [
                expand(r, depth + 1, max_depth, min_ll, max_mw, n_beams, model, url, visited)
                for r in reactant_smiles
            ],
        }
        node["reactions"].append(reaction_node)

    return node


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("smiles", help="SMILES of the target molecule")
    parser.add_argument("--model", default=None, help="Model name stem (omit for API default)")
    parser.add_argument("--url", default=API_URL, help="API base URL")
    parser.add_argument("--n-beams", type=int, default=5)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--min-ll", type=float, default=-5.0,
                        help="Min log-likelihood to treat a reaction as viable")
    parser.add_argument("--max-mw", type=float, default=200.0,
                        help="MW threshold below which a fragment is treated as a purchasable leaf")
    parser.add_argument("--out", default=None, help="Output YAML file (default: stdout)")
    args = parser.parse_args()

    tree = expand(
        smiles=args.smiles,
        depth=0,
        max_depth=args.max_depth,
        min_ll=args.min_ll,
        max_mw=args.max_mw,
        n_beams=args.n_beams,
        model=args.model,
        url=args.url,
        visited=set(),
    )

    output = yaml.dump({"query_smiles": args.smiles, "tree": tree},
                       allow_unicode=True, sort_keys=False, width=120)

    if args.out:
        with open(args.out, "w") as f:
            f.write(output)
        print(f"Wrote tree → {args.out}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
