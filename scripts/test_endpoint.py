"""Test the retrosynthesis FastAPI endpoint using molecules from zydus_queries.yaml.

Usage:
    python scripts/test_endpoint.py [--query-name Aspirin] [--url http://localhost:8000]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import requests
import yaml


QUERIES_FILE = Path(__file__).parent / "zydus_queries.yaml"


def load_molecule(query_name: str) -> dict:
    with open(QUERIES_FILE) as f:
        data = yaml.safe_load(f)
    matches = [m for m in data["molecules"] if m["query_name"] == query_name]
    if not matches:
        available = sorted({m["query_name"] for m in data["molecules"]})
        raise SystemExit(
            f"Query name {query_name!r} not found. Available: {available}"
        )
    return matches[0]


def summarise_routes(routes: list[dict], label: str) -> None:
    if not routes:
        print(f"  {label}: (none)")
        return
    print(f"  {label}: {len(routes)} route(s)")
    for i, route in enumerate(routes):
        scores = route.get("scores", {})
        n_rxn = scores.get("number of reactions", "?")
        n_prec = scores.get("number of pre-cursors", "?")
        n_stock = scores.get("number of pre-cursors in stock", "?")
        state = scores.get("state score", "?")
        created = route.get("metadata", {}).get("created_at_iteration", "?")
        print(
            f"    [{i}] state={state:.4f}  "
            f"reactions={n_rxn}  precursors={n_prec}  in_stock={n_stock}  "
            f"iteration={created}"
        )
        # Print the first reaction in each route
        children = route.get("children", [])
        if children and children[0].get("type") == "reaction":
            rxn_smiles = children[0].get("smiles", "")
            prob = children[0].get("metadata", {}).get("policy_probability", "?")
            policy = children[0].get("metadata", {}).get("policy_name", "?")
            print(f"         rxn: {rxn_smiles}")
            print(f"         policy={policy}  prob={prob:.4f}" if isinstance(prob, float) else f"         policy={policy}  prob={prob}")
            leaf_smiles = [
                c["smiles"]
                for c in children[0].get("children", [])
                if c.get("type") == "mol"
            ]
            print(f"         precursors: {leaf_smiles}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-name", default="Aspirin")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--iteration-limit", type=int, default=100)
    parser.add_argument("--max-routes", type=int, default=10)
    args = parser.parse_args()

    mol = load_molecule(args.query_name)
    smiles = mol["smiles"]
    print(f"Query : {mol['query_name']} (PubChem CID {mol['pubchem_cid']})")
    print(f"Name  : {mol['pubchem_name']}")
    print(f"SMILES: {smiles}")
    print()

    endpoint = f"{args.url.rstrip('/')}/retrosynthesis"
    payload = {
        "smiles": smiles,
        "iteration_limit": args.iteration_limit,
        "max_routes": args.max_routes,
    }

    print(f"POST {endpoint}")
    try:
        resp = requests.post(endpoint, json=payload, timeout=300)
    except requests.ConnectionError:
        raise SystemExit(
            f"Could not connect to {endpoint}. "
            "Start the server first:\n"
            "  AIZYNTH_CONFIG=... uvicorn service:app --port 8000"
        )

    if resp.status_code != 200:
        print(f"ERROR {resp.status_code}: {resp.text}", file=sys.stderr)
        sys.exit(1)

    result = resp.json()
    stats = result["stats"]
    print(f"Search time : {stats['search_time_seconds']:.2f}s")
    print(f"Iterations  : {stats['iterations']}")
    print(f"Nodes       : {stats['number_of_nodes']}")
    print(f"Routes found: {stats['number_of_routes']}  solved: {stats['number_of_solved_routes']}")
    print(f"Top score   : {stats['top_score']:.6f}")
    print(f"Stock       : {'configured' if result['stock_configured'] else 'not configured'}")
    print()
    summarise_routes(result["purchasable_routes"], "Purchasable routes")
    print()
    summarise_routes(result["non_purchasable_routes"], "Non-purchasable routes")
    print()

    out_path = Path(__file__).parent.parent / "aizynthfinder" / "data" / f"endpoint_test_{mol['query_name']}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"Full response saved to {out_path}")


if __name__ == "__main__":
    main()
