"""Load zinc_stock.hdf5 into a Pandas DataFrame and report basic stats.

The HDF5 file stores a single column of InChI keys under the HDF5 key "table".
It was built with aizynthfinder's make_stock tool from ZINC purchasable compounds.

Usage:
    python scripts/etl_zinc_stock.py
    python scripts/etl_zinc_stock.py --path /app/bigdata/zinc_stock.hdf5
    python scripts/etl_zinc_stock.py --lookup "CC(=O)Oc1ccccc1C(=O)O"
    python scripts/etl_zinc_stock.py --lookup-inchikey BSYNRYMUTXBXSQ-UHFFFAOYSA-N
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

DEFAULT_PATHS = [
    os.environ.get("BIGDATA_DIR", ""),
    Path(__file__).parent.parent.parent / "ai-routes-generator" / "bigdata",
    Path("/app/bigdata"),
]
FILENAME = "zinc_stock.hdf5"
HDF5_KEY = "table"
INCHI_KEY_COL = "inchi_key"


def find_stock_file() -> Path:
    for base in DEFAULT_PATHS:
        if not base:
            continue
        candidate = Path(base) / FILENAME
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find {FILENAME}. Pass --path explicitly or set BIGDATA_DIR."
    )


def load_stock(path: Path) -> pd.DataFrame:
    print(f"Loading {path} ...", flush=True)
    df = pd.read_hdf(path, key=HDF5_KEY)
    return df


def smiles_to_inchikey(smiles: str) -> str:
    try:
        from rdkit import Chem
        from rdkit.Chem.inchi import InchiToInchiKey, MolToInchi
    except ImportError:
        raise SystemExit("rdkit is required for SMILES lookup. Install it first.")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise SystemExit(f"RDKit could not parse SMILES: {smiles!r}")
    inchi = MolToInchi(mol)
    return InchiToInchiKey(inchi)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--path", type=Path, help="Path to zinc_stock.hdf5")
    parser.add_argument("--lookup", metavar="SMILES", help="Check if a SMILES is in stock")
    parser.add_argument("--lookup-inchikey", metavar="INCHIKEY", help="Check if an InChI key is in stock")
    parser.add_argument("--head", type=int, default=5, metavar="N", help="Print first N rows (default: 5)")
    args = parser.parse_args()

    path = args.path or find_stock_file()
    df = load_stock(path)

    # Basic info
    print(f"\n{'='*50}")
    print(f"File   : {path}")
    print(f"Rows   : {len(df):,}")
    print(f"Columns: {list(df.columns)}")
    print(f"Memory : {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    print(f"Dtype  : {df[INCHI_KEY_COL].dtype}")
    print(f"Unique : {df[INCHI_KEY_COL].nunique():,}")
    print(f"\nFirst {args.head} rows:")
    print(df.head(args.head).to_string(index=True))
    print(f"{'='*50}\n")

    # Optional membership lookups
    stock_set = set(df[INCHI_KEY_COL].values)

    if args.lookup_inchikey:
        key = args.lookup_inchikey
        found = key in stock_set
        print(f"InChI key {key!r}: {'IN STOCK' if found else 'not in stock'}")

    if args.lookup:
        key = smiles_to_inchikey(args.lookup)
        found = key in stock_set
        print(f"SMILES {args.lookup!r}")
        print(f"  InChI key: {key}")
        print(f"  Status   : {'IN STOCK' if found else 'not in stock'}")


if __name__ == "__main__":
    main()
