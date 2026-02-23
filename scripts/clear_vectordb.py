#!/usr/bin/env python3
"""Clear all data from the LanceDB vector database."""

import argparse
import shutil
import sys
from pathlib import Path

# Resolve project root relative to this script's location
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "lancedb"

# Try loading from config.yaml
try:
    import yaml
    config_path = PROJECT_ROOT / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
        db_from_config = config.get("lancedb", {}).get("db_path", None)
        if db_from_config:
            DEFAULT_DB_PATH = (PROJECT_ROOT / db_from_config).resolve()
except ImportError:
    pass  # yaml not available, use default

def clear_db(db_path: Path, force: bool = False):
    print(f"📂 Database path: {db_path}")
    
    if not db_path.exists():
        print(f"ℹ️  Database path does not exist: {db_path}")
        return

    # List tables
    tables = [d for d in db_path.iterdir() if d.is_dir() and d.suffix == ".lance"]
    
    if not tables:
        print("ℹ️  Database is already empty.")
        return

    print(f"📦 Found {len(tables)} table(s):")
    for t in tables:
        print(f"   - {t.name}")

    if not force:
        confirm = input("\n⚠️  Delete ALL data? (y/N): ").strip().lower()
        if confirm != "y":
            print("❌ Cancelled.")
            return

    for t in tables:
        shutil.rmtree(t)
        print(f"   🗑️  Deleted {t.name}")

    print("\n✅ Vector database cleared!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clear LanceDB vector database")
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH, help="Path to LanceDB directory")
    parser.add_argument("--force", "-f", action="store_true", help="Skip confirmation prompt")
    args = parser.parse_args()

    clear_db(args.db_path, args.force)
