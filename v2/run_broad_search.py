import os
import sys
from pathlib import Path

import pandas as pd


def _add_src_to_path() -> None:
    here = Path(__file__).resolve().parent
    src = here / "src"
    sys.path.insert(0, str(src))


def main() -> None:
    _add_src_to_path()
    from utils.io import load_config, ensure_dirs
    from train.tune import run_broad_search

    cfg_path = Path(__file__).resolve().parent / "config.yaml"
    cfg = load_config(cfg_path)

    data_csv = Path(__file__).resolve().parent / cfg.get("paths.data_csv")
    # Robust CSV loading (handles semicolon-delimited files labeled as .csv)
    df = pd.read_csv(data_csv)
    if len(df.columns) == 1 and ";" in df.columns[0]:
        df = pd.read_csv(data_csv, sep=";")

    ensure_dirs(cfg.get("paths.logs_dir"), cfg.get("paths.artifacts_dir"))

    results = run_broad_search(df, cfg)
    out_csv = Path(cfg.get("paths.logs_dir")) / "broad_search_results.csv"
    results.to_csv(out_csv, index=False)
    print("Saved:", out_csv)
    print(results)


if __name__ == "__main__":
    main()


