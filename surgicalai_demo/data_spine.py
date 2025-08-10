from __future__ import annotations
import hashlib
from pathlib import Path
import duckdb
import pandas as pd

def sha1_path(p: Path) -> str:
    h = hashlib.sha1(p.read_bytes()).hexdigest()
    return f"{h}{p.suffix.lower()}"

def excel_to_parquet(excel_path: Path, parquet_path: Path) -> pd.DataFrame:
    df = pd.read_excel(excel_path)
    # Normalize minimal columns for Tier‑0
    expected = ["image_id","path","dx","dx_method","age","sex","body_site","breslow_mm","clark_level","followup","malignancy_flag"]
    for col in expected:
        if col not in df.columns:
            df[col] = None
    df.to_parquet(parquet_path, index=False)
    return df

def build_duckdb(parquet_path: Path, db_path: Path):
    con = duckdb.connect(str(db_path))
    con.execute("CREATE OR REPLACE TABLE lesions AS SELECT * FROM read_parquet(?)", [str(parquet_path)])
    con.close()
