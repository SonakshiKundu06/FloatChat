# src/warehouse/duck.py
import duckdb
import os
from pathlib import Path

DB_URL = os.getenv("DB_URL", "duckdb:///data/duck/argo.duckdb")

class DuckDBWarehouse:
    def __init__(self, db_url: str = DB_URL):
        db_path = db_url.split("///")[-1]
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = duckdb.connect(database=db_path, read_only=False)

    def register_dataframe(self, df, view_name="profiles"):
        """Register a pandas DataFrame as a DuckDB view"""
        self.conn.register(view_name + "_df", df)
        self.conn.execute(f"CREATE OR REPLACE VIEW {view_name} AS SELECT * FROM {view_name}_df")

    def query(self, sql: str):
        return self.conn.execute(sql).fetchdf()

    def close(self):
        self.conn.close()
