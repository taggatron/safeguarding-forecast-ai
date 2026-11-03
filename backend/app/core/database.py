"""SQLite persistence layer for Safeguarding Forecast AI."""
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent.parent / "data_store" / "safeguarding_forecast.db"

SYNTHETIC_BANNER = "SYNTHETIC DEMONSTRATION DATA — NOT ACTUAL COLLEGE SAFEGUARDING RECORDS"
IMPORTED_BANNER = "AGGREGATED OPERATIONAL DATA — DECISION-SUPPORT PROTOTYPE (NO INDIVIDUAL STUDENT RECORDS)"


class DatabaseManager:
    """Manages SQLite storage for aggregated datasets, configurations, and reproducible forecast runs."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path or os.environ.get("SAFEGUARDING_DB_PATH", str(DEFAULT_DB_PATH)))
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS datasets (
                    dataset_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    is_synthetic INTEGER NOT NULL DEFAULT 1,
                    seed INTEGER,
                    academic_year_start_month INTEGER NOT NULL DEFAULT 9,
                    created_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS safeguarding_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dataset_id TEXT NOT NULL,
                    period_start TEXT NOT NULL,
                    academic_year TEXT NOT NULL,
                    academic_term TEXT NOT NULL,
                    provision TEXT NOT NULL,
                    safeguarding_category TEXT NOT NULL,
                    concern_count INTEGER NOT NULL,
                    enrolled_students INTEGER NOT NULL,
                    teaching_days INTEGER NOT NULL,
                    attendance_rate REAL,
                    reporting_change_flag INTEGER NOT NULL DEFAULT 0,
                    FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id)
                )
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_records_dataset_period
                ON safeguarding_records(dataset_id, period_start)
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS app_config (
                    config_key TEXT PRIMARY KEY,
                    config_value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS forecast_runs (
                    run_id TEXT PRIMARY KEY,
                    dataset_id TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    horizon INTEGER NOT NULL,
                    provision TEXT NOT NULL,
                    safeguarding_category TEXT NOT NULL,
                    prediction_interval INTEGER NOT NULL,
                    scenario_json TEXT NOT NULL,
                    annual_total INTEGER NOT NULL,
                    lower_bound INTEGER NOT NULL,
                    upper_bound INTEGER NOT NULL,
                    runtime_ms REAL NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.commit()

            # Initialize default configuration values if not present
            defaults = {
                "active_dataset_id": "synthetic_sdc_default",
                "academic_year_start_month": "9",
                "governance_mode": "research_prototype",
                "timesfm_license_acknowledged": "true",
                "small_group_suppression_enabled": "false",
            }
            now_str = datetime.now(timezone.utc).isoformat()
            for k, v in defaults.items():
                cursor.execute(
                    "INSERT OR IGNORE INTO app_config (config_key, config_value, updated_at) VALUES (?, ?, ?)",
                    (k, v, now_str),
                )
            conn.commit()

    def get_config(self, key: str, default: Optional[str] = None) -> Optional[str]:
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT config_value FROM app_config WHERE config_key = ?", (key,)
            ).fetchone()
            return row["config_value"] if row else default

    def set_config(self, key: str, value: str) -> None:
        now_str = datetime.now(timezone.utc).isoformat()
        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT INTO app_config (config_key, config_value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(config_key) DO UPDATE SET config_value = excluded.config_value, updated_at = excluded.updated_at
                """,
                (key, str(value), now_str),
            )
            conn.commit()

    def get_all_configs(self) -> Dict[str, str]:
        with self._get_conn() as conn:
            rows = conn.execute("SELECT config_key, config_value FROM app_config").fetchall()
            return {r["config_key"]: r["config_value"] for r in rows}

    def save_dataset(
        self,
        dataset_id: str,
        name: str,
        df: pd.DataFrame,
        is_synthetic: bool = True,
        seed: Optional[int] = None,
        academic_year_start_month: int = 9,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Persist a dataset and replace its records atomically (never combining synthetic and imported)."""
        now_str = datetime.now(timezone.utc).isoformat()
        meta_json = json.dumps(metadata or {})
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM safeguarding_records WHERE dataset_id = ?", (dataset_id,))
            cursor.execute(
                """
                INSERT INTO datasets (dataset_id, name, is_synthetic, seed, academic_year_start_month, created_at, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(dataset_id) DO UPDATE SET
                    name = excluded.name,
                    is_synthetic = excluded.is_synthetic,
                    seed = excluded.seed,
                    academic_year_start_month = excluded.academic_year_start_month,
                    created_at = excluded.created_at,
                    metadata_json = excluded.metadata_json
                """,
                (dataset_id, name, 1 if is_synthetic else 0, seed, academic_year_start_month, now_str, meta_json),
            )
            records = []
            for _, row in df.iterrows():
                att_val = None if pd.isna(row.get("attendance_rate")) else float(row["attendance_rate"])
                records.append(
                    (
                        dataset_id,
                        str(row["period_start"]),
                        str(row["academic_year"]),
                        str(row["academic_term"]),
                        str(row["provision"]),
                        str(row["safeguarding_category"]),
                        int(row["concern_count"]),
                        int(row["enrolled_students"]),
                        int(row["teaching_days"]),
                        att_val,
                        int(row.get("reporting_change_flag", 0)),
                    )
                )
            cursor.executemany(
                """
                INSERT INTO safeguarding_records (
                    dataset_id, period_start, academic_year, academic_term,
                    provision, safeguarding_category, concern_count,
                    enrolled_students, teaching_days, attendance_rate, reporting_change_flag
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                records,
            )
            conn.commit()

    def dataset_exists(self, dataset_id: str) -> bool:
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM datasets WHERE dataset_id = ?", (dataset_id,)
            ).fetchone()
            return row is not None

    def get_dataset_info(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM datasets WHERE dataset_id = ?", (dataset_id,)
            ).fetchone()
            if not row:
                return None
            return {
                "dataset_id": row["dataset_id"],
                "name": row["name"],
                "is_synthetic": bool(row["is_synthetic"]),
                "seed": row["seed"],
                "academic_year_start_month": int(row["academic_year_start_month"]),
                "created_at": row["created_at"],
                "metadata": json.loads(row["metadata_json"] or "{}"),
            }

    def list_datasets(self) -> List[Dict[str, Any]]:
        with self._get_conn() as conn:
            rows = conn.execute(
                """
                SELECT d.*, COUNT(r.id) as row_count, SUM(r.concern_count) as total_concerns
                FROM datasets d
                LEFT JOIN safeguarding_records r ON d.dataset_id = r.dataset_id
                GROUP BY d.dataset_id
                ORDER BY d.is_synthetic DESC, d.created_at DESC
                """
            ).fetchall()
            result = []
            for r in rows:
                result.append(
                    {
                        "dataset_id": r["dataset_id"],
                        "name": r["name"],
                        "is_synthetic": bool(r["is_synthetic"]),
                        "seed": r["seed"],
                        "academic_year_start_month": int(r["academic_year_start_month"]),
                        "created_at": r["created_at"],
                        "row_count": int(r["row_count"] or 0),
                        "total_concerns": int(r["total_concerns"] or 0),
                    }
                )
            return result

    def load_records_df(self, dataset_id: str) -> pd.DataFrame:
        with self._get_conn() as conn:
            df = pd.read_sql_query(
                """
                SELECT period_start, academic_year, academic_term, provision,
                       safeguarding_category, concern_count, enrolled_students,
                       teaching_days, attendance_rate, reporting_change_flag
                FROM safeguarding_records
                WHERE dataset_id = ?
                ORDER BY period_start ASC, provision ASC, safeguarding_category ASC
                """,
                conn,
                params=(dataset_id,),
            )
        return df

    def log_forecast_run(
        self,
        run_id: str,
        dataset_id: str,
        model_id: str,
        model_version: str,
        horizon: int,
        provision: str,
        safeguarding_category: str,
        prediction_interval: int,
        scenario: Dict[str, Any],
        annual_total: int,
        lower_bound: int,
        upper_bound: int,
        runtime_ms: float,
    ) -> None:
        now_str = datetime.now(timezone.utc).isoformat()
        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO forecast_runs (
                    run_id, dataset_id, model_id, model_version, horizon,
                    provision, safeguarding_category, prediction_interval,
                    scenario_json, annual_total, lower_bound, upper_bound,
                    runtime_ms, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    dataset_id,
                    model_id,
                    model_version,
                    horizon,
                    provision,
                    safeguarding_category,
                    prediction_interval,
                    json.dumps(scenario),
                    int(annual_total),
                    int(lower_bound),
                    int(upper_bound),
                    float(runtime_ms),
                    now_str,
                ),
            )
            conn.commit()

    def list_forecast_runs(self, limit: int = 25) -> List[Dict[str, Any]]:
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM forecast_runs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]


db_manager = DatabaseManager()

