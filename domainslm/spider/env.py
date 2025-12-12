import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Self

import sqlparse
from langchain_community.utilities import SQLDatabase
from pydantic import BaseModel

from spider_eval.exec_eval import exec_on_db, postprocess, result_eq

type SpiderSplit = Literal["train", "test"]


class EnvironmentError(BaseException):
    pass


class ExecutionEvalResult(BaseModel):
    match: int
    execution_result: str
    gold_execution_result: str
    error_message: str | None = None


@dataclass
class SpiderEnvironment:
    db_id: str
    db_root: Path
    temp_dir: Path | None = None
    truncation_limit: int = 1500

    @classmethod
    def from_db_id(cls, db_id: str, split: SpiderSplit) -> Self:
        spider_dir = Path("data")
        if split == "train":
            db_path = spider_dir / "database" / db_id / f"{db_id}.sqlite"
        else:
            db_path = spider_dir / "test_database" / db_id / f"{db_id}.sqlite"
        return cls(db_id=db_id, db_root=db_path)

    def __enter__(self) -> Self:
        self._temp_dir = tempfile.TemporaryDirectory()
        self.temp_dir = Path(self._temp_dir.name)
        shutil.copy(self.db_root, self.temp_dir / f"{self.db_id}.sqlite")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self.temp_dir is not None:
            self._temp_dir.cleanup()
            self.temp_dir = None

    def get_db_temp_path(self) -> Path:
        if self.temp_dir is None:
            raise RuntimeError("SpiderEnvironment must be used as a context manager.")
        return self.temp_dir / f"{self.db_id}.sqlite"

    async def execute_query(self, query: str) -> tuple[str, Any]:
        if self.temp_dir is None:
            raise RuntimeError("SpiderEnvironment must be used as a context manager.")
        db_path = self.temp_dir / f"{self.db_id}.sqlite"
        return await exec_on_db(
            sqlite_path=db_path,
            query=query,
        )

    async def evaluate(
        self,
        gold_query: str,
        predicted_query: str,
    ) -> ExecutionEvalResult:
        p_str = postprocess(predicted_query)
        g_str = postprocess(gold_query)
        order_matters = "order by" in g_str.lower()

        match = 1
        exec_result_str = ""
        gold_result_str = ""
        error_msg = None

        # Execute Gold
        # We expect gold to always succeed
        g_flag, g_res = await self.execute_query(g_str)
        if g_flag == "exception":
            raise EnvironmentError("Gold query failed")

        # Execute Pred
        p_flag, p_res = await self.execute_query(p_str)

        gold_result_str = str(g_res)[: self.truncation_limit]
        if p_flag == "exception":
            error_msg = f"{type(p_res).__name__}: {p_res}"
            exec_result_str = error_msg
        else:
            exec_result_str = str(p_res)[: self.truncation_limit]

        # Check correctness
        if p_flag == "exception":
            match = 0
        elif not result_eq(g_res, p_res, order_matters=order_matters):
            match = 0

        return ExecutionEvalResult(
            match=match,
            execution_result=exec_result_str,
            gold_execution_result=gold_result_str,
            error_message=error_msg,
        )

    def get_table_info(self) -> str:
        schema_path = self.db_root.parent / "schema.sql"
        if schema_path.exists():
            return schema_path.read_text()
        try:
            db = SQLDatabase.from_uri(f"sqlite:///{self.get_db_temp_path()}")
            return remove_inserts(db.get_table_info())
        except Exception as e:
            raise EnvironmentError(f"Failed to get table info: {e}")

    def dialect(self) -> str:
        db = SQLDatabase.from_uri(f"sqlite:///{self.get_db_temp_path()}")
        return db.dialect


def remove_inserts(input_sql: str) -> str:
    # Parse the SQL into a list of statement objects
    parsed = sqlparse.parse(input_sql)

    out = ""

    for statement in parsed:
        # Get the type of the statement (INSERT, UPDATE, DELETE, CREATE, etc.)
        stmt_type = statement.get_type().upper()

        # We skip INSERT, UPDATE, DELETE. We keep CREATE, ALTER, DROP, PRAGMA.
        if stmt_type not in ("INSERT", "UPDATE", "DELETE"):
            # str(statement) reconstructs the proper SQL from the tokens
            out += str(statement).strip() + "\n"
    return out
