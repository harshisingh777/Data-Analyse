"""In-memory, size-bounded store of the datasets that are currently being analysed."""

from __future__ import annotations

import threading
import uuid
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd

from .config import Settings
from .modeling import TrainResult
from .profiling import profile_dataframe, semantic_types
from .sql import SqlWorkbench
from .target import infer_target, infer_task
from .utils import DataLensError


@dataclass
class DatasetSession:
    id: str
    name: str
    source: str
    df: pd.DataFrame
    profile: dict
    created_at: str
    model: TrainResult | None = None
    statistics: dict | None = None
    _sql: SqlWorkbench | None = field(default=None, repr=False)
    _sql_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def semantic(self) -> dict[str, str]:
        return semantic_types(self.profile)

    @property
    def sql(self) -> SqlWorkbench:
        with self._sql_lock:
            if self._sql is None:
                self._sql = SqlWorkbench(self.df)
            return self._sql

    def suggested_target(self) -> dict | None:
        target = infer_target(self.df, self.semantic)
        if target is None:
            return None
        try:
            return {"column": target, "task": infer_task(self.df[target])}
        except DataLensError:
            return None

    def meta(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "source": self.source,
            "created_at": self.created_at,
            "rows": int(self.df.shape[0]),
            "columns": int(self.df.shape[1]),
            "suggested_target": self.suggested_target(),
            "has_model": self.model is not None,
        }

    def close(self) -> None:
        if self._sql is not None:
            self._sql.close()


Fallback = Callable[[str], "tuple[str, pd.DataFrame, str] | None"]


class DatasetStore:
    """Least-recently-used cache so memory stays bounded on a long-running server.

    On serverless hosts each instance has its own memory, so a request can land on an instance
    that never saw the dataset. ``fallback`` lets the store rebuild a dataset from its id
    (e.g. a bundled sample) instead of failing.
    """

    def __init__(self, capacity: int | None = None, fallback: Fallback | None = None):
        self.capacity = capacity or Settings.MAX_DATASETS
        self.fallback = fallback
        self._items: OrderedDict[str, DatasetSession] = OrderedDict()
        self._lock = threading.Lock()

    def add(self, name: str, df: pd.DataFrame, source: str = "upload", dataset_id: str | None = None) -> DatasetSession:
        """Store a dataset. Re-adding an id that is already loaded keeps the existing session (and its model)."""
        if dataset_id is not None:
            with self._lock:
                existing = self._items.get(dataset_id)
                if existing is not None:
                    self._items.move_to_end(dataset_id)
                    return existing
        session = DatasetSession(
            id=dataset_id or uuid.uuid4().hex[:12],
            name=name,
            source=source,
            df=df,
            profile=profile_dataframe(df),
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        with self._lock:
            self._items[session.id] = session
            while len(self._items) > self.capacity:
                _, evicted = self._items.popitem(last=False)
                evicted.close()
        return session

    def get(self, dataset_id: str) -> DatasetSession:
        with self._lock:
            session = self._items.get(dataset_id)
            if session is not None:
                self._items.move_to_end(dataset_id)
                return session
        rebuilt = self.fallback(dataset_id) if self.fallback else None
        if rebuilt is None:
            raise DataLensError("Dataset not found. It may have expired; please load it again.", 404)
        name, df, source = rebuilt
        return self.add(name, df, source=source, dataset_id=dataset_id)

    def remove(self, dataset_id: str) -> None:
        with self._lock:
            session = self._items.pop(dataset_id, None)
        if session is None:
            raise DataLensError("Dataset not found.", 404)
        session.close()

    def __len__(self) -> int:
        return len(self._items)
