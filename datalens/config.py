"""Central configuration values. Each can be overridden with an environment variable."""

import os


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


class Settings:
    MAX_UPLOAD_MB = _int_env("DATALENS_MAX_UPLOAD_MB", 50)
    MAX_DATASETS = _int_env("DATALENS_MAX_DATASETS", 8)
    MAX_TRAIN_ROWS = _int_env("DATALENS_MAX_TRAIN_ROWS", 20_000)
    MAX_CORRELATION_COLUMNS = _int_env("DATALENS_MAX_CORRELATION_COLUMNS", 25)
    SQL_ROW_LIMIT = _int_env("DATALENS_SQL_ROW_LIMIT", 1_000)
    SQL_TIMEOUT_SECONDS = _int_env("DATALENS_SQL_TIMEOUT_SECONDS", 5)
    RANDOM_STATE = 42
