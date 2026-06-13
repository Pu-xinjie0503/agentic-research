"""长期记忆 Store 单例。"""

import os
from pathlib import Path

from app.memory.store import SQLiteMemoryStore


APP_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MEMORY_DB = APP_ROOT / "data" / "memory.sqlite3"

memory_store = SQLiteMemoryStore(
    os.getenv("DEEPSEARCH_MEMORY_DB", str(DEFAULT_MEMORY_DB))
)
