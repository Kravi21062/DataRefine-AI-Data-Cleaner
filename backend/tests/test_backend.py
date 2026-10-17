from __future__ import annotations

from copy import copy
from datetime import datetime
from typing import Any, Dict, Optional

SESSIONS: Dict[str, Dict[str, Any]] = {}


def create_session() -> str:
    import uuid

    session_id = str(uuid.uuid4())
    SESSIONS[session_id] = {
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
        "raw_df": None,
        "clean_df": None,
        "dataset_name": None,
        "file_info": {},
        "history": [],
    }
    return session_id


def get_session(session_id: str) -> Optional[Dict[str, Any]]:
    return SESSIONS.get(session_id)


def set_dataset(session_id: str, df, dataset_name: str, file_info: Dict[str, Any]):
    session = SESSIONS.get(session_id)
    if session is None:
        raise KeyError("session not found")
    session["raw_df"] = df.copy()
    session["clean_df"] = df.copy()
    session["dataset_name"] = dataset_name
    session["file_info"] = file_info
    session["history"] = []
    session["updated_at"] = datetime.utcnow().isoformat()


def get_active_df(session_id: str):
    session = get_session(session_id)
    if session is None:
        return None
    return session.get("clean_df") or session.get("raw_df")


def set_clean_df(session_id: str, df):
    session = SESSIONS.get(session_id)
    if session is None:
        raise KeyError("session not found")
    session["clean_df"] = df.copy()
    session["updated_at"] = datetime.utcnow().isoformat()
