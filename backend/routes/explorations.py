"""
CRUD routes for exploration persistence.

All endpoints are protected by JWT authentication. The DatabaseService
is injected from main.py at startup (same pattern as other route modules).
"""

import logging
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from backend.auth import get_current_user
from backend.services.database import DatabaseService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")

# Injected from main.py at startup
db_service: DatabaseService = None

# ── Pydantic Models ─────────────────────────────────────────


class NodePayload(BaseModel):
    node_id: str
    node_type: str = Field(..., pattern="^(input|question|reflection)$")
    text: str
    parent_node_id: Optional[str] = None
    depth: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)
    children: list[str] = Field(default_factory=list)


class SaveExplorationRequest(BaseModel):
    exploration_id: Optional[str] = None  # None = create new
    title: str = Field(..., min_length=1, max_length=200)
    root_node_id: str
    nodes: list[NodePayload]


class ExplorationSummary(BaseModel):
    id: str
    title: str
    root_node_id: str
    node_count: int
    created_at: str
    updated_at: str


class ExplorationNodeRow(BaseModel):
    node_id: str
    node_type: str
    text: str
    parent_node_id: Optional[str] = None
    depth: int
    metadata: dict[str, Any]
    children: list[str]
    sort_order: int


class ExplorationDetail(BaseModel):
    id: str
    title: str
    root_node_id: str
    node_count: int
    created_at: str
    updated_at: str
    nodes: list[ExplorationNodeRow]


# ── Routes ──────────────────────────────────────────────────


@router.get("/explorations", response_model=list[ExplorationSummary])
async def list_explorations(user_id: str = Depends(get_current_user)):
    """List all explorations for the authenticated user."""
    _check_db()
    return db_service.list_explorations(user_id)


@router.get("/explorations/{exploration_id}", response_model=ExplorationDetail)
async def get_exploration(
    exploration_id: str,
    user_id: str = Depends(get_current_user),
):
    """Get a single exploration with all its nodes."""
    _check_db()
    result = db_service.get_exploration(user_id, exploration_id)
    if not result:
        raise HTTPException(status_code=404, detail="Exploration not found")
    return result


@router.post("/explorations", response_model=ExplorationSummary)
async def save_exploration(
    request: SaveExplorationRequest,
    user_id: str = Depends(get_current_user),
):
    """
    Create or update an exploration.

    If ``exploration_id`` is provided, upserts nodes into the existing
    exploration. Otherwise creates a new exploration first.
    """
    _check_db()

    exploration_id = request.exploration_id

    if not exploration_id:
        # Create new exploration
        created = db_service.create_exploration(
            user_id=user_id,
            title=request.title,
            root_node_id=request.root_node_id,
        )
        exploration_id = created["id"]

    # Upsert all nodes
    try:
        db_service.upsert_nodes(
            user_id=user_id,
            exploration_id=exploration_id,
            nodes=[node.model_dump() for node in request.nodes],
        )
    except PermissionError:
        raise HTTPException(status_code=404, detail="Exploration not found")

    # Return updated summary
    result = db_service.get_exploration(user_id, exploration_id)
    if not result:
        raise HTTPException(status_code=404, detail="Exploration not found")

    return ExplorationSummary(
        id=result["id"],
        title=result["title"],
        root_node_id=result["root_node_id"],
        node_count=result["node_count"],
        created_at=result["created_at"],
        updated_at=result["updated_at"],
    )


@router.delete("/explorations/{exploration_id}")
async def delete_exploration(
    exploration_id: str,
    user_id: str = Depends(get_current_user),
):
    """Delete an exploration and all its nodes."""
    _check_db()
    deleted = db_service.delete_exploration(user_id, exploration_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Exploration not found")
    return {"detail": "Exploration deleted"}


# ── Helpers ─────────────────────────────────────────────────


def _check_db() -> None:
    if not db_service or not db_service.is_loaded:
        raise HTTPException(
            status_code=503, detail="Database not available"
        )
