"""
DatabaseService — Supabase CRUD operations for explorations.

Uses the service-role key (bypasses RLS) so the backend can operate on behalf
of any authenticated user. Defence-in-depth: every method takes ``user_id``
and includes it in queries to enforce row-level ownership.
"""

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


class DatabaseService:
    """Manages exploration persistence via Supabase PostgreSQL."""

    def __init__(self) -> None:
        self._client = None

    def load(self) -> None:
        """Initialise the Supabase client. Call once at startup."""
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

        if not url or not key:
            logger.warning(
                "SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY not set — "
                "database features disabled"
            )
            return

        from supabase import create_client

        self._client = create_client(url, key)
        logger.info("DatabaseService connected to %s", url)

    @property
    def is_loaded(self) -> bool:
        return self._client is not None

    # ── Explorations ────────────────────────────────────────────

    def list_explorations(self, user_id: str) -> list[dict[str, Any]]:
        """Return summaries of user's explorations, newest first."""
        result = (
            self._client.table("explorations")
            .select("id, title, root_node_id, node_count, created_at, updated_at")
            .eq("user_id", user_id)
            .order("updated_at", desc=True)
            .execute()
        )
        return result.data

    def get_exploration(
        self, user_id: str, exploration_id: str
    ) -> dict[str, Any] | None:
        """Return exploration metadata + all nodes. None if not found/not owned."""
        # Fetch metadata
        meta_result = (
            self._client.table("explorations")
            .select("*")
            .eq("id", exploration_id)
            .eq("user_id", user_id)
            .execute()
        )
        if not meta_result.data:
            return None

        # Fetch nodes ordered by sort_order
        nodes_result = (
            self._client.table("exploration_nodes")
            .select("*")
            .eq("exploration_id", exploration_id)
            .order("sort_order")
            .execute()
        )

        exploration = meta_result.data[0]
        exploration["nodes"] = nodes_result.data
        return exploration

    def create_exploration(
        self, user_id: str, title: str, root_node_id: str
    ) -> dict[str, Any]:
        """Create a new exploration record. Returns the created row."""
        result = (
            self._client.table("explorations")
            .insert({
                "user_id": user_id,
                "title": title[:200],
                "root_node_id": root_node_id,
                "node_count": 0,
            })
            .execute()
        )
        return result.data[0]

    def upsert_nodes(
        self,
        user_id: str,
        exploration_id: str,
        nodes: list[dict[str, Any]],
    ) -> None:
        """
        Batch upsert nodes for an exploration.

        Verifies ownership before writing. Existing nodes (matched by
        exploration_id + node_id) are updated; new nodes are inserted.
        Also updates the parent exploration's node_count.
        """
        # Verify ownership
        owner_check = (
            self._client.table("explorations")
            .select("id")
            .eq("id", exploration_id)
            .eq("user_id", user_id)
            .execute()
        )
        if not owner_check.data:
            raise PermissionError("Exploration not found or not owned by user")

        # Prepare rows with exploration_id and sort_order
        rows = []
        for i, node in enumerate(nodes):
            rows.append({
                "exploration_id": exploration_id,
                "node_id": node["node_id"],
                "node_type": node["node_type"],
                "text": node["text"],
                "parent_node_id": node.get("parent_node_id"),
                "depth": node.get("depth", 0),
                "metadata": node.get("metadata", {}),
                "children": node.get("children", []),
                "sort_order": i,
            })

        # Upsert on the unique (exploration_id, node_id) constraint
        self._client.table("exploration_nodes").upsert(
            rows,
            on_conflict="exploration_id,node_id",
        ).execute()

        # Update node count
        (
            self._client.table("explorations")
            .update({"node_count": len(nodes)})
            .eq("id", exploration_id)
            .eq("user_id", user_id)
            .execute()
        )

    def delete_exploration(self, user_id: str, exploration_id: str) -> bool:
        """Delete an exploration and its nodes (CASCADE). Returns True if deleted."""
        result = (
            self._client.table("explorations")
            .delete()
            .eq("id", exploration_id)
            .eq("user_id", user_id)
            .execute()
        )
        return len(result.data) > 0
