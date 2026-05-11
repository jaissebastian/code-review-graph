"""Tools 21-23: list_repos_func, cross_repo_search_func, cross_repo_callers_func."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..graph import GraphStore, node_to_dict
from ..incremental import get_db_path
from ..search import hybrid_search

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool 21: list_repos  [REGISTRY]
# ---------------------------------------------------------------------------


def list_repos_func(detail_level: str = "standard") -> dict[str, Any]:
    """List all registered repositories.

    [REGISTRY] Returns the list of repositories registered in the global
    multi-repo registry at ``~/.code-review-graph/registry.json``.

    Args:
        detail_level: "standard" returns full repo metadata; "minimal"
            returns only alias and path per repo.

    Returns:
        List of registered repos with paths and aliases.
    """
    from ..registry import Registry

    try:
        registry = Registry()
        repos = registry.list_repos()
        if detail_level == "minimal":
            repos = [
                {k: r[k] for k in ("alias", "path") if k in r}
                for r in repos
            ]
        return {
            "status": "ok",
            "summary": f"{len(repos)} registered repository(ies)",
            "repos": repos,
        }
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


# ---------------------------------------------------------------------------
# Tool 22: cross_repo_search  [REGISTRY]
# ---------------------------------------------------------------------------


def cross_repo_search_func(
    query: str,
    kind: str | None = None,
    limit: int = 20,
    detail_level: str = "standard",
) -> dict[str, Any]:
    """Search across all registered repositories.

    [REGISTRY] Runs hybrid_search on each registered repo's graph database
    and merges the results.

    Args:
        query: Search query string.
        kind: Optional node kind filter (e.g. "Function", "Class").
        limit: Maximum results per repo (default: 20).
        detail_level: "standard" returns full node data; "minimal" returns
            only name, kind, repo, and file_path per result.

    Returns:
        Combined search results from all registered repos.
    """
    from ..registry import Registry

    try:
        registry = Registry()
        repos = registry.list_repos()
        if not repos:
            return {
                "status": "ok",
                "summary": (
                    "No repositories registered. "
                    "Use 'register' to add repos."
                ),
                "results": [],
            }

        all_results: list[dict[str, Any]] = []
        searched_repos: list[str] = []

        for repo_entry in repos:
            repo_path = Path(repo_entry["path"])
            db_path = get_db_path(repo_path)
            if not db_path.exists():
                continue

            try:
                store = GraphStore(str(db_path))
                try:
                    results = hybrid_search(
                        store, query, kind=kind, limit=limit
                    )
                    alias = repo_entry.get("alias", repo_path.name)
                    for r in results:
                        r["repo"] = alias
                        r["repo_path"] = str(repo_path)
                    all_results.extend(results)
                    searched_repos.append(alias)
                finally:
                    store.close()
            except Exception as exc:
                logger.warning(
                    "Search failed for %s: %s", repo_path, exc
                )

        # Sort all results by score descending
        all_results.sort(
            key=lambda r: r.get("score", 0), reverse=True
        )

        trimmed = all_results[:limit]
        if detail_level == "minimal":
            trimmed = [
                {k: r[k] for k in ("name", "kind", "repo", "file_path") if k in r}
                for r in trimmed
            ]

        return {
            "status": "ok",
            "summary": (
                f"Found {len(all_results)} result(s) across "
                f"{len(searched_repos)} repo(s) for '{query}'"
            ),
            "results": trimmed,
            "repos_searched": searched_repos,
        }
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


# ---------------------------------------------------------------------------
# Tool 23: cross_repo_callers  [REGISTRY]
# ---------------------------------------------------------------------------


def cross_repo_callers_func(
    symbol: str,
    limit: int = 50,
    detail_level: str = "standard",
) -> dict[str, Any]:
    """Find all callers of a symbol across every registered repository.

    [REGISTRY] Iterates every repo in the registry, runs a callers_of query
    on each graph database, and returns a merged, repo-annotated result set.
    Useful for answering "what calls ServiceA.processPayment across all services?"
    in a single call.

    Resolution order per repo:
      1. Exact qualified-name match (e.g. ``com.example.ServiceA.processPayment``)
      2. Prefix/substring search via ``store.search_nodes`` → first candidate
      3. Bare-name fallback for CALLS edges that store unqualified target names

    Args:
        symbol: Qualified name or plain name of the target symbol.
        limit: Maximum total results to return across all repos. Default: 50.
        detail_level: ``"standard"`` for full node data; ``"minimal"`` for
            compact output (name, kind, file_path, repo, edge_confidence only).

    Returns:
        Merged caller list with ``repo`` and ``repo_path`` annotations per result.
    """
    from ..registry import Registry

    try:
        registry = Registry()
        repos = registry.list_repos()
        if not repos:
            return {
                "status": "ok",
                "summary": (
                    "No repositories registered. "
                    "Use 'register' to add repos."
                ),
                "symbol": symbol,
                "results": [],
                "total_callers": 0,
                "results_omitted": 0,
                "repos_searched": [],
                "repos_with_callers": [],
            }

        all_callers: list[dict[str, Any]] = []
        repos_searched: list[str] = []
        repos_with_callers: list[str] = []

        for repo_entry in repos:
            repo_path = Path(repo_entry["path"])
            db_path = get_db_path(repo_path)
            if not db_path.exists():
                continue

            alias = repo_entry.get("alias", repo_path.name)

            try:
                store = GraphStore(str(db_path))
                try:
                    repos_searched.append(alias)
                    repo_callers = _callers_in_store(store, symbol)
                    for c in repo_callers:
                        c["repo"] = alias
                        c["repo_path"] = str(repo_path)
                    all_callers.extend(repo_callers)
                    if repo_callers:
                        repos_with_callers.append(alias)
                finally:
                    store.close()
            except Exception as exc:
                logger.warning(
                    "cross_repo_callers failed for %s: %s", repo_path, exc
                )

        total = len(all_callers)
        trimmed = all_callers[:limit]

        if detail_level == "minimal":
            trimmed = [
                {
                    k: r[k]
                    for k in ("name", "kind", "file_path", "repo", "edge_confidence")
                    if k in r
                }
                for r in trimmed
            ]

        return {
            "status": "ok",
            "summary": (
                f"Found {total} caller(s) of '{symbol}' across "
                f"{len(repos_with_callers)} repo(s)"
            ),
            "symbol": symbol,
            "results": trimmed,
            "total_callers": total,
            "results_omitted": max(0, total - limit),
            "repos_searched": repos_searched,
            "repos_with_callers": repos_with_callers,
        }
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


def _callers_in_store(store: GraphStore, symbol: str) -> list[dict[str, Any]]:
    """Return caller node dicts for *symbol* from a single GraphStore.

    Resolution order:
    1. CALLS edges by exact qualified name
    2. CALLS edges by bare name (fallback for unqualified targets)
    3. TEMPORAL_STUB edges by bare name (Temporal workflow/activity cross-repo)
    """
    # Step 1: resolve symbol → qualified name.
    # Prefer Class/Function nodes over File/Module nodes so that interface→impl
    # resolution (Step 5) and bare-name callers (Step 3) work correctly.
    node = store.get_node(symbol)
    if not node:
        candidates = store.search_nodes(symbol, limit=10)
        non_file = [c for c in candidates if c.kind not in ("File", "Module")]
        node = non_file[0] if non_file else (candidates[0] if candidates else None)
    qn = node.qualified_name if node else symbol
    # For TEMPORAL_STUB lookup: always derive bare name from the *input* symbol
    # rather than the resolved node, because the Temporal interface may live in a
    # different repo and won't be found by search_nodes in this store.
    bare_for_temporal = symbol.split(".")[-1]

    callers: list[dict[str, Any]] = []
    seen: set[str] = set()

    # Step 2: primary lookup — CALLS edges where target_qualified == qn
    for edge in store.get_edges_by_target(qn):
        if edge.kind != "CALLS" or edge.source_qualified in seen:
            continue
        caller_node = store.get_node(edge.source_qualified)
        if caller_node:
            d = node_to_dict(caller_node)
            d["edge_confidence"] = edge.confidence
            d["edge_line"] = edge.line
            d["edge_kind"] = "CALLS"
            callers.append(d)
            seen.add(edge.source_qualified)

    # Step 3: bare-name fallback — CALLS edges sometimes store unqualified target names
    if not callers and node:
        for edge in store.search_edges_by_target_name(node.name, kind="CALLS"):
            if edge.source_qualified in seen:
                continue
            caller_node = store.get_node(edge.source_qualified)
            if caller_node:
                d = node_to_dict(caller_node)
                d["edge_confidence"] = edge.confidence
                d["edge_line"] = edge.line
                d["edge_kind"] = "CALLS"
                callers.append(d)
                seen.add(edge.source_qualified)

    # Step 4: Temporal workflow/activity stubs — TEMPORAL_STUB edges point to bare
    # interface names (e.g. "PaymentActivity") and live in a different repo than the
    # interface definition, making this the primary cross-repo Temporal signal.
    for edge in store.search_edges_by_target_name(bare_for_temporal, kind="TEMPORAL_STUB"):
        if edge.source_qualified in seen:
            continue
        caller_node = store.get_node(edge.source_qualified)
        if caller_node:
            d = node_to_dict(caller_node)
            d["edge_confidence"] = edge.confidence
            d["edge_line"] = edge.line
            d["edge_kind"] = "TEMPORAL_STUB"
            d["relationship"] = "temporal"
            callers.append(d)
            seen.add(edge.source_qualified)

    # Step 5: Interface → implementation resolution.
    # When a symbol is a Java interface/abstract class, CALLS edges target the
    # concrete implementation (e.g. ServiceImpl), not the interface name.
    # Find implementations via INHERITS edges, then fetch their callers.
    # Use three candidate target names: full qn, node.name, and bare input name.
    # The INHERITS edge stores the bare class name (e.g. "ServiceA") regardless of
    # which repo's graph is being searched, so the bare fallback is essential.
    if not callers and node:
        impl_rows = store._conn.execute(
            "SELECT DISTINCT source_qualified FROM edges"
            " WHERE kind = 'INHERITS'"
            " AND (target_qualified = ? OR target_qualified = ? OR target_qualified = ?)",
            (qn, node.name, bare_for_temporal),
        ).fetchall()
        for (impl_qn,) in impl_rows:
            # Extract bare implementation class name for LIKE and bare-name search
            impl_bare = impl_qn.split("::")[-1].split(".")[-1]
            # CALLS edges to any method of this implementation (dot-qualified form)
            impl_call_rows = store._conn.execute(
                "SELECT source_qualified, confidence, line FROM edges"
                " WHERE kind = 'CALLS' AND ("
                "  target_qualified = ? OR"
                "  target_qualified LIKE ? OR"
                "  target_qualified LIKE ?"
                ")",
                (impl_qn, f"{impl_bare}.%", f"{impl_qn}.%"),
            ).fetchall()
            for src_qn, confidence, line in impl_call_rows:
                if src_qn in seen:
                    continue
                caller_node = store.get_node(src_qn)
                if caller_node:
                    d = node_to_dict(caller_node)
                    d["edge_confidence"] = confidence
                    d["edge_line"] = line
                    d["edge_kind"] = "CALLS"
                    d["via_implementation"] = impl_bare
                    callers.append(d)
                    seen.add(src_qn)

    return callers


# ---------------------------------------------------------------------------
# Tool 24: cross_repo_kafka_impact  [REGISTRY]
# ---------------------------------------------------------------------------


def cross_repo_kafka_impact_func(
    topic_or_type: str,
    limit: int = 50,
) -> dict[str, Any]:
    """Find all producers and consumers of a Kafka topic or message type across repos.

    [REGISTRY] Scans CONSUMES and PRODUCES edges across every registered repo
    and matches by:
    - Topic name in ``target_qualified`` (e.g. ``kafka:order.created`` or
      ``kafka:${order.topic}``)
    - OR ``message_type`` in the edge ``extra`` JSON
      (e.g. ``OrderEvent``, ``PaymentEvent``)

    Useful for tracing: "which services publish or consume the
    ``OrderEvent`` event?" across a multi-repo layout.

    Args:
        topic_or_type: Topic name fragment or message type class name.
        limit: Maximum results per role (producers / consumers). Default: 50.

    Returns:
        Dict with ``producers``, ``consumers``, and ``repos_searched``.
    """
    from ..registry import Registry

    try:
        registry = Registry()
        repos = registry.list_repos()
        if not repos:
            return {
                "status": "ok",
                "summary": "No repositories registered.",
                "topic_or_type": topic_or_type,
                "producers": [],
                "consumers": [],
                "repos_searched": [],
            }

        all_producers: list[dict[str, Any]] = []
        all_consumers: list[dict[str, Any]] = []
        repos_searched: list[str] = []
        query_lower = topic_or_type.lower()

        for repo_entry in repos:
            repo_path = Path(repo_entry["path"])
            db_path = get_db_path(repo_path)
            if not db_path.exists():
                continue

            alias = repo_entry.get("alias", repo_path.name)

            try:
                store = GraphStore(str(db_path))
                try:
                    repos_searched.append(alias)
                    producers, consumers = _kafka_edges_in_store(
                        store, query_lower, alias, str(repo_path)
                    )
                    all_producers.extend(producers)
                    all_consumers.extend(consumers)
                finally:
                    store.close()
            except Exception as exc:
                logger.warning(
                    "cross_repo_kafka_impact failed for %s: %s", repo_path, exc
                )

        total = len(all_producers) + len(all_consumers)
        return {
            "status": "ok",
            "summary": (
                f"Found {len(all_producers)} producer(s) and {len(all_consumers)} "
                f"consumer(s) of '{topic_or_type}' across {len(repos_searched)} repo(s)"
            ),
            "topic_or_type": topic_or_type,
            "producers": all_producers[:limit],
            "consumers": all_consumers[:limit],
            "total_matches": total,
            "repos_searched": repos_searched,
        }
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


def _kafka_edges_in_store(
    store: GraphStore,
    query_lower: str,
    alias: str,
    repo_path_str: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (producers, consumers) matching *query_lower* from a single GraphStore."""
    import json

    producers: list[dict[str, Any]] = []
    consumers: list[dict[str, Any]] = []

    # Query all CONSUMES/PRODUCES edges and filter in Python for flexibility
    rows = store._conn.execute(
        "SELECT source_qualified, target_qualified, kind, extra, confidence, line"
        " FROM edges WHERE kind IN ('CONSUMES', 'PRODUCES')"
    ).fetchall()

    for row in rows:
        src, tgt, kind, extra_raw, confidence, line = (
            row[0], row[1], row[2], row[3], row[4], row[5]
        )
        # Match by topic name in target_qualified
        topic_match = query_lower in tgt.lower()
        # Match by message_type in extra JSON
        msg_match = False
        extra: dict = {}
        try:
            extra = json.loads(extra_raw) if extra_raw else {}
            msg_type = extra.get("message_type", "")
            msg_match = bool(msg_type) and query_lower in msg_type.lower()
        except (json.JSONDecodeError, TypeError):
            pass

        if not (topic_match or msg_match):
            continue

        # Resolve the source node for metadata
        source_node = store.get_node(src)
        entry: dict[str, Any] = {
            "source": src,
            "target": tgt,
            "edge_kind": kind,
            "confidence": confidence,
            "line": line,
            "repo": alias,
            "repo_path": repo_path_str,
            "extra": extra,
        }
        if source_node:
            entry["name"] = source_node.name
            entry["file_path"] = source_node.file_path
            entry["kind"] = source_node.kind

        if kind == "PRODUCES":
            producers.append(entry)
        else:
            consumers.append(entry)

    return producers, consumers


# ---------------------------------------------------------------------------
# Tool 25: cross_repo_rest_callers  [REGISTRY]
# ---------------------------------------------------------------------------


def cross_repo_rest_callers_func(
    path: str,
    limit: int = 50,
) -> dict[str, Any]:
    """Find REST callers and endpoint implementations for a path across all registered repos.

    [REGISTRY] Matches:
    - ``REST_CALLS`` edges whose target is ``rest:path:<path>`` (callers using WebClient)
    - Nodes whose ``extra.rest_endpoint`` equals ``<path>`` (``@RestController`` endpoints)

    Useful for answering "which services call ``/orders`` and which services expose it?"
    across a multi-repo layout.

    Args:
        path: REST path fragment to search for, e.g. ``/orders``, ``/payments``.
        limit: Maximum results per role (callers / endpoints). Default: 50.

    Returns:
        Dict with ``callers``, ``endpoints``, and ``repos_searched``.
    """
    from ..registry import Registry

    try:
        registry = Registry()
        repos = registry.list_repos()
        if not repos:
            return {
                "status": "ok",
                "summary": "No repositories registered.",
                "path": path,
                "callers": [],
                "endpoints": [],
                "repos_searched": [],
            }

        normalized = path.rstrip("/") or "/"
        all_callers: list[dict[str, Any]] = []
        all_endpoints: list[dict[str, Any]] = []
        repos_searched: list[str] = []

        for repo_entry in repos:
            repo_path = Path(repo_entry["path"])
            db_path = get_db_path(repo_path)
            if not db_path.exists():
                continue

            alias = repo_entry.get("alias", repo_path.name)

            try:
                store = GraphStore(str(db_path))
                try:
                    repos_searched.append(alias)
                    callers, endpoints = _rest_matches_in_store(
                        store, normalized, alias, str(repo_path)
                    )
                    all_callers.extend(callers)
                    all_endpoints.extend(endpoints)
                finally:
                    store.close()
            except Exception as exc:
                logger.warning(
                    "cross_repo_rest_callers failed for %s: %s", repo_path, exc
                )

        return {
            "status": "ok",
            "summary": (
                f"Found {len(all_callers)} caller(s) and {len(all_endpoints)} "
                f"endpoint(s) for '{path}' across {len(repos_searched)} repo(s)"
            ),
            "path": path,
            "callers": all_callers[:limit],
            "endpoints": all_endpoints[:limit],
            "total_callers": len(all_callers),
            "total_endpoints": len(all_endpoints),
            "repos_searched": repos_searched,
        }
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


def _rest_matches_in_store(
    store: GraphStore,
    path: str,
    alias: str,
    repo_path_str: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (callers, endpoints) for *path* from a single GraphStore."""
    import json

    callers: list[dict[str, Any]] = []
    endpoints: list[dict[str, Any]] = []

    # REST_CALLS edges where target_qualified == "rest:path:<path>"
    target = f"rest:path:{path}"
    for edge in store.get_edges_by_target(target):
        if edge.kind != "REST_CALLS":
            continue
        caller_node = store.get_node(edge.source_qualified)
        if caller_node:
            d = node_to_dict(caller_node)
            d["repo"] = alias
            d["repo_path"] = repo_path_str
            d["called_path"] = path
            d["edge_line"] = edge.line
            callers.append(d)

    # Nodes with extra.rest_endpoint == path (@RestController annotation style)
    rows = store._conn.execute(
        "SELECT qualified_name, kind, name, file_path, line_start, line_end, extra"
        " FROM nodes WHERE extra IS NOT NULL AND extra LIKE '%rest_endpoint%'"
    ).fetchall()
    for row in rows:
        qn, kind, name, fp, ls, le, extra_raw = row
        try:
            extra = json.loads(extra_raw) if extra_raw else {}
        except (json.JSONDecodeError, TypeError):
            extra = {}
        if extra.get("rest_endpoint") == path:
            endpoints.append({
                "qualified_name": qn,
                "kind": kind,
                "name": name,
                "file_path": fp,
                "line_start": ls,
                "http_method": extra.get("http_method", "ANY"),
                "rest_endpoint": path,
                "repo": alias,
                "repo_path": repo_path_str,
            })

    # REST_ENDPOINT edges (Spring WebFlux functional RouterFunction style)
    seen_endpoints: set[str] = {e["qualified_name"] for e in endpoints}
    for edge in store.get_edges_by_target(target):
        if edge.kind != "REST_ENDPOINT":
            continue
        if edge.source_qualified in seen_endpoints:
            continue
        endpoint_node = store.get_node(edge.source_qualified)
        if endpoint_node:
            http_method = edge.extra.get("http_method", "ANY") if edge.extra else "ANY"
            endpoints.append({
                "qualified_name": edge.source_qualified,
                "kind": endpoint_node.kind,
                "name": endpoint_node.name,
                "file_path": endpoint_node.file_path,
                "line_start": endpoint_node.line_start,
                "http_method": http_method,
                "rest_endpoint": path,
                "repo": alias,
                "repo_path": repo_path_str,
            })
            seen_endpoints.add(edge.source_qualified)

    # HANDLES edges — emitted for WebFlux static-predicate pattern:
    #   RouterFunctions.route(POST("/path"), handler::method)
    # target format: "http:<METHOD>:<path>" (note: different from rest:path:<path>)
    # We match any HTTP method by using a LIKE query over the path suffix.
    handles_rows = store._conn.execute(
        "SELECT source_qualified, target_qualified, extra FROM edges"
        " WHERE kind = 'HANDLES' AND target_qualified LIKE ?",
        (f"http:%:{path}",),
    ).fetchall()
    for src_qn, tgt_qn, extra_raw in handles_rows:
        if src_qn in seen_endpoints:
            continue
        # Extract HTTP method from target_qualified ("http:POST:/invoice" → "POST")
        parts = tgt_qn.split(":", 2)
        http_method = parts[1] if len(parts) >= 3 else "ANY"
        endpoint_node = store.get_node(src_qn)
        if endpoint_node:
            endpoints.append({
                "qualified_name": src_qn,
                "kind": endpoint_node.kind,
                "name": endpoint_node.name,
                "file_path": endpoint_node.file_path,
                "line_start": endpoint_node.line_start,
                "http_method": http_method,
                "rest_endpoint": path,
                "repo": alias,
                "repo_path": repo_path_str,
            })
            seen_endpoints.add(src_qn)

    return callers, endpoints
