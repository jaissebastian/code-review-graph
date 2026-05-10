"""Tools 2, 3, 5, 6, 9: query / search / stats helpers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..embeddings import EmbeddingStore
from ..graph import GraphNode, _sanitize_name, edge_to_dict, node_to_dict
from ..hints import generate_hints, get_session
from ..incremental import get_changed_files, get_db_path, get_staged_and_unstaged
from ..search import hybrid_search
from ._common import _BUILTIN_CALL_NAMES, _get_store, graph_meta

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool 2: get_impact_radius
# ---------------------------------------------------------------------------

_QUERY_PATTERNS = {
    "callers_of": "Find all functions that call a given function",
    "callees_of": "Find all functions called by a given function",
    "imports_of": "Find all imports of a given file or module",
    "importers_of": "Find all files that import a given file or module",
    "children_of": "Find all nodes contained in a file or class",
    "tests_for": "Find all tests for a given function or class",
    "inheritors_of": "Find all classes that inherit from a given class",
    "consumers_of": "Find all classes/methods that read a given config property",
    "file_summary": "Get a summary of all nodes in a file",
}


def get_impact_radius(
    changed_files: list[str] | None = None,
    max_depth: int = 2,
    max_results: int = 500,
    repo_root: str | None = None,
    base: str = "HEAD~1",
    detail_level: str = "standard",
) -> dict[str, Any]:
    """Analyze the blast radius of changed files.

    Args:
        changed_files: Explicit list of changed file paths (relative to repo root).
                       If omitted, auto-detects from git diff.
        max_depth: How many hops to traverse in the graph (default: 2).
        max_results: Maximum impacted nodes to return (default: 500).
        repo_root: Repository root path. Auto-detected if omitted.
        base: Git ref for auto-detecting changes (default: HEAD~1).
        detail_level: "standard" (full output) or "minimal" (summary only).

    Returns:
        Changed nodes, impacted nodes, impacted files, connecting edges,
        plus ``truncated`` flag and ``total_impacted`` count.
    """
    store, root = _get_store(repo_root)
    try:
        if changed_files is None:
            changed_files = get_changed_files(root, base)
            if not changed_files:
                changed_files = get_staged_and_unstaged(root)

        if not changed_files:
            return {
                "status": "ok",
                "summary": "No changed files detected.",
                "changed_nodes": [],
                "impacted_nodes": [],
                "impacted_files": [],
                "truncated": False,
                "total_impacted": 0,
            }

        # Convert to absolute paths for graph lookup
        abs_files = [str(root / f) for f in changed_files]
        result = store.get_impact_radius(
            abs_files, max_depth=max_depth, max_nodes=max_results
        )

        changed_dicts = [node_to_dict(n) for n in result["changed_nodes"]]
        impacted_dicts = [node_to_dict(n) for n in result["impacted_nodes"]]
        edge_dicts = [edge_to_dict(e) for e in result["edges"]]
        truncated = result["truncated"]
        total_impacted = result["total_impacted"]

        summary_parts = [
            f"Blast radius for {len(changed_files)} changed file(s):",
            f"  - {len(changed_dicts)} nodes directly changed",
            f"  - {len(impacted_dicts)} nodes impacted (within {max_depth} hops)",
            f"  - {len(result['impacted_files'])} additional files affected",
        ]
        if truncated:
            summary_parts.append(
                f"  - Results truncated: showing {len(impacted_dicts)}"
                f" of {total_impacted} impacted nodes"
            )

        meta = graph_meta(store, root)

        if detail_level == "minimal":
            impacted_count = len(impacted_dicts)
            if impacted_count > 20:
                risk = "high"
            elif impacted_count > 5:
                risk = "medium"
            else:
                risk = "low"
            key_entities = [n["name"] for n in impacted_dicts[:5]]
            nodes_omitted = max(0, total_impacted - len(impacted_dicts))
            return {
                "status": "ok",
                "summary": "\n".join(summary_parts),
                "risk": risk,
                "impacted_file_count": len(result["impacted_files"]),
                "key_entities": key_entities,
                "truncated": truncated,
                "nodes_omitted": nodes_omitted,
                "graph_meta": meta,
            }

        return {
            "status": "ok",
            "summary": "\n".join(summary_parts),
            "changed_files": changed_files,
            "changed_nodes": changed_dicts,
            "impacted_nodes": impacted_dicts,
            "impacted_files": result["impacted_files"],
            "edges": edge_dicts,
            "truncated": truncated,
            "total_impacted": total_impacted,
            "graph_meta": meta,
        }
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Tool 3: query_graph
# ---------------------------------------------------------------------------


def _rank_disambiguation_candidates(candidates: list[GraphNode], target: str) -> list[dict]:
    """Rank disambiguation candidates by match quality and return enriched dicts.

    Ranking: exact qualified_name match > exact name match > partial match.
    Each entry includes qualified_name, name, kind, file_path, and line_start
    so the agent can pick the right one without a follow-up call.
    """
    def _score(node: GraphNode) -> int:
        if node.qualified_name == target:
            return 0
        if node.name == target:
            return 1
        if target.lower() in node.qualified_name.lower():
            return 2
        return 3

    ranked = sorted(candidates, key=_score)
    return [
        {
            "qualified_name": _sanitize_name(n.qualified_name),
            "name": _sanitize_name(n.name),
            "kind": n.kind,
            "file_path": n.file_path,
            "line_start": n.line_start,
        }
        for n in ranked
    ]


def query_graph(
    pattern: str,
    target: str,
    repo_root: str | None = None,
    detail_level: str = "standard",
    max_results: int = 100,
) -> dict[str, Any]:
    """Run a predefined graph query.

    Args:
        pattern: Query pattern. One of: callers_of, callees_of, imports_of,
                 importers_of, children_of, tests_for, inheritors_of, file_summary.
        target: The node name, qualified name, or file path to query about.
        repo_root: Repository root path. Auto-detected if omitted.
        detail_level: "standard" (full output) or "minimal" (summary only).
        max_results: Cap on results returned (default 100). Excess count
            is reported in ``results_omitted`` so agents know when to refine.

    Returns:
        Matching nodes and edges for the query, plus graph_meta for staleness
        awareness and results_omitted when the cap is hit.
    """
    store, root = _get_store(repo_root)
    try:
        meta = graph_meta(store, root)

        if pattern not in _QUERY_PATTERNS:
            return {
                "status": "error",
                "error": (
                    f"Unknown pattern '{pattern}'. "
                    f"Available: {list(_QUERY_PATTERNS.keys())}"
                ),
            }

        results: list[dict] = []
        edges_out: list[dict] = []

        # For callers_of, skip common builtins early (bare names only)
        # "Who calls .map()?" returns hundreds of useless hits.
        # Qualified names (e.g. "utils.py::map") bypass this filter.
        if (
            pattern == "callers_of"
            and target in _BUILTIN_CALL_NAMES
            and "::" not in target
        ):
            return {
                "status": "ok", "pattern": pattern, "target": target,
                "description": _QUERY_PATTERNS[pattern],
                "summary": (
                    f"'{target}' is a common builtin "
                    "— callers_of skipped to avoid noise."
                ),
                "results": [], "edges": [], "graph_meta": meta,
            }

        # Resolve target — four-step cascade:
        # 1. Exact graph node lookup (file-path QN, e.g. "/repo/Foo.java::Bar.method")
        # 2. Absolute path lookup (for file nodes passed as relative paths)
        # 3. FTS name search on the raw target string
        # 4. Java FQN decomposition: "a.b.Class.method" → try "Class.method",
        #    then "method" — agents and IDEs naturally write Java FQNs; the graph
        #    stores file-path QNs, so without this step a valid node is silently
        #    missed and the response says not_found instead of finding the symbol.
        node = store.get_node(target)
        if not node:
            abs_target = str(root / target)
            node = store.get_node(abs_target)

        candidates: list = []
        if not node:
            candidates = store.search_nodes(target, limit=10)

        # Step 4: Java FQN fallback — only runs when FTS returned nothing and the
        # target looks like a dotted package name (no "::" means it's not already
        # a file-path QN; multiple dots means it's not a bare name).
        if not node and not candidates and "::" not in target and target.count(".") >= 1:
            parts = target.split(".")
            # Try progressively shorter suffixes: "Class.method", then "method"
            for width in (2, 1):
                short = ".".join(parts[-width:])
                candidates = store.search_nodes(short, limit=10)
                if candidates:
                    break

        if not node:
            if len(candidates) == 1:
                node = candidates[0]
                target = node.qualified_name
            elif len(candidates) > 1:
                ranked = _rank_disambiguation_candidates(candidates, target)
                return {
                    "status": "ambiguous",
                    "summary": (
                        f"'{target}' matches {len(candidates)} node(s). "
                        "Re-run with the qualified_name from the "
                        "disambiguation list."
                    ),
                    "disambiguation": ranked,
                    "hint": (
                        "Use the 'qualified_name' field from one of the entries "
                        "above as the 'target' parameter."
                    ),
                    "graph_meta": meta,
                }

        # consumers_of can resolve via config edge keys even without a node
        if not node and pattern not in ("file_summary", "consumers_of"):
            return {
                "status": "not_found",
                "summary": f"No node found matching '{target}'.",
                "graph_meta": meta,
            }

        qn = node.qualified_name if node else target

        if pattern == "callers_of":
            for e in store.get_edges_by_target(qn):
                if e.kind == "CALLS":
                    caller = store.get_node(e.source_qualified)
                    if caller:
                        results.append(node_to_dict(caller))
                    edges_out.append(edge_to_dict(e))
            # Fallback: CALLS edges store unqualified target names
            # (e.g. "generateTestCode") while qn is fully qualified
            # (e.g. "file.ts::generateTestCode"). Search by plain name too.
            if not results and node:
                for e in store.search_edges_by_target_name(node.name):
                    caller = store.get_node(e.source_qualified)
                    if caller:
                        results.append(node_to_dict(caller))
                    edges_out.append(edge_to_dict(e))

        elif pattern == "callees_of":
            for e in store.get_edges_by_source(qn):
                if e.kind == "CALLS":
                    callee = store.get_node(e.target_qualified)
                    if callee:
                        results.append(node_to_dict(callee))
                    edges_out.append(edge_to_dict(e))

        elif pattern == "imports_of":
            for e in store.get_edges_by_source(qn):
                if e.kind == "IMPORTS_FROM":
                    results.append({"import_target": e.target_qualified})
                    edges_out.append(edge_to_dict(e))

        elif pattern == "importers_of":
            # Find edges where target matches this file.
            # Use resolve() to canonicalize the path, matching how
            # _resolve_module_to_file stores edge targets.
            abs_target = (
                str((root / target).resolve()) if node is None
                else node.file_path
            )
            for e in store.get_edges_by_target(abs_target):
                if e.kind == "IMPORTS_FROM":
                    results.append({
                        "importer": e.source_qualified,
                        "file": e.file_path,
                    })
                    edges_out.append(edge_to_dict(e))

        elif pattern == "children_of":
            for e in store.get_edges_by_source(qn):
                if e.kind == "CONTAINS":
                    child = store.get_node(e.target_qualified)
                    if child:
                        results.append(node_to_dict(child))

        elif pattern == "tests_for":
            transitive = store.get_transitive_tests(qn, max_depth=3)
            seen: set[str] = set()
            for t in transitive:
                qn_t = t.get("qualified_name", "")
                if qn_t not in seen:
                    seen.add(qn_t)
                    results.append(t)
            # Naming-convention fallback for tests not linked by TESTED_BY edges
            name = node.name if node else target
            for prefix in (f"test_{name}", f"Test{name}"):
                for t in store.search_nodes(prefix, limit=10):
                    if t.qualified_name not in seen and t.is_test:
                        seen.add(t.qualified_name)
                        results.append(node_to_dict(t))

        elif pattern == "inheritors_of":
            for e in store.get_edges_by_target(qn):
                if e.kind in ("INHERITS", "IMPLEMENTS"):
                    child = store.get_node(e.source_qualified)
                    if child:
                        results.append(node_to_dict(child))
                    edges_out.append(edge_to_dict(e))
            # Fallback: INHERITS/IMPLEMENTS edges store unqualified base names
            # (e.g. "Animal") while qn is fully qualified
            # (e.g. "sample.dart::Animal"). Search by plain name too. See: #87
            if not results and node:
                for kind in ("INHERITS", "IMPLEMENTS"):
                    for e in store.search_edges_by_target_name(node.name, kind=kind):
                        child = store.get_node(e.source_qualified)
                        if child:
                            results.append(node_to_dict(child))
                        edges_out.append(edge_to_dict(e))

        elif pattern == "consumers_of":
            # Find classes/methods that read this config property.
            # DEPENDS_ON_CONFIG edges use 'config:{key}' as target_qualified
            # while config nodes are stored as 'file.yml::{key}'.
            # Use the node's plain name (the config key) to bridge that gap.
            key = node.name if node else target.removeprefix("config:")
            for e in store.get_edges_by_config_key(key):
                consumer = store.get_node(e.source_qualified)
                if consumer:
                    results.append(node_to_dict(consumer))
                edges_out.append(edge_to_dict(e))
            # Prefix-wildcard: @ConfigurationProperties emits 'config:prefix.*'
            for e in store.get_edges_by_config_key(f"{key}.*"):
                consumer = store.get_node(e.source_qualified)
                if consumer:
                    results.append(node_to_dict(consumer))
                edges_out.append(edge_to_dict(e))

        elif pattern == "file_summary":
            abs_path = str(root / target)
            file_nodes = store.get_nodes_by_file(abs_path)
            for n in file_nodes:
                results.append(node_to_dict(n))

        # Apply max_results cap and record how many were omitted
        total_results = len(results)
        results_omitted = max(0, total_results - max_results)
        results = results[:max_results]
        edges_out = edges_out[:max_results]

        summary = (
            f"Found {total_results} result(s) "
            f"for {pattern}('{target}')"
            + (f" — showing {max_results}, {results_omitted} omitted" if results_omitted else "")
        )

        if detail_level == "minimal":
            minimal_results = [
                {
                    k: r[k]
                    for k in ("name", "kind", "file_path")
                    if k in r
                }
                for r in results[:5]
            ]
            results_omitted_minimal = max(0, total_results - 5)
            return {
                "status": "ok",
                "pattern": pattern,
                "target": target,
                "description": _QUERY_PATTERNS[pattern],
                "summary": summary,
                "result_count": total_results,
                "results": minimal_results,
                "results_omitted": results_omitted_minimal,
                "graph_meta": meta,
            }

        return {
            "status": "ok",
            "pattern": pattern,
            "target": target,
            "description": _QUERY_PATTERNS[pattern],
            "summary": summary,
            "results": results,
            "edges": edges_out,
            "results_omitted": results_omitted,
            "graph_meta": meta,
        }
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Tool 5: semantic_search_nodes
# ---------------------------------------------------------------------------


def semantic_search_nodes(
    query: str,
    kind: str | None = None,
    limit: int = 20,
    repo_root: str | None = None,
    context_files: list[str] | None = None,
    model: str | None = None,
    provider: str | None = None,
    detail_level: str = "standard",
) -> dict[str, Any]:
    """Search for nodes by name, keyword, or semantic similarity.

    Uses hybrid search (FTS5 BM25 + vector embeddings merged via Reciprocal
    Rank Fusion) as the primary search path, with graceful fallback to
    keyword matching.

    Args:
        query: Search string to match against node names and qualified names.
        kind: Optional filter by node kind (File, Class, Function, Type, Test).
        limit: Maximum results to return (default: 20).
        repo_root: Repository root path. Auto-detected if omitted.
        context_files: Optional list of file paths. Nodes in these files
            receive a relevance boost.
        detail_level: "standard" (full output) or "minimal" (summary only).

    Returns:
        Ranked list of matching nodes.
    """
    store, root = _get_store(repo_root)
    try:
        results = hybrid_search(
            store, query, kind=kind, limit=limit, context_files=context_files,
            model=model, provider=provider,
        )

        search_mode = "hybrid"
        if not results:
            search_mode = "keyword"

        summary = f"Found {len(results)} node(s) matching '{query}'" + (
            f" (kind={kind})" if kind else ""
        )

        meta = graph_meta(store, root)

        if detail_level == "minimal":
            minimal_results = [
                {
                    k: r[k]
                    for k in ("name", "kind", "file_path", "score")
                    if k in r
                }
                for r in results[:5]
            ]
            results_omitted = max(0, len(results) - 5)
            return {
                "status": "ok",
                "query": query,
                "search_mode": search_mode,
                "summary": summary,
                "results": minimal_results,
                "results_omitted": results_omitted,
                "graph_meta": meta,
            }

        result: dict[str, object] = {
            "status": "ok",
            "query": query,
            "search_mode": search_mode,
            "summary": summary,
            "results": results,
            "graph_meta": meta,
        }
        result["_hints"] = generate_hints(
            "semantic_search_nodes", result, get_session()
        )
        return result
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Tool 6: list_graph_stats
# ---------------------------------------------------------------------------


def list_graph_stats(
    repo_root: str | None = None,
    detail_level: str = "standard",
) -> dict[str, Any]:
    """Get aggregate statistics about the knowledge graph.

    Args:
        repo_root: Repository root path. Auto-detected if omitted.
        detail_level: "standard" returns full breakdown by kind; "minimal"
            returns only total_nodes, total_edges, files_count, and
            last_updated.

    Returns:
        Total nodes, edges, breakdown by kind, languages, and last update time.
    """
    store, root = _get_store(repo_root)
    try:
        stats = store.get_stats()

        emb_count = 0
        emb_store = EmbeddingStore(get_db_path(root))
        try:
            emb_count = emb_store.count()
        finally:
            emb_store.close()

        if detail_level == "minimal":
            summary = (
                f"{stats.total_nodes} nodes, {stats.total_edges} edges"
                f" across {stats.files_count} files"
                + (f" ({', '.join(stats.languages[:3])})" if stats.languages else "")
            )
            return {
                "status": "ok",
                "summary": summary,
                "total_nodes": stats.total_nodes,
                "total_edges": stats.total_edges,
                "files_count": stats.files_count,
                "last_updated": stats.last_updated,
            }

        summary_parts = [
            f"Graph statistics for {root.name}:",
            f"  Files: {stats.files_count}",
            f"  Total nodes: {stats.total_nodes}",
            f"  Total edges: {stats.total_edges}",
            f"  Languages: {', '.join(stats.languages) if stats.languages else 'none'}",
            f"  Last updated: {stats.last_updated or 'never'}",
            "",
            "Nodes by kind:",
        ]
        for kind, count in sorted(stats.nodes_by_kind.items()):
            summary_parts.append(f"  {kind}: {count}")
        summary_parts.append("")
        summary_parts.append("Edges by kind:")
        for kind, count in sorted(stats.edges_by_kind.items()):
            summary_parts.append(f"  {kind}: {count}")
        summary_parts.append("")
        summary_parts.append(f"Embeddings: {emb_count} nodes embedded")

        return {
            "status": "ok",
            "summary": "\n".join(summary_parts),
            "total_nodes": stats.total_nodes,
            "total_edges": stats.total_edges,
            "nodes_by_kind": stats.nodes_by_kind,
            "edges_by_kind": stats.edges_by_kind,
            "languages": stats.languages,
            "files_count": stats.files_count,
            "last_updated": stats.last_updated,
            "embeddings_count": emb_count,
        }
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Tool 9: find_large_functions
# ---------------------------------------------------------------------------


def find_large_functions(
    min_lines: int = 50,
    kind: str | None = None,
    file_path_pattern: str | None = None,
    limit: int = 50,
    repo_root: str | None = None,
    detail_level: str = "standard",
) -> dict[str, Any]:
    """Find functions, classes, or files exceeding a line-count threshold.

    Useful for identifying decomposition targets, code-quality audits,
    and enforcing size limits during code review.

    Args:
        min_lines: Minimum line count to flag (default: 50).
        kind: Filter by node kind: Function, Class, File, or Test.
        file_path_pattern: Filter by file path substring (e.g. "components/").
        limit: Maximum results (default: 50).
        repo_root: Repository root path. Auto-detected if omitted.
        detail_level: "standard" returns full node data; "minimal" returns
            only name, kind, line_count, and relative_path per result.

    Returns:
        Oversized nodes with line counts, ordered largest first.
    """
    store, root = _get_store(repo_root)
    try:
        nodes = store.get_nodes_by_size(
            min_lines=min_lines,
            kind=kind,
            file_path_pattern=file_path_pattern,
            limit=limit,
        )

        results = []
        for n in nodes:
            d = node_to_dict(n)
            d["line_count"] = (
                (n.line_end - n.line_start + 1)
                if n.line_start and n.line_end
                else 0
            )
            try:
                d["relative_path"] = str(Path(n.file_path).relative_to(root))
            except ValueError:
                d["relative_path"] = n.file_path
            results.append(d)

        summary_parts = [
            f"Found {len(results)} node(s) with >= {min_lines} lines"
            + (f" (kind={kind})" if kind else "")
            + (f" matching '{file_path_pattern}'" if file_path_pattern else "")
            + ":",
        ]
        for r in results[:10]:
            summary_parts.append(
                f"  {r['line_count']:>4} lines | {r['kind']:>8} | "
                f"{r['name']} ({r['relative_path']}:{r['line_start']})"
            )
        if len(results) > 10:
            summary_parts.append(f"  ... and {len(results) - 10} more")

        if detail_level == "minimal":
            minimal = [
                {k: r[k] for k in ("name", "kind", "line_count", "relative_path") if k in r}
                for r in results
            ]
            return {
                "status": "ok",
                "summary": "\n".join(summary_parts),
                "total_found": len(results),
                "min_lines": min_lines,
                "results": minimal,
            }

        return {
            "status": "ok",
            "summary": "\n".join(summary_parts),
            "total_found": len(results),
            "min_lines": min_lines,
            "results": results,
        }
    finally:
        store.close()


# -------------------------------------------------------------------
# traverse_graph: free-form BFS / DFS traversal
# -------------------------------------------------------------------


def traverse_graph_func(
    query: str,
    mode: str = "bfs",
    depth: int = 3,
    token_budget: int = 2000,
    repo_root: str | None = None,
) -> dict[str, Any]:
    """BFS/DFS traversal from best-matching node.

    Args:
        query: Search string to find the starting node.
        mode: "bfs" (breadth-first) or "dfs" (depth-first).
        depth: Max traversal depth (1-6). Default: 3.
        token_budget: Approximate token limit for results.
        repo_root: Repository root path.
    """
    store, root = _get_store(repo_root)
    try:
        results = hybrid_search(store, query, limit=1)
        if not results:
            return {
                "error": f"No node matching '{query}'",
                "nodes": [],
            }

        start_qn = results[0]["qualified_name"]
        depth = max(1, min(depth, 6))

        # BFS / DFS traversal
        visited: dict[str, int] = {}  # qn -> depth
        queue: list[tuple[str, int]] = [
            (start_qn, 0),
        ]
        traversal: list[dict] = []
        approx_tokens = 0

        while queue:
            if mode == "bfs":
                current_qn, cur_depth = queue.pop(0)
            else:
                current_qn, cur_depth = queue.pop()

            if current_qn in visited:
                continue
            if cur_depth > depth:
                continue

            visited[current_qn] = cur_depth
            node = store.get_node(current_qn)
            if not node:
                continue

            entry = {
                "name": _sanitize_name(node.name),
                "qualified_name": node.qualified_name,
                "kind": node.kind,
                "file": node.file_path,
                "depth": cur_depth,
            }
            approx_tokens += len(str(entry)) // 4
            if approx_tokens > token_budget:
                break

            traversal.append(entry)

            # Get neighbours
            out_edges = store.get_edges_by_source(current_qn)
            in_edges = store.get_edges_by_target(current_qn)

            # ConfigProperty nodes: DEPENDS_ON_CONFIG edges store targets as
            # 'config:{key}' but nodes are stored as 'file.yml::{key}'.
            # Bridge this gap so "who reads this property?" works in BFS.
            if node.kind == "ConfigProperty":
                in_edges = list(in_edges) + store.get_edges_by_config_key(node.name)

            # Endpoint nodes: HANDLES edges store targets as 'http:METHOD:path'
            # but nodes are stored as 'file.java::Class.METHOD path'.
            # Bridge this so BFS from an Endpoint reaches the handler method.
            if node.kind == "Endpoint":
                http_key = "http:" + node.name.replace(" ", ":", 1)
                in_edges = list(in_edges) + store.get_edges_by_endpoint_key(http_key)

            for e in out_edges:
                tgt = e.target_qualified
                # Resolve 'config:{key}' edge targets to the actual node qn
                if tgt.startswith("config:") and not tgt.endswith(".*"):
                    key = tgt[len("config:"):]
                    candidates = store.search_nodes(key, limit=3)
                    config_nodes = [c for c in candidates if c.kind == "ConfigProperty"]
                    for cn in config_nodes:
                        if cn.qualified_name not in visited:
                            queue.append((cn.qualified_name, cur_depth + 1))
                    continue
                # Resolve 'http:METHOD:path' edge targets to the actual Endpoint node qn
                if tgt.startswith("http:"):
                    parts = tgt.split(":", 2)
                    if len(parts) == 3:
                        ep_name = f"{parts[1]} {parts[2]}"
                        candidates = store.search_nodes(ep_name, limit=3)
                        ep_nodes = [c for c in candidates if c.kind == "Endpoint"]
                        for ep in ep_nodes:
                            if ep.qualified_name not in visited:
                                queue.append((ep.qualified_name, cur_depth + 1))
                    continue
                if tgt not in visited:
                    queue.append((tgt, cur_depth + 1))
            for e in in_edges:
                src = e.source_qualified
                if src not in visited:
                    queue.append((src, cur_depth + 1))

        return {
            "start_node": start_qn,
            "mode": mode,
            "max_depth": depth,
            "nodes_visited": len(traversal),
            "traversal": traversal,
            "truncated": approx_tokens > token_budget,
            "next_tool_suggestions": [
                "query_graph callers_of -- focused call relationship query",
                "query_graph consumers_of -- who reads a config property",
                "get_impact_radius -- blast radius analysis",
            ],
        }
    finally:
        store.close()
