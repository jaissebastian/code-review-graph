"""Tests for agent-transparency improvements in tool responses:

  graph_meta      — confidence + staleness surfaced in every tool response
  omitted counts  — results_omitted / nodes_omitted when results are capped
  disambiguation  — ranked FQN list with file:line on multi-match symbols
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from code_review_graph.graph import GraphStore
from code_review_graph.parser import EdgeInfo, NodeInfo
from code_review_graph.tools._common import graph_meta
from code_review_graph.tools.query import (
    _rank_disambiguation_candidates,
    query_graph,
    semantic_search_nodes,
)


# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------

def _make_repo(tmp_path: Path) -> tuple[GraphStore, Path]:
    """Create a minimal repo dir with .git and a graph db."""
    root = tmp_path / "repo"
    root.mkdir()
    (root / ".git").mkdir()
    (root / ".code-review-graph").mkdir()
    db = GraphStore(str(root / ".code-review-graph" / "graph.db"))
    return db, root


def _seed_basic(store: GraphStore, root: Path) -> None:
    store.upsert_node(NodeInfo(
        kind="File", name="auth.py", file_path=str(root / "auth.py"),
        line_start=1, line_end=50, language="python",
    ))
    store.upsert_node(NodeInfo(
        kind="Function", name="login", file_path=str(root / "auth.py"),
        line_start=10, line_end=20, language="python",
        parent_name="AuthService",
    ))
    store.upsert_node(NodeInfo(
        kind="Function", name="process", file_path=str(root / "main.py"),
        line_start=5, line_end=15, language="python",
    ))
    # Edge target matches the qualified name: file_path::ParentClass.method
    store.upsert_edge(EdgeInfo(
        kind="CALLS",
        source=str(root / "main.py") + "::process",
        target=str(root / "auth.py") + "::AuthService.login",
        file_path=str(root / "main.py"),
        line=10,
    ))
    store.commit()


# ---------------------------------------------------------------------------
# graph_meta helper
# ---------------------------------------------------------------------------

class TestGraphMeta:
    def test_returns_indexed_at(self, tmp_path):
        store, root = _make_repo(tmp_path)
        try:
            store.set_metadata("last_updated", "2024-01-15T12:00:00")
            meta = graph_meta(store, root)
            assert meta["indexed_at"] == "2024-01-15T12:00:00"
        finally:
            store.close()

    def test_unknown_when_no_last_updated(self, tmp_path):
        store, root = _make_repo(tmp_path)
        try:
            meta = graph_meta(store, root)
            assert meta["indexed_at"] == "unknown"
        finally:
            store.close()

    def test_is_stale_false_when_same_commit(self, tmp_path):
        store, root = _make_repo(tmp_path)
        try:
            sha = "abc1234def5678ab1234def5678ab1234def5678"
            store.set_metadata("git_head_sha", sha)

            with patch(
                "code_review_graph.tools._common.subprocess.run"
            ) as mock_run:
                mock_run.return_value.returncode = 0
                mock_run.return_value.stdout = sha + "\n"
                meta = graph_meta(store, root)

            assert meta["is_stale"] is False
            assert meta["indexed_commit"] == sha[:8]
            assert meta["head_commit"] == sha[:8]
        finally:
            store.close()

    def test_is_stale_true_when_different_commit(self, tmp_path):
        store, root = _make_repo(tmp_path)
        try:
            indexed_sha = "aaaa1234" + "a" * 32
            head_sha = "bbbb5678" + "b" * 32
            store.set_metadata("git_head_sha", indexed_sha)

            with patch(
                "code_review_graph.tools._common.subprocess.run"
            ) as mock_run:
                mock_run.return_value.returncode = 0
                mock_run.return_value.stdout = head_sha + "\n"
                meta = graph_meta(store, root)

            assert meta["is_stale"] is True
        finally:
            store.close()

    def test_no_is_stale_when_no_indexed_commit(self, tmp_path):
        store, root = _make_repo(tmp_path)
        try:
            with patch(
                "code_review_graph.tools._common.subprocess.run"
            ) as mock_run:
                mock_run.return_value.returncode = 0
                mock_run.return_value.stdout = "deadbeef12345678\n"
                meta = graph_meta(store, root)

            assert "is_stale" not in meta
            assert "indexed_commit" not in meta
        finally:
            store.close()

    def test_git_failure_gives_no_head_commit(self, tmp_path):
        store, root = _make_repo(tmp_path)
        try:
            with patch(
                "code_review_graph.tools._common.subprocess.run"
            ) as mock_run:
                mock_run.return_value.returncode = 128
                mock_run.return_value.stdout = ""
                meta = graph_meta(store, root)

            assert "head_commit" not in meta
        finally:
            store.close()


# ---------------------------------------------------------------------------
# graph_meta in query_graph responses
# ---------------------------------------------------------------------------

class TestQueryGraphMeta:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp()
        self.root = Path(self.tmp) / "repo"
        self.root.mkdir()
        (self.root / ".git").mkdir()
        (self.root / ".code-review-graph").mkdir()
        self.db = GraphStore(str(self.root / ".code-review-graph" / "graph.db"))
        _seed_basic(self.db, self.root)
        self.db.set_metadata("last_updated", "2024-06-01T00:00:00")
        self.db.close()

    def teardown_method(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_graph_meta_in_ok_response(self):
        # QN includes parent class: file_path::AuthService.login
        target = str(self.root / "auth.py") + "::AuthService.login"
        result = query_graph(
            pattern="callers_of",
            target=target,
            repo_root=str(self.root),
        )
        assert result["status"] == "ok"
        assert "graph_meta" in result
        assert result["graph_meta"]["indexed_at"] == "2024-06-01T00:00:00"

    def test_graph_meta_in_not_found_response(self):
        result = query_graph(
            pattern="callers_of",
            target="nonexistent_symbol_xyz",
            repo_root=str(self.root),
        )
        assert result["status"] == "not_found"
        assert "graph_meta" in result

    def test_graph_meta_in_minimal_mode(self):
        result = query_graph(
            pattern="callers_of",
            target=str(self.root / "auth.py") + "::login",
            repo_root=str(self.root),
            detail_level="minimal",
        )
        assert "graph_meta" in result

    def test_confidence_in_edge_results(self):
        # QN includes parent class: file_path::AuthService.login
        target = str(self.root / "auth.py") + "::AuthService.login"
        result = query_graph(
            pattern="callers_of",
            target=target,
            repo_root=str(self.root),
        )
        assert result["status"] == "ok"
        if result["edges"]:
            edge = result["edges"][0]
            assert "confidence" in edge
            assert isinstance(edge["confidence"], float)
            assert "confidence_tier" in edge


# ---------------------------------------------------------------------------
# graph_meta in semantic_search_nodes responses
# ---------------------------------------------------------------------------

class TestSemanticSearchMeta:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp()
        self.root = Path(self.tmp) / "repo"
        self.root.mkdir()
        (self.root / ".git").mkdir()
        (self.root / ".code-review-graph").mkdir()
        self.db = GraphStore(str(self.root / ".code-review-graph" / "graph.db"))
        _seed_basic(self.db, self.root)
        self.db.set_metadata("last_updated", "2024-06-01T00:00:00")
        self.db.close()

    def teardown_method(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_graph_meta_in_standard_response(self):
        result = semantic_search_nodes(
            query="login", repo_root=str(self.root),
        )
        assert "graph_meta" in result
        assert result["graph_meta"]["indexed_at"] == "2024-06-01T00:00:00"

    def test_graph_meta_in_minimal_response(self):
        result = semantic_search_nodes(
            query="login", repo_root=str(self.root), detail_level="minimal",
        )
        assert "graph_meta" in result


# ---------------------------------------------------------------------------
# Truncation counts (results_omitted / nodes_omitted)
# ---------------------------------------------------------------------------

class TestTruncationCounts:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp()
        self.root = Path(self.tmp) / "repo"
        self.root.mkdir()
        (self.root / ".git").mkdir()
        (self.root / ".code-review-graph").mkdir()
        self.db = GraphStore(str(self.root / ".code-review-graph" / "graph.db"))
        self._seed_many_callers()
        self.db.close()

    def _seed_many_callers(self) -> None:
        target_qn = str(self.root / "lib.py") + "::target_fn"
        self.db.upsert_node(NodeInfo(
            kind="Function", name="target_fn", file_path=str(self.root / "lib.py"),
            line_start=1, line_end=5, language="python",
        ))
        for i in range(15):
            caller_qn = str(self.root / f"caller_{i}.py") + f"::caller_{i}"
            self.db.upsert_node(NodeInfo(
                kind="Function", name=f"caller_{i}",
                file_path=str(self.root / f"caller_{i}.py"),
                line_start=1, line_end=3, language="python",
            ))
            self.db.upsert_edge(EdgeInfo(
                kind="CALLS", source=caller_qn, target=target_qn,
                file_path=str(self.root / f"caller_{i}.py"), line=2,
            ))
        self.db.commit()

    def teardown_method(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_results_omitted_zero_when_under_cap(self):
        target_qn = str(self.root / "lib.py") + "::target_fn"
        result = query_graph(
            pattern="callers_of", target=target_qn,
            repo_root=str(self.root), max_results=100,
        )
        assert result["status"] == "ok"
        assert result["results_omitted"] == 0

    def test_results_omitted_when_cap_exceeded(self):
        target_qn = str(self.root / "lib.py") + "::target_fn"
        result = query_graph(
            pattern="callers_of", target=target_qn,
            repo_root=str(self.root), max_results=5,
        )
        assert result["status"] == "ok"
        assert result["results_omitted"] == 10  # 15 total - 5 cap
        assert len(result["results"]) == 5

    def test_results_omitted_in_minimal_mode(self):
        target_qn = str(self.root / "lib.py") + "::target_fn"
        result = query_graph(
            pattern="callers_of", target=target_qn,
            repo_root=str(self.root), detail_level="minimal",
        )
        assert result["status"] == "ok"
        # minimal shows only 5, total is 15 → 10 omitted
        assert result["results_omitted"] == 10
        assert len(result["results"]) == 5

    def test_result_count_reflects_total(self):
        target_qn = str(self.root / "lib.py") + "::target_fn"
        result = query_graph(
            pattern="callers_of", target=target_qn,
            repo_root=str(self.root), detail_level="minimal",
        )
        assert result["result_count"] == 15

    def test_summary_mentions_omitted_when_capped(self):
        target_qn = str(self.root / "lib.py") + "::target_fn"
        result = query_graph(
            pattern="callers_of", target=target_qn,
            repo_root=str(self.root), max_results=3,
        )
        assert "omitted" in result["summary"]

    def test_semantic_search_results_omitted_in_minimal(self):
        # Seed 10 functions with similar names
        db = GraphStore(str(self.root / ".code-review-graph" / "graph.db"))
        for i in range(10):
            db.upsert_node(NodeInfo(
                kind="Function", name=f"do_thing_{i}",
                file_path=str(self.root / f"m{i}.py"),
                line_start=1, line_end=3, language="python",
            ))
        db.commit()
        db.close()

        result = semantic_search_nodes(
            query="do_thing", repo_root=str(self.root),
            limit=10, detail_level="minimal",
        )
        # minimal caps at 5; results_omitted should reflect the difference
        assert "results_omitted" in result
        assert isinstance(result["results_omitted"], int)
        assert result["results_omitted"] >= 0


# ---------------------------------------------------------------------------
# Symbol disambiguation
# ---------------------------------------------------------------------------

class TestDisambiguationRanking:
    """Unit tests for _rank_disambiguation_candidates."""

    def _make_node(self, qn: str, name: str, kind: str = "Function",
                   file_path: str = "/repo/a.py", line_start: int = 1) -> object:
        from code_review_graph.graph import GraphNode
        return GraphNode(
            id=1, kind=kind, name=name, qualified_name=qn,
            file_path=file_path, line_start=line_start, line_end=10,
            language="python", parent_name=None, params=None,
            return_type=None, is_test=False, file_hash=None, extra={},
        )

    def test_exact_qn_ranked_first(self):
        nodes = [
            self._make_node("pkg.foo::login", "login", file_path="/a/foo.py"),
            self._make_node("pkg.bar::login", "login", file_path="/a/bar.py"),
            self._make_node("pkg.foo::login", "login", file_path="/a/exact.py"),
        ]
        # Use exact match as target
        nodes[2] = self._make_node("exact_target", "login", file_path="/a/exact.py")
        result = _rank_disambiguation_candidates(nodes, "exact_target")
        assert result[0]["qualified_name"] == "exact_target"

    def test_exact_name_ranked_before_partial(self):
        nodes = [
            self._make_node("partial_match::login_helper", "login_helper"),
            self._make_node("exact_match::login", "login"),
        ]
        result = _rank_disambiguation_candidates(nodes, "login")
        assert result[0]["name"] == "login"

    def test_enriched_with_file_and_line(self):
        nodes = [
            self._make_node("pkg::foo", "foo", file_path="/repo/foo.py", line_start=42),
        ]
        result = _rank_disambiguation_candidates(nodes, "foo")
        assert result[0]["file_path"] == "/repo/foo.py"
        assert result[0]["line_start"] == 42
        assert result[0]["qualified_name"] == "pkg::foo"
        assert result[0]["kind"] == "Function"

    def test_names_are_sanitized(self):
        nodes = [
            self._make_node("pkg::bad\x00name", "bad\x00name"),
        ]
        result = _rank_disambiguation_candidates(nodes, "badname")
        assert "\x00" not in result[0]["qualified_name"]
        assert "\x00" not in result[0]["name"]


class TestQueryGraphDisambiguation:
    """Integration tests for ambiguous query_graph responses."""

    def setup_method(self):
        self.tmp = tempfile.mkdtemp()
        self.root = Path(self.tmp) / "repo"
        self.root.mkdir()
        (self.root / ".git").mkdir()
        (self.root / ".code-review-graph").mkdir()
        self.db = GraphStore(str(self.root / ".code-review-graph" / "graph.db"))
        # Two functions with the same bare name in different files
        self.db.upsert_node(NodeInfo(
            kind="Function", name="process", file_path=str(self.root / "a.py"),
            line_start=1, line_end=10, language="python",
        ))
        self.db.upsert_node(NodeInfo(
            kind="Function", name="process", file_path=str(self.root / "b.py"),
            line_start=20, line_end=30, language="python",
        ))
        self.db.commit()
        self.db.close()

    def teardown_method(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_ambiguous_status_on_multi_match(self):
        result = query_graph(
            pattern="callers_of", target="process",
            repo_root=str(self.root),
        )
        assert result["status"] == "ambiguous"

    def test_disambiguation_key_present(self):
        result = query_graph(
            pattern="callers_of", target="process",
            repo_root=str(self.root),
        )
        assert "disambiguation" in result
        assert isinstance(result["disambiguation"], list)
        assert len(result["disambiguation"]) >= 2

    def test_disambiguation_includes_required_fields(self):
        result = query_graph(
            pattern="callers_of", target="process",
            repo_root=str(self.root),
        )
        for entry in result["disambiguation"]:
            assert "qualified_name" in entry
            assert "name" in entry
            assert "kind" in entry
            assert "file_path" in entry
            assert "line_start" in entry

    def test_hint_field_present(self):
        result = query_graph(
            pattern="callers_of", target="process",
            repo_root=str(self.root),
        )
        assert "hint" in result
        assert "qualified_name" in result["hint"]

    def test_graph_meta_in_ambiguous_response(self):
        result = query_graph(
            pattern="callers_of", target="process",
            repo_root=str(self.root),
        )
        assert "graph_meta" in result

    def test_summary_mentions_count(self):
        result = query_graph(
            pattern="callers_of", target="process",
            repo_root=str(self.root),
        )
        assert "2" in result["summary"] or "multiple" in result["summary"].lower()

    def test_no_more_candidates_key(self):
        """Old 'candidates' key should be gone; replaced by 'disambiguation'."""
        result = query_graph(
            pattern="callers_of", target="process",
            repo_root=str(self.root),
        )
        assert "candidates" not in result


# ---------------------------------------------------------------------------
# Java FQN resolution (package.Class.method → graph file-path QN)
# ---------------------------------------------------------------------------

class TestJavaFQNResolution:
    """query_graph must resolve Java package-qualified names, not just graph QNs.

    Agents and IDEs write targets like ``com.example.service.OrderHandler.process``.
    The graph stores nodes as ``/path/to/OrderHandler.java::OrderHandler.process``.
    Without FQN decomposition the lookup silently returns not_found even though
    the node exists.
    """

    def setup_method(self):
        self.tmp = tempfile.mkdtemp()
        self.root = Path(self.tmp) / "repo"
        self.root.mkdir()
        (self.root / ".git").mkdir()
        (self.root / ".code-review-graph").mkdir()
        self.db = GraphStore(str(self.root / ".code-review-graph" / "graph.db"))

        handler_file = str(self.root / "OrderHandler.java")
        caller_file = str(self.root / "OrderRouter.java")

        self.db.upsert_node(NodeInfo(
            kind="Class", name="OrderHandler",
            file_path=handler_file,
            line_start=1, line_end=50, language="java",
        ))
        self.db.upsert_node(NodeInfo(
            kind="Function", name="process",
            file_path=handler_file,
            line_start=10, line_end=30, language="java",
            parent_name="OrderHandler",
        ))
        self.db.upsert_node(NodeInfo(
            kind="Function", name="routes",
            file_path=caller_file,
            line_start=5, line_end=15, language="java",
        ))
        # routes() calls process()
        target_qn = handler_file + "::OrderHandler.process"
        self.db.upsert_edge(EdgeInfo(
            kind="CALLS",
            source=caller_file + "::routes",
            target=target_qn,
            file_path=caller_file, line=10,
        ))
        self.db.commit()
        self.db.close()

    def teardown_method(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_java_fqn_class_dot_method_resolves(self):
        """ClassName.method short form resolves to the graph node."""
        result = query_graph(
            pattern="callers_of",
            target="OrderHandler.process",
            repo_root=str(self.root),
        )
        assert result["status"] == "ok", (
            f"Expected ok, got {result['status']}: {result.get('summary')}"
        )
        names = [r["name"] for r in result["results"]]
        assert "routes" in names

    def test_java_fqn_package_qualified_resolves(self):
        """Full package.Class.method form resolves via FQN decomposition."""
        result = query_graph(
            pattern="callers_of",
            target="com.example.service.OrderHandler.process",
            repo_root=str(self.root),
        )
        assert result["status"] == "ok", (
            f"Expected ok, got {result['status']}: {result.get('summary')}"
        )
        names = [r["name"] for r in result["results"]]
        assert "routes" in names

    def test_bare_method_name_with_unique_match_resolves(self):
        """Plain method name resolves when only one node with that exact name exists.

        The setup includes an OrderHandler class node whose FTS tokens overlap with
        'process', so searching the bare method name can be ambiguous. A
        uniquely-named method with no class name collision resolves cleanly.
        """
        db = GraphStore(str(self.root / ".code-review-graph" / "graph.db"))
        unique_file = str(self.root / "Fulfillment.java")
        db.upsert_node(NodeInfo(
            kind="Function", name="fulfillOrder",
            file_path=unique_file,
            line_start=1, line_end=5, language="java",
        ))
        caller_file = str(self.root / "FulfillmentRouter.java")
        db.upsert_node(NodeInfo(
            kind="Function", name="routeFulfillment",
            file_path=caller_file,
            line_start=1, line_end=5, language="java",
        ))
        db.upsert_edge(EdgeInfo(
            kind="CALLS",
            source=caller_file + "::routeFulfillment",
            target=unique_file + "::fulfillOrder",
            file_path=caller_file, line=3,
        ))
        db.commit()
        db.close()

        result = query_graph(
            pattern="callers_of",
            target="fulfillOrder",
            repo_root=str(self.root),
        )
        assert result["status"] == "ok"
        names = [r["name"] for r in result["results"]]
        assert "routeFulfillment" in names

    def test_ambiguous_fqn_returns_disambiguation(self):
        """FQN decomposition returns disambiguation when ClassName.method is in two files."""
        db = GraphStore(str(self.root / ".code-review-graph" / "graph.db"))
        # Same class name in a second module — same ClassName.method decomposition
        file2 = str(self.root / "v2" / "OrderHandler.java")
        Path(file2).parent.mkdir(exist_ok=True)
        db.upsert_node(NodeInfo(
            kind="Function", name="process",
            file_path=file2,
            line_start=1, line_end=5, language="java",
            parent_name="OrderHandler",
        ))
        db.commit()
        db.close()

        # "OrderHandler.process" (width=2) now matches two nodes in different files
        result = query_graph(
            pattern="callers_of",
            target="com.example.v2.OrderHandler.process",
            repo_root=str(self.root),
        )
        assert result["status"] == "ambiguous"
        assert "disambiguation" in result
        assert len(result["disambiguation"]) >= 2
