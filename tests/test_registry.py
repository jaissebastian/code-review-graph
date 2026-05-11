"""Tests for multi-repo registry and connection pool."""

import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from code_review_graph.registry import ConnectionPool, Registry, resolve_repo


class TestRegistry:
    def setup_method(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.registry_path = Path(self.tmp_dir) / "registry.json"
        self.registry = Registry(path=self.registry_path)

        # Create fake repos
        self.repo1 = Path(self.tmp_dir) / "repo1"
        self.repo1.mkdir()
        (self.repo1 / ".git").mkdir()

        self.repo2 = Path(self.tmp_dir) / "repo2"
        self.repo2.mkdir()
        (self.repo2 / ".code-review-graph").mkdir()

    def teardown_method(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_register_and_list(self):
        """Register repos and list them back."""
        self.registry.register(str(self.repo1), alias="r1")
        self.registry.register(str(self.repo2), alias="r2")

        repos = self.registry.list_repos()
        assert len(repos) == 2
        paths = [r["path"] for r in repos]
        assert str(self.repo1.resolve()) in paths
        assert str(self.repo2.resolve()) in paths

    def test_register_duplicate_path(self):
        """Registering the same path twice updates alias."""
        self.registry.register(str(self.repo1), alias="first")
        self.registry.register(str(self.repo1), alias="second")

        repos = self.registry.list_repos()
        assert len(repos) == 1
        assert repos[0]["alias"] == "second"

    def test_register_invalid_path(self):
        """Registering a non-existent path raises ValueError."""
        import pytest
        with pytest.raises(ValueError, match="not a directory"):
            self.registry.register("/nonexistent/path/repo")

    def test_register_not_a_repo(self):
        """Registering a dir without .git or .code-review-graph raises ValueError."""
        import pytest
        bare_dir = Path(self.tmp_dir) / "bare"
        bare_dir.mkdir()
        with pytest.raises(ValueError, match="does not look like a repository"):
            self.registry.register(str(bare_dir))

    def test_unregister_by_path(self):
        """Unregister a repo by path."""
        self.registry.register(str(self.repo1), alias="r1")
        assert len(self.registry.list_repos()) == 1

        result = self.registry.unregister(str(self.repo1))
        assert result is True
        assert len(self.registry.list_repos()) == 0

    def test_unregister_by_alias(self):
        """Unregister a repo by alias."""
        self.registry.register(str(self.repo1), alias="myalias")
        assert len(self.registry.list_repos()) == 1

        result = self.registry.unregister("myalias")
        assert result is True
        assert len(self.registry.list_repos()) == 0

    def test_unregister_not_found(self):
        """Unregistering a non-registered repo returns False."""
        result = self.registry.unregister("nonexistent")
        assert result is False

    def test_find_by_alias(self):
        """find_by_alias returns correct entry."""
        self.registry.register(str(self.repo1), alias="myrepo")
        entry = self.registry.find_by_alias("myrepo")
        assert entry is not None
        assert entry["alias"] == "myrepo"
        assert entry["path"] == str(self.repo1.resolve())

    def test_find_by_alias_not_found(self):
        """find_by_alias returns None for unknown alias."""
        entry = self.registry.find_by_alias("nope")
        assert entry is None

    def test_find_by_path(self):
        """find_by_path returns correct entry."""
        self.registry.register(str(self.repo1), alias="r1")
        entry = self.registry.find_by_path(str(self.repo1))
        assert entry is not None
        assert entry["path"] == str(self.repo1.resolve())

    def test_persistence(self):
        """Registry persists to disk and reloads correctly."""
        self.registry.register(str(self.repo1), alias="persistent")

        # Create a new registry from the same file
        registry2 = Registry(path=self.registry_path)
        repos = registry2.list_repos()
        assert len(repos) == 1
        assert repos[0]["alias"] == "persistent"

    def test_resolve_by_alias(self):
        """resolve_repo resolves alias to path."""
        self.registry.register(str(self.repo1), alias="r1")
        result = resolve_repo(self.registry, "r1")
        assert result == str(self.repo1.resolve())

    def test_resolve_by_direct_path(self):
        """resolve_repo resolves direct path."""
        result = resolve_repo(self.registry, str(self.repo1))
        assert result == str(self.repo1.resolve())

    def test_resolve_by_cwd(self):
        """resolve_repo falls back to cwd when repo is None."""
        result = resolve_repo(self.registry, None, cwd=str(self.repo1))
        assert result == str(self.repo1.resolve())

    def test_resolve_returns_none(self):
        """resolve_repo returns None when nothing matches."""
        result = resolve_repo(self.registry, None)
        assert result is None


class TestConnectionPool:
    def setup_method(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.pool = ConnectionPool(max_size=3)

    def teardown_method(self):
        self.pool.close_all()
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _make_db(self, name: str) -> str:
        """Create a temporary SQLite database file."""
        db_path = str(Path(self.tmp_dir) / f"{name}.db")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE IF NOT EXISTS test (id INTEGER)")
        conn.close()
        return db_path

    def test_get_creates_connection(self):
        """get() creates a new connection."""
        db_path = self._make_db("test1")
        conn = self.pool.get(db_path)
        assert conn is not None
        assert self.pool.size == 1

    def test_get_reuses_connection(self):
        """get() returns the same connection for the same path."""
        db_path = self._make_db("test1")
        conn1 = self.pool.get(db_path)
        conn2 = self.pool.get(db_path)
        assert conn1 is conn2
        assert self.pool.size == 1

    def test_eviction_on_full(self):
        """Pool evicts LRU connection when full."""
        db1 = self._make_db("db1")
        db2 = self._make_db("db2")
        db3 = self._make_db("db3")
        db4 = self._make_db("db4")

        self.pool.get(db1)
        self.pool.get(db2)
        self.pool.get(db3)
        assert self.pool.size == 3

        # Adding 4th should evict db1 (LRU)
        self.pool.get(db4)
        assert self.pool.size == 3

    def test_close_all(self):
        """close_all() clears all connections."""
        db1 = self._make_db("db1")
        db2 = self._make_db("db2")

        self.pool.get(db1)
        self.pool.get(db2)
        assert self.pool.size == 2

        self.pool.close_all()
        assert self.pool.size == 0

    def test_lru_ordering(self):
        """Recently used connections are kept over stale ones."""
        db1 = self._make_db("db1")
        db2 = self._make_db("db2")
        db3 = self._make_db("db3")
        db4 = self._make_db("db4")

        conn1 = self.pool.get(db1)
        self.pool.get(db2)
        self.pool.get(db3)

        # Access db1 again to make it recently used
        self.pool.get(db1)

        # Now add db4 — db2 should be evicted (LRU), not db1
        self.pool.get(db4)
        assert self.pool.size == 3

        # db1 should still be in pool
        conn1_again = self.pool.get(db1)
        assert conn1_again is conn1


class TestCrossRepoSearch:
    def test_cross_repo_search_no_repos(self):
        """cross_repo_search with empty registry returns empty results."""
        from code_review_graph.tools import cross_repo_search_func

        tmp_dir = tempfile.mkdtemp()

        with patch("code_review_graph.registry.Registry") as mock_registry_cls:
            mock_instance = MagicMock()
            mock_instance.list_repos.return_value = []
            mock_registry_cls.return_value = mock_instance

            result = cross_repo_search_func(query="test")
            assert result["status"] == "ok"
            assert result["results"] == []

        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)


class TestCrossRepoCallers:
    """Tests for cross_repo_callers_func — merges caller results across repos."""

    _SCHEMA = """
        CREATE TABLE nodes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            kind TEXT NOT NULL DEFAULT 'Function',
            name TEXT NOT NULL,
            qualified_name TEXT NOT NULL,
            file_path TEXT NOT NULL DEFAULT '',
            line_start INTEGER NOT NULL DEFAULT 0,
            line_end INTEGER NOT NULL DEFAULT 0,
            language TEXT NOT NULL DEFAULT 'java',
            parent_name TEXT,
            params TEXT,
            return_type TEXT,
            is_test INTEGER NOT NULL DEFAULT 0,
            file_hash TEXT,
            extra TEXT NOT NULL DEFAULT '{}'
        );
        CREATE TABLE edges (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            kind TEXT NOT NULL,
            source_qualified TEXT NOT NULL,
            target_qualified TEXT NOT NULL,
            file_path TEXT NOT NULL DEFAULT '',
            line INTEGER NOT NULL DEFAULT 0,
            extra TEXT NOT NULL DEFAULT '{}',
            confidence REAL NOT NULL DEFAULT 1.0,
            confidence_tier TEXT NOT NULL DEFAULT 'EXTRACTED',
            updated_at REAL NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
    """

    def _make_db(
        self,
        tmp_path: Path,
        nodes: list[tuple[str, str]],
        edges: list[tuple[str, str, str]],
    ) -> str:
        """Create a minimal graph.db.

        Args:
            tmp_path: Base directory (a unique tmpdir per test).
            nodes: List of (name, qualified_name) tuples.
            edges: List of (kind, source_qualified, target_qualified) tuples.

        Returns:
            The repo root path (parent of ``.code-review-graph/``).
        """
        data_dir = tmp_path / ".code-review-graph"
        data_dir.mkdir(parents=True)
        db = data_dir / "graph.db"
        conn = sqlite3.connect(str(db))
        conn.executescript(self._SCHEMA)
        for name, qn in nodes:
            conn.execute(
                "INSERT INTO nodes (name, qualified_name) VALUES (?, ?)",
                (name, qn),
            )
        for kind, src, tgt in edges:
            conn.execute(
                "INSERT INTO edges (kind, source_qualified, target_qualified)"
                " VALUES (?, ?, ?)",
                (kind, src, tgt),
            )
        conn.commit()
        conn.close()
        return str(tmp_path)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _mock_registry(self, repo_paths: list[str]):
        """Return a patch context that makes Registry.list_repos() return *repo_paths*.

        Registry is imported lazily inside the function body, so we patch the
        class at its definition site (code_review_graph.registry.Registry).

        get_data_dir_for_repo must return None so that incremental.get_data_dir
        falls through to its default (<repo>/.code-review-graph/) resolution.
        """
        entries = [{"path": p, "alias": Path(p).name} for p in repo_paths]
        mock_instance = MagicMock()
        mock_instance.list_repos.return_value = entries
        mock_instance.get_data_dir_for_repo.return_value = None
        return patch(
            "code_review_graph.registry.Registry",
            return_value=mock_instance,
        )

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def test_empty_registry(self, tmp_path):
        """Empty registry returns ok with zero callers."""
        from code_review_graph.tools.registry_tools import cross_repo_callers_func

        with patch(
            "code_review_graph.registry.Registry"
        ) as mock_cls:
            mock_cls.return_value.list_repos.return_value = []
            mock_cls.return_value.get_data_dir_for_repo.return_value = None
            result = cross_repo_callers_func(symbol="ServiceA.processPayment")

        assert result["status"] == "ok"
        assert result["results"] == []
        assert result["total_callers"] == 0
        assert result["repos_searched"] == []

    def test_no_matching_callers(self, tmp_path):
        """Repo exists but has no CALLS edges for the symbol → empty results."""
        from code_review_graph.tools.registry_tools import cross_repo_callers_func

        repo = self._make_db(
            tmp_path / "order-service",
            nodes=[("processPayment", "ServiceA.processPayment")],
            edges=[],
        )
        with self._mock_registry([repo]):
            result = cross_repo_callers_func(symbol="ServiceA.processPayment")

        assert result["status"] == "ok"
        assert result["results"] == []
        assert result["total_callers"] == 0
        assert result["repos_with_callers"] == []
        assert len(result["repos_searched"]) == 1

    def test_single_repo_callers(self, tmp_path):
        """Single repo with a CALLS edge returns caller annotated with repo."""
        from code_review_graph.tools.registry_tools import cross_repo_callers_func

        repo = self._make_db(
            tmp_path / "payment-service",
            nodes=[
                ("processPayment", "ServiceA.processPayment"),
                ("handleRequest", "ServiceB.handleRequest"),
            ],
            edges=[
                ("CALLS", "ServiceB.handleRequest", "ServiceA.processPayment"),
            ],
        )
        with self._mock_registry([repo]):
            result = cross_repo_callers_func(symbol="ServiceA.processPayment")

        assert result["status"] == "ok"
        assert result["total_callers"] == 1
        assert len(result["results"]) == 1
        caller = result["results"][0]
        assert caller["name"] == "handleRequest"
        assert caller["repo"] == "payment-service"
        assert "repo_path" in caller
        assert "edge_confidence" in caller

    def test_multi_repo_callers(self, tmp_path):
        """Two repos each call the same symbol → merged results from both."""
        from code_review_graph.tools.registry_tools import cross_repo_callers_func

        repo_a = self._make_db(
            tmp_path / "order-service",
            nodes=[
                ("processPayment", "ServiceA.processPayment"),
                ("submitOrder", "OrderController.submitOrder"),
            ],
            edges=[("CALLS", "OrderController.submitOrder", "ServiceA.processPayment")],
        )
        repo_b = self._make_db(
            tmp_path / "billing-service",
            nodes=[
                ("processPayment", "ServiceA.processPayment"),
                ("charge", "BillingService.charge"),
            ],
            edges=[("CALLS", "BillingService.charge", "ServiceA.processPayment")],
        )
        with self._mock_registry([repo_a, repo_b]):
            result = cross_repo_callers_func(symbol="ServiceA.processPayment")

        assert result["status"] == "ok"
        assert result["total_callers"] == 2
        repos_with = result["repos_with_callers"]
        assert "order-service" in repos_with
        assert "billing-service" in repos_with
        caller_names = {r["name"] for r in result["results"]}
        assert "submitOrder" in caller_names
        assert "charge" in caller_names

    def test_detail_level_minimal(self, tmp_path):
        """detail_level='minimal' returns only compact fields per result."""
        from code_review_graph.tools.registry_tools import cross_repo_callers_func

        repo = self._make_db(
            tmp_path / "catalog-service",
            nodes=[
                ("processPayment", "ServiceA.processPayment"),
                ("fetchItem", "CatalogService.fetchItem"),
            ],
            edges=[("CALLS", "CatalogService.fetchItem", "ServiceA.processPayment")],
        )
        with self._mock_registry([repo]):
            result = cross_repo_callers_func(
                symbol="ServiceA.processPayment", detail_level="minimal"
            )

        assert result["status"] == "ok"
        assert result["total_callers"] == 1
        caller = result["results"][0]
        allowed = {"name", "kind", "file_path", "repo", "edge_confidence"}
        assert set(caller.keys()) <= allowed
        assert "qualified_name" not in caller
        assert "repo_path" not in caller

    def test_limit_truncates_results(self, tmp_path):
        """limit parameter caps returned results and reports results_omitted."""
        from code_review_graph.tools.registry_tools import cross_repo_callers_func

        callers = [
            (f"method{i}", f"ServiceX.method{i}") for i in range(5)
        ]
        edges = [
            ("CALLS", f"ServiceX.method{i}", "ServiceA.processPayment")
            for i in range(5)
        ]
        repo = self._make_db(
            tmp_path / "large-service",
            nodes=[("processPayment", "ServiceA.processPayment")] + callers,
            edges=edges,
        )
        with self._mock_registry([repo]):
            result = cross_repo_callers_func(
                symbol="ServiceA.processPayment", limit=3
            )

        assert result["status"] == "ok"
        assert len(result["results"]) == 3
        assert result["total_callers"] == 5
        assert result["results_omitted"] == 2

    def test_failed_repo_skipped(self, tmp_path):
        """If one repo raises an error, remaining repos still return results."""
        from code_review_graph.tools.registry_tools import cross_repo_callers_func

        good_repo = self._make_db(
            tmp_path / "good-service",
            nodes=[
                ("processPayment", "ServiceA.processPayment"),
                ("handleRequest", "GoodService.handleRequest"),
            ],
            edges=[("CALLS", "GoodService.handleRequest", "ServiceA.processPayment")],
        )
        bad_repo_path = str(tmp_path / "bad-service")

        entries = [
            {"path": bad_repo_path, "alias": "bad-service"},
            {"path": good_repo, "alias": "good-service"},
        ]
        with patch(
            "code_review_graph.registry.Registry"
        ) as mock_cls:
            mock_cls.return_value.list_repos.return_value = entries
            mock_cls.return_value.get_data_dir_for_repo.return_value = None
            result = cross_repo_callers_func(symbol="ServiceA.processPayment")

        assert result["status"] == "ok"
        assert result["total_callers"] == 1
        assert result["results"][0]["repo"] == "good-service"


class TestCrossRepoCallersTemporalStub:
    """TEMPORAL_STUB edges are included in cross_repo_callers results."""

    _SCHEMA = TestCrossRepoCallers._SCHEMA  # reuse minimal schema

    def _make_db(self, tmp_path: Path, nodes, edges) -> str:
        return TestCrossRepoCallers._make_db(self, tmp_path, nodes, edges)

    def _mock_registry(self, repo_paths):
        return TestCrossRepoCallers._mock_registry(self, repo_paths)

    def test_temporal_stub_callers_found(self, tmp_path):
        """TEMPORAL_STUB edges from a workflow to an activity appear as callers."""
        from code_review_graph.tools.registry_tools import cross_repo_callers_func

        repo = self._make_db(
            tmp_path / "workflow-service",
            nodes=[
                ("processOrder", "OrderActivity.processOrder"),
                ("executeWorkflow", "OrderWorkflowImpl.executeWorkflow"),
            ],
            edges=[
                ("TEMPORAL_STUB", "OrderWorkflowImpl.executeWorkflow", "OrderActivity"),
            ],
        )
        with self._mock_registry([repo]):
            result = cross_repo_callers_func(symbol="OrderActivity")

        assert result["status"] == "ok"
        assert result["total_callers"] == 1
        caller = result["results"][0]
        assert caller["name"] == "executeWorkflow"
        assert caller["edge_kind"] == "TEMPORAL_STUB"
        assert caller.get("relationship") == "temporal"

    def test_temporal_stub_cross_repo(self, tmp_path):
        """TEMPORAL_STUB in repo-A resolves callers for activity defined in repo-B."""
        from code_review_graph.tools.registry_tools import cross_repo_callers_func

        # repo-A: has the workflow that stubs ShipmentActivity
        repo_a = self._make_db(
            tmp_path / "order-service",
            nodes=[
                ("processOrder", "OrderWorkflowImpl.processOrder"),
            ],
            edges=[
                ("TEMPORAL_STUB", "OrderWorkflowImpl.processOrder", "ShipmentActivity"),
            ],
        )
        # repo-B: defines ShipmentActivity interface (node exists here)
        repo_b = self._make_db(
            tmp_path / "shipment-service",
            nodes=[
                ("ShipmentActivity", "ShipmentActivity"),
                ("dispatchShipment", "ShipmentActivityImpl.dispatchShipment"),
            ],
            edges=[],
        )
        with self._mock_registry([repo_a, repo_b]):
            result = cross_repo_callers_func(symbol="ShipmentActivity")

        assert result["status"] == "ok"
        # order-service provides 1 TEMPORAL_STUB caller
        assert result["total_callers"] >= 1
        kinds = {r["edge_kind"] for r in result["results"]}
        assert "TEMPORAL_STUB" in kinds
        repos = {r["repo"] for r in result["results"]}
        assert "order-service" in repos


class TestCrossRepoKafkaImpact:
    """Tests for cross_repo_kafka_impact_func — Kafka topic/message-type queries."""

    _SCHEMA = TestCrossRepoCallers._SCHEMA

    def _make_db_with_kafka(
        self,
        tmp_path: Path,
        kafka_edges: list[tuple[str, str, str, str]],
    ) -> str:
        """Create a minimal graph.db with Kafka PRODUCES/CONSUMES edges.

        Args:
            kafka_edges: List of (kind, source_qualified, target_qualified, extra_json).
        """
        data_dir = tmp_path / ".code-review-graph"
        data_dir.mkdir(parents=True)
        db = data_dir / "graph.db"
        conn = sqlite3.connect(str(db))
        conn.executescript(TestCrossRepoCallers._SCHEMA)
        # Insert node for each source
        sources = {e[1] for e in kafka_edges}
        for src in sources:
            name = src.split(".")[-1]
            conn.execute(
                "INSERT INTO nodes (name, qualified_name) VALUES (?, ?)",
                (name, src),
            )
        for kind, src, tgt, extra in kafka_edges:
            conn.execute(
                "INSERT INTO edges (kind, source_qualified, target_qualified, extra)"
                " VALUES (?, ?, ?, ?)",
                (kind, src, tgt, extra),
            )
        conn.commit()
        conn.close()
        return str(tmp_path)

    def _mock_registry(self, repo_paths):
        return TestCrossRepoCallers._mock_registry(self, repo_paths)

    def test_empty_registry(self, tmp_path):
        """Empty registry returns ok with empty producers/consumers."""
        from code_review_graph.tools.registry_tools import cross_repo_kafka_impact_func

        with patch("code_review_graph.registry.Registry") as mock_cls:
            mock_cls.return_value.list_repos.return_value = []
            mock_cls.return_value.get_data_dir_for_repo.return_value = None
            result = cross_repo_kafka_impact_func("OrderEvent")

        assert result["status"] == "ok"
        assert result["producers"] == []
        assert result["consumers"] == []

    def test_match_by_message_type(self, tmp_path):
        """Matches PRODUCES/CONSUMES edges by message_type in extra JSON."""
        from code_review_graph.tools.registry_tools import cross_repo_kafka_impact_func

        import json
        producer_repo = self._make_db_with_kafka(
            tmp_path / "producer-service",
            kafka_edges=[
                (
                    "PRODUCES",
                    "OrderService.publishOrder",
                    "kafka:config",
                    json.dumps({"message_type": "OrderEvent", "kafka_type": "KafkaTemplate"}),
                )
            ],
        )
        consumer_repo = self._make_db_with_kafka(
            tmp_path / "notification-service",
            kafka_edges=[
                (
                    "CONSUMES",
                    "NotificationListener.onOrder",
                    "kafka:config",
                    json.dumps({"message_type": "OrderEvent", "kafka_type": "KafkaListener"}),
                )
            ],
        )
        with self._mock_registry([producer_repo, consumer_repo]):
            result = cross_repo_kafka_impact_func("OrderEvent")

        assert result["status"] == "ok"
        assert len(result["producers"]) == 1
        assert len(result["consumers"]) == 1
        assert result["producers"][0]["repo"] == "producer-service"
        assert result["consumers"][0]["repo"] == "notification-service"

    def test_match_by_topic_name(self, tmp_path):
        """Matches edges by topic name in target_qualified."""
        from code_review_graph.tools.registry_tools import cross_repo_kafka_impact_func

        repo = self._make_db_with_kafka(
            tmp_path / "order-service",
            kafka_edges=[
                ("CONSUMES", "OrderConsumer.handle", "kafka:order.created", "{}"),
                ("PRODUCES", "OrderProducer.send", "kafka:order.created", "{}"),
            ],
        )
        with self._mock_registry([repo]):
            result = cross_repo_kafka_impact_func("order.created")

        assert result["status"] == "ok"
        assert len(result["producers"]) == 1
        assert len(result["consumers"]) == 1

    def test_no_match(self, tmp_path):
        """Query with no matching topic or message type returns empty lists."""
        from code_review_graph.tools.registry_tools import cross_repo_kafka_impact_func

        import json
        repo = self._make_db_with_kafka(
            tmp_path / "other-service",
            kafka_edges=[
                (
                    "CONSUMES",
                    "SomeConsumer.handle",
                    "kafka:config",
                    json.dumps({"message_type": "ShipmentEvent"}),
                )
            ],
        )
        with self._mock_registry([repo]):
            result = cross_repo_kafka_impact_func("OrderEvent")

        assert result["status"] == "ok"
        assert result["producers"] == []
        assert result["consumers"] == []


class TestSetDataDir:
    """Tests for set_data_dir and get_data_dir_for_repo methods."""

    def setup_method(self):
        """Set up isolated test registry."""
        self.tmp_dir = tempfile.mkdtemp()
        self.registry_path = Path(self.tmp_dir) / "registry.json"
        self.registry = Registry(path=self.registry_path)

    def teardown_method(self):
        """Clean up temporary directory."""
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_set_data_dir_new_repo(self):
        """set_data_dir should create new registry entry if repo not registered."""
        repo = Path(self.tmp_dir) / "project"
        repo.mkdir()
        data_dir = Path(self.tmp_dir) / "data"

        entry = self.registry.set_data_dir(str(repo), str(data_dir))

        assert entry["path"] == str(repo.resolve())
        assert entry["data_dir"] == str(data_dir.resolve())

        # Verify it can be retrieved
        retrieved = self.registry.get_data_dir_for_repo(str(repo))
        assert retrieved == str(data_dir.resolve())

        # Verify entry is in list
        repos = self.registry.list_repos()
        assert len(repos) == 1
        assert repos[0]["path"] == str(repo.resolve())

    def test_set_data_dir_existing_repo(self):
        """set_data_dir should update data_dir for already registered repo."""
        repo = Path(self.tmp_dir) / "project"
        repo.mkdir()
        data_dir1 = Path(self.tmp_dir) / "data1"
        data_dir2 = Path(self.tmp_dir) / "data2"

        # Initial registration
        entry1 = self.registry.set_data_dir(str(repo), str(data_dir1))
        assert entry1["data_dir"] == str(data_dir1.resolve())

        # Update with new data_dir
        entry2 = self.registry.set_data_dir(str(repo), str(data_dir2))
        assert entry2["data_dir"] == str(data_dir2.resolve())

        # Verify only one entry exists
        repos = self.registry.list_repos()
        assert len(repos) == 1

    def test_get_data_dir_for_repo_unknown(self):
        """get_data_dir_for_repo should return None for unknown repo."""
        unknown_repo = Path(self.tmp_dir) / "unknown"

        result = self.registry.get_data_dir_for_repo(str(unknown_repo))
        assert result is None

    def test_set_data_dir_with_alias(self):
        """register() with data_dir should store both."""
        repo = Path(self.tmp_dir) / "project"
        repo.mkdir()
        (repo / ".git").mkdir()
        data_dir = Path(self.tmp_dir) / "data"
        alias = "my-project"

        entry = self.registry.register(str(repo), alias=alias, data_dir=str(data_dir))

        assert entry["path"] == str(repo.resolve())
        assert entry["alias"] == alias
        assert entry["data_dir"] == str(data_dir.resolve())

    def test_backward_compatibility(self):
        """Old registry entries without data_dir should work."""
        repo = Path(self.tmp_dir) / "project"
        repo.mkdir()

        # Create entry without data_dir (old format)
        self.registry._repos.append({
            "path": str(repo.resolve()),
            "alias": "old-project"
        })
        self.registry._save()

        # Should not crash
        result = self.registry.get_data_dir_for_repo(str(repo))
        assert result is None

        # Should be able to add data_dir
        data_dir = Path(self.tmp_dir) / "data"
        entry = self.registry.set_data_dir(str(repo), str(data_dir))
        assert entry["data_dir"] == str(data_dir.resolve())
