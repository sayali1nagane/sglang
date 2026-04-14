from types import SimpleNamespace

import torch

from sglang.srt.managers.schedule_batch import FINISH_ABORT
from sglang.srt.mem_cache.base_prefix_cache import MatchResult
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.mem_cache.session_aware_cache import SessionAwareCache, SessionSlot
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="stage-a-test-cpu")


class _FakeAllocator:
    def __init__(self):
        self.freed = []

    def free(self, free_index: torch.Tensor):
        self.freed.append(free_index.clone())


class _FakeInnerCache:
    def __init__(self, req_to_token_pool, allocator, page_size, match_results=None):
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = allocator
        self.page_size = page_size
        self.match_results = list(match_results or [])
        self.dec_lock_ref_calls = []

    def cache_finished_req(self, *args, **kwargs):
        raise AssertionError("Streaming requests should not delegate to inner cache")

    def match_prefix(self, *args, **kwargs):
        if not self.match_results:
            raise AssertionError("Unexpected match_prefix call")
        return self.match_results.pop(0)

    def dec_lock_ref(self, node, *args, **kwargs):
        self.dec_lock_ref_calls.append(node)

    def supports_mamba(self):
        return False

    def sanity_check(self):
        return None


class _FakeReq:
    def __init__(
        self, session_id: str, req_pool_idx: int, committed: int, allocated: int
    ):
        self.session = SimpleNamespace(
            session_id=session_id,
            streaming=True,
            finish_req=lambda req: None,
            abort_req=lambda: None,
            _inflight=False,
        )
        self.req_pool_idx = req_pool_idx
        self.kv_committed_len = committed
        self.kv_allocated_len = allocated
        self.kv_committed_freed = False
        self.kv_overallocated_freed = False
        self.origin_input_ids = list(range(committed))
        self.output_ids = []
        self.extra_key = None
        self.swa_evicted_seqlen = 0
        self.last_node = None
        self.cache_protected_len = 0
        self.swa_uuid_for_lock = None
        self.mamba_pool_idx = None
        self.mamba_ping_pong_track_buffer = None
        self.mamba_next_track_idx = None
        self.mamba_last_track_seqlen = None
        self.mamba_branching_seqlen = None
        self.pop_overallocated_calls = 0
        self.to_finish = None
        self.finished_reason = None

    def pop_committed_kv_cache(self):
        assert not self.kv_committed_freed
        self.kv_committed_freed = True
        return self.kv_committed_len

    def pop_overallocated_kv_cache(self):
        assert not self.kv_overallocated_freed
        self.pop_overallocated_calls += 1
        self.kv_overallocated_freed = True
        return self.kv_committed_len, self.kv_allocated_len


def test_streaming_release_kv_cache_defers_tail_free(monkeypatch):
    """Spec tail is NOT trimmed in cache_finished_req; it is deferred to
    match_prefix's orphan tail free on the next turn. cache_finished_req
    only sets bookkeeping flags and saves the slot as-is."""
    page_size = 16
    req_to_token = torch.arange(128, dtype=torch.int32).reshape(1, 128)
    req_to_token_pool = SimpleNamespace(req_to_token=req_to_token, free_slots=[])
    allocator = _FakeAllocator()
    tree_cache = SessionAwareCache(
        _FakeInnerCache(req_to_token_pool, allocator, page_size)
    )
    req = _FakeReq("session-a", req_pool_idx=0, committed=17, allocated=40)

    monkeypatch.setattr(
        "sglang.srt.mem_cache.common.get_global_server_args",
        lambda: SimpleNamespace(page_size=page_size, speculative_algorithm="eagle"),
    )

    release_kv_cache(req, tree_cache)

    slot = tree_cache.slots["session-a"]
    assert req.kv_committed_freed is True
    assert req.kv_overallocated_freed is True
    assert req.req_pool_idx is None
    # Slot keeps the full allocation — tail free is deferred to match_prefix.
    assert slot.kv_committed_len == 17
    assert slot.kv_allocated_len == 40
    assert len(allocator.freed) == 0


def test_match_prefix_abort_does_not_restore_live_session_slot():
    req_to_token = torch.arange(256, dtype=torch.int32).reshape(2, 128)
    req_to_token_pool = SimpleNamespace(req_to_token=req_to_token, free_slots=[])
    allocator = _FakeAllocator()
    inner = _FakeInnerCache(
        req_to_token_pool,
        allocator,
        page_size=16,
        match_results=[
            MatchResult(
                device_indices=torch.tensor([], dtype=torch.int64),
                last_device_node=None,
                last_host_node=None,
            )
        ],
    )
    tree_cache = SessionAwareCache(inner)
    tree_cache.slots["session-a"] = SessionSlot(
        req_pool_idx=0,
        kv_committed_len=48,
        kv_allocated_len=48,
        cache_protected_len=16,
    )

    req = _FakeReq("session-a", req_pool_idx=1, committed=1, allocated=1)
    req.to_finish = FINISH_ABORT("too long")

    result = tree_cache.match_prefix(
        SimpleNamespace(
            req=req,
            key=SimpleNamespace(token_ids=list(range(64))),
        )
    )

    slot = tree_cache.slots["session-a"]
    assert req.req_pool_idx == 1
    assert req.kv_committed_len == 1
    assert req.kv_allocated_len == 1
    assert slot.req_pool_idx == 0
    assert slot.kv_committed_len == 48
    assert slot.kv_allocated_len == 48
    assert len(result.device_indices) == 0


def test_aborted_streaming_turn_preserves_slot_and_accounting(monkeypatch):
    page_size = 16
    req_to_token = torch.arange(256, dtype=torch.int32).reshape(2, 128)
    req_to_token_pool = SimpleNamespace(req_to_token=req_to_token, free_slots=[])
    allocator = _FakeAllocator()
    tree_cache = SessionAwareCache(
        _FakeInnerCache(req_to_token_pool, allocator, page_size)
    )
    tree_cache.slots["session-a"] = SessionSlot(
        req_pool_idx=0,
        kv_committed_len=48,
        kv_allocated_len=48,
        cache_protected_len=16,
        swa_evicted_seqlen=8,
        last_node="lock-node",
    )

    req = _FakeReq("session-a", req_pool_idx=1, committed=5, allocated=23)
    req.finished_reason = FINISH_ABORT("too long")

    monkeypatch.setattr(
        "sglang.srt.mem_cache.common.get_global_server_args",
        lambda: SimpleNamespace(page_size=page_size, speculative_algorithm="eagle"),
    )

    release_kv_cache(req, tree_cache)

    # Abort nukes the slot (release_session) + frees transient pool slot.
    assert "session-a" not in tree_cache.slots
    assert req.kv_committed_freed is True
    assert req.kv_overallocated_freed is True
    assert req.req_pool_idx is None
    assert tree_cache.session_held_tokens() == 0
    assert tree_cache.session_held_req_count() == 0
    # Slot's KV freed by release_session + transient freed separately.
    assert 0 in req_to_token_pool.free_slots  # slot's pool_idx
    assert 1 in req_to_token_pool.free_slots  # transient pool_idx


def test_first_request_abort_does_not_create_slot():
    """When the very first request on a session is aborted, no slot should
    be created. The session stays empty for the next attempt."""
    page_size = 1
    req_to_token = torch.arange(128, dtype=torch.int32).reshape(1, 128)
    req_to_token_pool = SimpleNamespace(req_to_token=req_to_token, free_slots=[])
    allocator = _FakeAllocator()
    inner = _FakeInnerCache(req_to_token_pool, allocator, page_size)
    tree_cache = SessionAwareCache(inner)

    # No slot exists yet (first request).
    req = _FakeReq("session-a", req_pool_idx=0, committed=0, allocated=20)
    req.finished_reason = FINISH_ABORT("input too long")

    tree_cache.cache_finished_req(req)

    # Slot must NOT be created.
    assert "session-a" not in tree_cache.slots
    # Transient pool slot freed.
    assert req.req_pool_idx is None
    assert req_to_token_pool.free_slots == [0]
    assert len(allocator.freed) == 1
    assert allocator.freed[0].tolist() == list(range(20))
    # Bookkeeping flags set.
    assert req.kv_committed_freed is True
    assert req.kv_overallocated_freed is True


def test_mid_processing_abort_wipes_session_slot():
    """When a running streaming session req is aborted mid-processing,
    ALL KV is wiped (release_session). Slot is deleted. Token IDs stay
    in req_nodes for next turn's re-prefill."""
    page_size = 1
    req_to_token = torch.arange(256, dtype=torch.int32).reshape(2, 128)
    req_to_token_pool = SimpleNamespace(req_to_token=req_to_token, free_slots=[])
    allocator = _FakeAllocator()
    inner = _FakeInnerCache(req_to_token_pool, allocator, page_size)
    tree_cache = SessionAwareCache(inner)

    # Session already has a slot from a previous turn.
    tree_cache.slots["session-a"] = SessionSlot(
        req_pool_idx=0,
        kv_committed_len=50,
        kv_allocated_len=50,
        last_node=None,
        cache_protected_len=0,
    )

    # Mid-processing abort: req has the SESSION slot's pool_idx (restore_to_req ran).
    req = _FakeReq("session-a", req_pool_idx=0, committed=60, allocated=65)
    req.finished_reason = FINISH_ABORT("client disconnected")

    tree_cache.cache_finished_req(req)

    # Slot wiped — deleted from slots dict.
    assert "session-a" not in tree_cache.slots
    # All KV freed: [0, 65) from release_session (slot extended to req's allocated).
    assert len(allocator.freed) == 1
    assert allocator.freed[0].tolist() == list(range(65))
    # Pool slot returned.
    assert req_to_token_pool.free_slots == [0]
    assert req.req_pool_idx is None
    # Bookkeeping flags set.
    assert req.kv_committed_freed is True
    assert req.kv_overallocated_freed is True


# Shrink tests removed: streaming sessions are append-only after the
# rollback fix in session_controller (rollback_aborted_req).  The shrink
# code path in cache_finished_req no longer exists.
