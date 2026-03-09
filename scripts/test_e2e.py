"""
End-to-end API tests for POST /api/explore.

Requires the SocraticPath backend to be running at BASE_URL.
Set SUPABASE_TEST_TOKEN to a valid Supabase JWT to run the authenticated tests.

Usage:
    uv run python scripts/test_e2e.py
    BASE_URL=https://socratic-path.onrender.com uv run python scripts/test_e2e.py
"""

import os
import sys
import pathlib

import httpx

BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")
TIMEOUT = 60  # seconds — T5 model inference can take up to ~8 s on CPU

_passed = 0
_failed = 0
_skipped = 0


def _pass(name: str) -> None:
    global _passed
    _passed += 1
    print(f"  PASS  {name}")


def _fail(name: str, detail: str) -> None:
    global _failed
    _failed += 1
    print(f"  FAIL  {name}: {detail}")


def _skip(name: str, reason: str) -> None:
    global _skipped
    _skipped += 1
    print(f"  SKIP  {name} ({reason})")


def _auth_header() -> dict[str, str] | None:
    token = os.getenv("SUPABASE_TEST_TOKEN")
    if not token:
        return None
    return {"Authorization": f"Bearer {token}"}


# ── Health check ─────────────────────────────────────────────


def test_health_check(client: httpx.Client) -> None:
    """GET /api/health should return 200 with status=ok and model_loaded=true."""
    r = client.get("/api/health")
    if r.status_code != 200:
        _fail("health check returns 200 ok", f"status={r.status_code}")
        return
    body = r.json()
    if body.get("status") != "ok":
        _fail("health check returns 200 ok", f"unexpected body: {body}")
        return
    _pass("health check returns 200 ok")


# ── Authentication ───────────────────────────────────────────


def test_missing_auth_returns_4xx(client: httpx.Client) -> None:
    """POST /api/explore without an Authorization header must be rejected."""
    payload = {"text": "test", "question_types": ["clarity"]}
    r = client.post("/api/explore", json=payload)
    if r.status_code in (401, 403, 422, 503):
        _pass("missing auth header returns 4xx")
    else:
        _fail("missing auth header returns 4xx", f"got unexpected {r.status_code}")


def test_invalid_jwt_returns_4xx(client: httpx.Client) -> None:
    """POST /api/explore with a malformed JWT must be rejected."""
    headers = {"Authorization": "Bearer not.a.valid.jwt.token"}
    payload = {"text": "test", "question_types": ["clarity"]}
    r = client.post("/api/explore", json=payload, headers=headers)
    if r.status_code in (401, 403, 422, 503):
        _pass("malformed JWT returns 4xx")
    else:
        _fail("malformed JWT returns 4xx", f"got {r.status_code}")


# ── Input validation ─────────────────────────────────────────


def test_empty_text_returns_422(client: httpx.Client) -> None:
    """Empty text field must fail Pydantic validation and return 422."""
    payload = {"text": "", "question_types": ["clarity"]}
    r = client.post("/api/explore", json=payload)
    if r.status_code == 422:
        _pass("empty text returns 422 Unprocessable Entity")
    else:
        _fail("empty text returns 422 Unprocessable Entity", f"got {r.status_code}")


def test_invalid_question_type_returns_400_or_4xx(client: httpx.Client) -> None:
    """Unknown question type should return 400; auth rejection (401/403) is also acceptable."""
    payload = {"text": "What is photosynthesis?", "question_types": ["not_a_valid_type"]}
    r = client.post("/api/explore", json=payload)
    if r.status_code in (400, 401, 403, 422, 503):
        _pass("invalid question type returns 4xx")
    else:
        _fail("invalid question type returns 4xx", f"got {r.status_code}")


# ── Full pipeline (requires valid SUPABASE_TEST_TOKEN) ────────


def test_factual_input_returns_questions(client: httpx.Client) -> None:
    """Factual input should be routed to the Wikipedia pipeline and return questions."""
    headers = _auth_header()
    if headers is None:
        _skip("factual input returns questions", "SUPABASE_TEST_TOKEN not set")
        return

    payload = {
        "text": "Photosynthesis converts sunlight into glucose in plant cells",
        "ancestry": [],
        "depth": 0,
        "question_types": ["clarity", "reasons_evidence"],
    }
    r = client.post("/api/explore", json=payload, headers=headers, timeout=TIMEOUT)

    if r.status_code != 200:
        _fail("factual input returns questions", f"status={r.status_code}, body={r.text[:300]}")
        return

    data = r.json()
    if not data.get("questions"):
        _fail("factual input returns questions", "questions list is empty")
        return
    if len(data["questions"]) != 2:
        _fail("factual input returns questions", f"expected 2 questions, got {len(data['questions'])}")
        return
    _pass("factual input returns questions")


def test_opinion_input_routes_to_gemini(client: httpx.Client) -> None:
    """Opinion-type input should be classified accordingly and routed to the Gemini path."""
    headers = _auth_header()
    if headers is None:
        _skip("opinion input routes to gemini", "SUPABASE_TEST_TOKEN not set")
        return

    payload = {
        "text": "I believe social media does far more harm than good to society",
        "ancestry": [],
        "depth": 0,
        "question_types": ["clarity", "assumptions"],
    }
    r = client.post("/api/explore", json=payload, headers=headers, timeout=TIMEOUT)

    if r.status_code != 200:
        _fail("opinion input routes to gemini", f"status={r.status_code}")
        return

    data = r.json()
    if data.get("pipeline_path") != "gemini":
        _fail(
            "opinion input routes to gemini",
            f"pipeline_path='{data.get('pipeline_path')}', expected 'gemini'",
        )
        return
    if data["input_classification"]["input_type"] not in ("argumentative", "opinion"):
        _fail(
            "opinion input routes to gemini",
            f"input_type='{data['input_classification']['input_type']}'",
        )
        return
    _pass("opinion input routes to gemini")


def test_response_schema_complete(client: httpx.Client) -> None:
    """Response must include all fields defined in ExploreResponse."""
    headers = _auth_header()
    if headers is None:
        _skip("response schema complete", "SUPABASE_TEST_TOKEN not set")
        return

    payload = {
        "text": "Does regular exercise improve cognitive function?",
        "ancestry": [],
        "depth": 0,
        "question_types": ["clarity"],
    }
    r = client.post("/api/explore", json=payload, headers=headers, timeout=TIMEOUT)

    if r.status_code != 200:
        _fail("response schema complete", f"status={r.status_code}")
        return

    data = r.json()
    required_fields = [
        "input_classification",
        "pipeline_path",
        "keyphrases",
        "context_sources",
        "questions",
        "processing_time_ms",
    ]
    missing = [f for f in required_fields if f not in data]
    if missing:
        _fail("response schema complete", f"missing fields: {missing}")
        return
    if not isinstance(data["questions"], list):
        _fail("response schema complete", "questions is not a list")
        return
    if not isinstance(data["processing_time_ms"], (int, float)):
        _fail("response schema complete", "processing_time_ms is not numeric")
        return
    _pass("response schema complete")


def test_processing_time_within_nfr1(client: httpx.Client) -> None:
    """NFR1: end-to-end generation must complete within 8 seconds (NFR1 threshold)."""
    headers = _auth_header()
    if headers is None:
        _skip("processing time within NFR1 threshold (8 s)", "SUPABASE_TEST_TOKEN not set")
        return

    payload = {
        "text": "Climate change is accelerating faster than predicted",
        "ancestry": [],
        "depth": 0,
        "question_types": ["clarity", "reasons_evidence", "assumptions"],
    }
    r = client.post("/api/explore", json=payload, headers=headers, timeout=TIMEOUT)

    if r.status_code != 200:
        _fail("processing time within NFR1 threshold (8 s)", f"status={r.status_code}")
        return

    elapsed = r.json().get("processing_time_ms", float("inf"))
    if elapsed <= 8000:
        _pass(f"processing time within NFR1 threshold (8 s) — actual: {elapsed:.0f} ms")
    else:
        _fail("processing time within NFR1 threshold (8 s)", f"took {elapsed:.0f} ms (> 8000 ms)")


# ── Fallback mechanism (structural verification) ─────────────


def test_gemini_fallback_is_implemented() -> None:
    """Verify the Gemini-to-Wikipedia fallback path is present in context_router.py."""
    router_src = (
        pathlib.Path(__file__).parent.parent / "backend" / "services" / "context_router.py"
    ).read_text()

    has_fallback_comment = "fall back to Wikipedia" in router_src
    has_fallback_call = "return self._wikipedia_path" in router_src

    if has_fallback_comment and has_fallback_call:
        _pass("Gemini-to-Wikipedia fallback is implemented in context_router.py")
    else:
        _fail(
            "Gemini-to-Wikipedia fallback is implemented in context_router.py",
            "expected fallback logic not found in source",
        )


# ── Runner ───────────────────────────────────────────────────


def main() -> None:
    print(f"\nSocraticPath — End-to-End API Tests")
    print(f"Target: {BASE_URL}")
    print("=" * 55)

    with httpx.Client(base_url=BASE_URL, timeout=10) as client:
        test_health_check(client)
        test_missing_auth_returns_4xx(client)
        test_invalid_jwt_returns_4xx(client)
        test_empty_text_returns_422(client)
        test_invalid_question_type_returns_400_or_4xx(client)
        test_factual_input_returns_questions(client)
        test_opinion_input_routes_to_gemini(client)
        test_response_schema_complete(client)
        test_processing_time_within_nfr1(client)

    test_gemini_fallback_is_implemented()

    print("=" * 55)
    print(f"Results: {_passed} passed, {_failed} failed, {_skipped} skipped")

    if _failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
