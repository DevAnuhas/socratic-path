"""
End-to-end integration tests for the SocraticPath API.

Usage:
    # Start backend first:  uv run uvicorn backend.main:app --port 8000
    # Then run:              uv run python scripts/test_e2e.py

Tests the /api/generate endpoint with diverse topics and validates
response structure, question types, and graceful handling of edge cases.
"""

import sys
import time
import json
import httpx

BASE_URL = "http://localhost:8000"
TIMEOUT = 90  # seconds — model inference on CPU can be slow

# ── Test topics covering diverse domains ──────────────────────────────────────

TEST_TOPICS = [
    # Standard academic topics
    "Climate change effects on agriculture",
    "Social media and mental health in teenagers",
    "The ethics of artificial intelligence in healthcare",
    # Short / minimal inputs
    "Democracy",
    "Vaccines",
    # Longer / more argumentative inputs
    "Universal basic income would reduce poverty and inequality while also stimulating economic growth through increased consumer spending",
    "The education system should prioritise critical thinking over rote memorisation",
    # Technical / scientific
    "Quantum computing and its implications for cryptography",
    "CRISPR gene editing technology",
    # Niche / less common
    "The impact of microplastics on marine ecosystems",
    "Space colonisation as a solution to overpopulation",
    # Edge cases
    "Why?",  # Very short
    "12345 67890",  # Numeric
]

VALID_QUESTION_TYPES = {
    "clarity",
    "reasons_evidence",
    "implication_consequences",
    "alternate_viewpoints_perspectives",
    "assumptions",
}


def check_health():
    """Verify the backend is running and model is loaded."""
    try:
        resp = httpx.get(f"{BASE_URL}/api/health", timeout=10)
        data = resp.json()
        if not data.get("model_loaded"):
            print("FAIL: Backend running but model not loaded yet. Wait and retry.")
            return False
        print("OK: Backend healthy, model loaded")
        return True
    except httpx.ConnectError:
        print("FAIL: Cannot connect to backend. Start it with:")
        print("  uv run uvicorn backend.main:app --port 8000")
        return False


def test_generate(topic: str, question_types: list[str] | None = None) -> dict:
    """Send a generate request and validate the response."""
    payload = {"topic": topic}
    if question_types:
        payload["question_types"] = question_types

    t0 = time.time()
    resp = httpx.post(f"{BASE_URL}/api/generate", json=payload, timeout=TIMEOUT)
    elapsed = time.time() - t0

    if resp.status_code != 200:
        return {
            "topic": topic,
            "status": "FAIL",
            "reason": f"HTTP {resp.status_code}: {resp.text[:200]}",
            "time_s": round(elapsed, 1),
        }

    data = resp.json()
    errors = []

    # Validate response structure
    for field in ["topic", "keyphrases", "context_sources", "questions", "processing_time_ms"]:
        if field not in data:
            errors.append(f"Missing field: {field}")

    # Validate keyphrases
    if not data.get("keyphrases"):
        errors.append("No keyphrases returned")
    else:
        for kp in data["keyphrases"]:
            if not kp.get("text") or not isinstance(kp.get("score"), (int, float)):
                errors.append(f"Invalid keyphrase: {kp}")

    # Validate questions
    expected_types = set(question_types) if question_types else VALID_QUESTION_TYPES
    returned_types = set()
    for q in data.get("questions", []):
        if not q.get("text") or len(q["text"].strip()) < 3:
            errors.append(f"Empty or trivial question for type '{q.get('type')}'")
        if q.get("type") not in VALID_QUESTION_TYPES:
            errors.append(f"Invalid question type: {q.get('type')}")
        returned_types.add(q.get("type"))
        if not q.get("related_keyphrases"):
            errors.append(f"Question '{q.get('id')}' has no related keyphrases")

    missing_types = expected_types - returned_types
    if missing_types:
        errors.append(f"Missing question types: {missing_types}")

    # Validate context sources (may be empty for obscure topics — not an error)
    for src in data.get("context_sources", []):
        if not src.get("summary"):
            errors.append(f"Empty summary for source keyphrase '{src.get('keyphrase')}'")

    status = "PASS" if not errors else "WARN"
    return {
        "topic": topic[:50] + ("..." if len(topic) > 50 else ""),
        "status": status,
        "questions": len(data.get("questions", [])),
        "keyphrases": len(data.get("keyphrases", [])),
        "sources": len(data.get("context_sources", [])),
        "time_s": round(elapsed, 1),
        "errors": errors if errors else None,
    }


def test_error_cases():
    """Test that invalid requests return proper error responses."""
    results = []

    # Empty topic
    resp = httpx.post(f"{BASE_URL}/api/generate", json={"topic": ""}, timeout=10)
    results.append({
        "case": "Empty topic",
        "status": "PASS" if resp.status_code == 422 else "FAIL",
        "detail": f"HTTP {resp.status_code}",
    })

    # Invalid question types only
    resp = httpx.post(
        f"{BASE_URL}/api/generate",
        json={"topic": "test", "question_types": ["invalid_type"]},
        timeout=10,
    )
    results.append({
        "case": "Invalid question types",
        "status": "PASS" if resp.status_code == 400 else "FAIL",
        "detail": f"HTTP {resp.status_code}",
    })

    # Subset of question types
    resp = httpx.post(
        f"{BASE_URL}/api/generate",
        json={"topic": "Democracy", "question_types": ["clarity", "assumptions"]},
        timeout=TIMEOUT,
    )
    data = resp.json()
    returned_types = {q["type"] for q in data.get("questions", [])}
    results.append({
        "case": "Subset of types (clarity + assumptions)",
        "status": "PASS" if returned_types == {"clarity", "assumptions"} else "FAIL",
        "detail": f"Got types: {returned_types}",
    })

    return results


def main():
    print("=" * 60)
    print("SocraticPath End-to-End Test Suite")
    print("=" * 60)
    print()

    # Health check
    if not check_health():
        sys.exit(1)

    print()

    # Error cases
    print("── Error handling tests ─────────────────────────────────")
    error_results = test_error_cases()
    for r in error_results:
        icon = "+" if r["status"] == "PASS" else "x"
        print(f"  [{icon}] {r['case']}: {r['status']} ({r['detail']})")

    print()

    # Generate tests
    print("── Generate endpoint tests ─────────────────────────────")
    generate_results = []
    for topic in TEST_TOPICS:
        result = test_generate(topic)
        generate_results.append(result)
        icon = "+" if result["status"] == "PASS" else "!" if result["status"] == "WARN" else "x"
        line = f"  [{icon}] {result['topic']}: {result['status']}"
        line += f" ({result.get('questions', '?')}q, {result.get('keyphrases', '?')}kp, {result.get('sources', '?')}src, {result['time_s']}s)"
        print(line)
        if result.get("errors"):
            for err in result["errors"]:
                print(f"      ^ {err}")

    # Summary
    print()
    print("── Summary ─────────────────────────────────────────────")
    total = len(generate_results)
    passed = sum(1 for r in generate_results if r["status"] == "PASS")
    warned = sum(1 for r in generate_results if r["status"] == "WARN")
    failed = sum(1 for r in generate_results if r["status"] == "FAIL")
    times = [r["time_s"] for r in generate_results if r["status"] != "FAIL"]

    print(f"  Generate tests: {passed} passed, {warned} warnings, {failed} failed / {total} total")
    if times:
        print(f"  Avg response time: {sum(times)/len(times):.1f}s (min: {min(times):.1f}s, max: {max(times):.1f}s)")

    error_passed = sum(1 for r in error_results if r["status"] == "PASS")
    print(f"  Error tests: {error_passed}/{len(error_results)} passed")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
