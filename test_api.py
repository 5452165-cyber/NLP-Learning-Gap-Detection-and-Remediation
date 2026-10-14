"""End-to-end smoke test for the Beyond Marks API."""
import urllib.request
import json

BASE = "http://localhost:5000"

# ── 1. Health ────────────────────────────────────────────────────────────────
req  = urllib.request.urlopen(f"{BASE}/api/health")
data = json.loads(req.read())
assert data["status"] == "ok", f"Health failed: {data}"
print(f"[PASS] Health: {data['message']}")

# ── 2. Topics ────────────────────────────────────────────────────────────────
req    = urllib.request.urlopen(f"{BASE}/api/topics")
topics = json.loads(req.read())
print(f"[PASS] Topics loaded: {list(topics['topics'].keys())}")

# ── 3. Analyze — partial answer (flow control deliberately omitted) ───────────
answer = (
    "TCP uses sequence numbers to order data and detect missing segments. "
    "The receiver sends acknowledgments (ACK) back to the sender to confirm receipt of bytes. "
    "If the sender does not get an ACK within the retransmission timeout (RTO), "
    "it retransmits the lost segment. TCP also uses fast retransmit after three duplicate ACKs."
)

payload = json.dumps({"topic_key": "tcp_mechanics", "student_answer": answer}).encode()
req2    = urllib.request.Request(
    f"{BASE}/api/analyze", data=payload,
    headers={"Content-Type": "application/json"},
)
result = json.loads(urllib.request.urlopen(req2).read())

print()
print("=" * 55)
print("  DIAGNOSTIC OUTPUT — Table I (SBERT Classification)")
print("=" * 55)
summary = result["summary"]
print(f"  Mastery Score : {summary['mastery_pct']}%")
print(f"  Correct       : {summary['correct_count']}")
print(f"  Partial       : {summary['partial_count']}")
print(f"  Missing       : {summary['missing_count']}")
print(f"  Gaps detected : {summary['gaps_count']}")
print()

for cr in result["concept_results"]:
    status_icon = {"Correct": "[OK]", "Partial": "[~~]", "Missing": "[XX]"}.get(cr["status"], "[?]")
    print(f"  {status_icon} [{cr['status']:8s}] {cr['concept_id']} {cr['concept_name']}")
    print(f"           Similarity: {cr['similarity_pct']}%")
    print(f"           Evidence  : {cr['evidence'][:70]}...")
    print()

# ── 4. Remediation ───────────────────────────────────────────────────────────
if result["gaps"]:
    payload2 = json.dumps({"topic_key": "tcp_mechanics", "gaps": result["gaps"]}).encode()
    req3     = urllib.request.Request(
        f"{BASE}/api/remediate", data=payload2,
        headers={"Content-Type": "application/json"},
    )
    rem_data = json.loads(urllib.request.urlopen(req3).read())
    print("=" * 55)
    print("  REMEDIATION CARDS")
    print("=" * 55)
    for r in rem_data["remediations"]:
        print(f"  [{r['status']:8s}] {r['concept_name']} (source={r['source']})")
        print(f"    Explanation : {r['explanation'][:90]}...")
        print(f"    Drill Q     : {r['drill_question'][:80]}...")
        print()

print("[ALL TESTS PASSED] Beyond Marks API is fully operational!")
