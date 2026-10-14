"""
Remediation Engine — Stage 5 of the 8-Stage Methodology
--------------------------------------------------------
Stage 5 : Gap Mapping & Personalized Remediation Generation via LLM

Paper: "Beyond Marks: NLP-Based Learning Gap Detection and Remediation"
"""

import json
import logging
import re
import warnings
from typing import Dict, List

# Suppress deprecation warnings from google packages (still functional on Python 3.10)
warnings.filterwarnings("ignore", category=FutureWarning, module="google")

import google.generativeai as genai

from config import GEMINI_API_KEY, REMEDIATION_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)

# ─── Gemini Client Setup ──────────────────────────────────────────────────────
_gemini_configured = False

def _configure_gemini() -> bool:
    global _gemini_configured
    if not _gemini_configured:
        if not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key_here":
            logger.warning("Gemini API key not set — will use fallback remediation.")
            return False
        genai.configure(api_key=GEMINI_API_KEY)
        _gemini_configured = True
    return True


# ─── Fallback Static Remediation Templates ───────────────────────────────────
STATIC_FALLBACKS: Dict[str, Dict] = {
    # ── TCP Mechanics ──────────────────────────────────────────────────────────
    "Reliability and Sequence Numbers": {
        "explanation": (
            "TCP assigns a sequence number to every byte of data it sends. "
            "The receiver uses these numbers to reorder segments that arrive out of order "
            "and to detect any missing data, ensuring the application always receives a "
            "complete and ordered byte stream."
        ),
        "example": (
            "If TCP sends three 500-byte segments with sequence numbers 1, 501, and 1001, "
            "and segment 501 arrives before segment 1, the receiver buffers it "
            "and waits for segment 1 before delivering data in order to the application."
        ),
        "drill_question": (
            "A receiver gets TCP segments with sequence numbers 1001, 1 and 501 in that order. "
            "In what order will the application receive the data, and why?"
        ),
        "answer_key": (
            "The application receives data in the order 1 → 501 → 1001. "
            "TCP reassembles segments by sequence number before passing them up, "
            "ensuring a correctly ordered byte stream regardless of network reordering."
        ),
    },
    "Acknowledgments (ACK)": {
        "explanation": (
            "TCP uses cumulative ACKs where the receiver sends back the sequence number of "
            "the next byte it expects. This confirms that all bytes up to that point have "
            "been received successfully, allowing the sender to track delivery progress."
        ),
        "example": (
            "If the receiver has correctly received bytes 1–1000, it sends ACK=1001, "
            "telling the sender 'I have everything up to byte 1000; send from 1001 next.' "
            "If byte 501 is missing, the receiver keeps sending ACK=501 until it arrives."
        ),
        "drill_question": (
            "A sender transmits bytes 1–500, 501–1000, and 1001–1500. "
            "The segment 501–1000 is lost. What ACK value will the receiver keep sending, and why?"
        ),
        "answer_key": (
            "The receiver keeps sending ACK=501. TCP uses cumulative acknowledgments, "
            "so it can only acknowledge contiguous data. Even though 1001–1500 arrived, "
            "the ACK cannot advance past the gap caused by the missing 501–1000 segment."
        ),
    },
    "Retransmission on Timeout or Packet Loss": {
        "explanation": (
            "If a sender does not receive an ACK within the Retransmission Timeout (RTO), "
            "it assumes the segment was lost and retransmits it. "
            "TCP also retransmits immediately after receiving three duplicate ACKs (fast retransmit), "
            "recovering faster without waiting for the full timeout."
        ),
        "example": (
            "If RTO is 200 ms and the sender transmits at t=0 but gets no ACK by t=200 ms, "
            "it retransmits. Alternatively, if the receiver sends ACK=501 three times in a row, "
            "the sender immediately retransmits starting at byte 501 (fast retransmit)."
        ),
        "drill_question": (
            "What is the difference between TCP timeout-based retransmission and fast retransmit? "
            "When is each triggered?"
        ),
        "answer_key": (
            "Timeout-based retransmission fires when the RTO timer expires before an ACK arrives; "
            "the sender retransmits the oldest unacknowledged segment. "
            "Fast retransmit fires immediately upon receiving three duplicate ACKs for the same "
            "sequence number, enabling quicker loss recovery without waiting for the RTO to expire."
        ),
    },
    "Flow Control and Windowing": {
        "explanation": (
            "TCP flow control prevents the sender from overwhelming the receiver's buffer. "
            "The receiver advertises its available buffer space as the window size (rwnd), "
            "and the sender must keep in-flight unacknowledged bytes below this limit."
        ),
        "example": (
            "If the receiver has a 16 KB buffer with 4 KB already occupied, it advertises "
            "rwnd=12 KB. The sender limits in-flight data to 12 KB. Once the application "
            "reads those 4 KB, the receiver sends an updated rwnd=16 KB, allowing more data."
        ),
        "drill_question": (
            "A receiver has a 32 KB buffer. Its application has read 10 KB, and 6 KB of "
            "unread data is still buffered. What rwnd value should the receiver advertise?"
        ),
        "answer_key": (
            "rwnd = 32 KB − 6 KB = 26 KB. The receive window equals total buffer size "
            "minus currently buffered unread data, representing the remaining free space "
            "the sender may fill without overwhelming the receiver."
        ),
    },
    # ── OS Scheduling ──────────────────────────────────────────────────────────
    "Scheduling Criteria (CPU Utilization & Throughput)": {
        "explanation": (
            "CPU scheduling aims to maximize CPU utilization (keep the processor busy) "
            "and throughput (complete as many processes per unit time as possible). "
            "These two metrics together measure how productively the CPU is being used."
        ),
        "example": (
            "A batch server running at 100% CPU utilization and completing 50 jobs/hour "
            "is better utilized than one at 60% with 30 jobs/hour. "
            "However, 100% utilization with very few completions indicates too much overhead."
        ),
        "drill_question": "Why are both CPU utilization and throughput important scheduling metrics?",
        "answer_key": (
            "CPU utilization measures efficiency — idle CPU wastes resources. "
            "Throughput measures productivity — completing more processes per second. "
            "High utilization with low throughput could indicate excessive context-switching overhead, "
            "so both must be balanced for an effective scheduler."
        ),
    },
    "Preemptive vs Non-Preemptive Scheduling": {
        "explanation": (
            "In non-preemptive scheduling a running process holds the CPU until it terminates "
            "or blocks voluntarily. Preemptive scheduling lets the OS forcibly remove a process "
            "from the CPU (e.g., on a timer interrupt), enabling better responsiveness."
        ),
        "example": (
            "Round Robin is preemptive — a timer interrupt forces a context switch after each "
            "quantum. FCFS is non-preemptive — a CPU-bound process runs to completion before "
            "any other process gets the CPU, causing long wait times for short jobs."
        ),
        "drill_question": "Give one advantage and one disadvantage of preemptive scheduling.",
        "answer_key": (
            "Advantage: better responsiveness for interactive and real-time processes since "
            "high-priority jobs are never blocked by a running low-priority process. "
            "Disadvantage: overhead from frequent context switches and potential race conditions "
            "that require synchronisation mechanisms like mutexes."
        ),
    },
    "Round Robin and Time Quantum": {
        "explanation": (
            "Round Robin gives each ready process a fixed time slice (quantum) in circular order. "
            "When the quantum expires the process is preempted and added to the end of the ready "
            "queue, ensuring all processes get equal CPU time."
        ),
        "example": (
            "With a 20 ms quantum and three processes (P1=60 ms, P2=40 ms, P3=20 ms): "
            "P1 runs 0–20 ms, P2 20–40 ms, P3 40–60 ms (done), P1 60–80 ms, "
            "P2 80–100 ms (done), P1 100–120 ms (done)."
        ),
        "drill_question": "How does the choice of time quantum affect Round Robin performance?",
        "answer_key": (
            "A very small quantum causes excessive context switches, increasing overhead and "
            "reducing useful work. A very large quantum degrades to FCFS behaviour, hurting "
            "responsiveness. The optimal quantum balances responsiveness with context-switch "
            "overhead — typically 10–100 ms in practice."
        ),
    },
    "Priority Scheduling and Starvation": {
        "explanation": (
            "Priority scheduling always runs the highest-priority ready process first. "
            "Low-priority processes may wait indefinitely (starvation) if high-priority "
            "processes keep arriving. Aging — gradually increasing a waiting process's priority "
            "— is the standard solution to prevent starvation."
        ),
        "example": (
            "A low-priority P1 waits while high-priority P2, P3, … keep arriving. "
            "With aging, every minute P1's priority increases by 1. After 10 minutes its "
            "priority exceeds that of new arrivals, and it finally gets scheduled."
        ),
        "drill_question": "What is starvation in priority scheduling, and how does aging prevent it?",
        "answer_key": (
            "Starvation occurs when a low-priority process never gets CPU time because "
            "higher-priority processes continuously arrive. Aging prevents this by incrementally "
            "raising the priority of waiting processes over time, guaranteeing that any process "
            "will eventually reach a high enough priority to be scheduled."
        ),
    },
    # ── OOP Principles ─────────────────────────────────────────────────────────
    "Encapsulation": {
        "explanation": (
            "Encapsulation bundles data (attributes) and the methods that operate on it into "
            "a single class, restricting direct access to internal state via access modifiers "
            "(private, protected, public) and exposing only a controlled public interface."
        ),
        "example": (
            "A BankAccount class has a private _balance attribute. "
            "Only deposit() and withdraw() public methods can modify it, "
            "preventing external code from directly setting balance to a negative value."
        ),
        "drill_question": "How does encapsulation improve code maintainability?",
        "answer_key": (
            "By hiding internal implementation details, changes to private attributes or "
            "methods don't break external code that only uses the public interface. "
            "This reduces coupling, making refactoring safer and the codebase easier to maintain."
        ),
    },
    "Inheritance": {
        "explanation": (
            "Inheritance lets a child (derived) class acquire the attributes and methods of a "
            "parent (base) class, enabling code reuse. The child can override or extend inherited "
            "behaviour to specialise the parent's functionality without rewriting shared code."
        ),
        "example": (
            "class Animal has a speak() method. class Dog(Animal) overrides speak() to return "
            "'Woof', and class Cat(Animal) overrides it to return 'Meow'. Both reuse Animal's "
            "other methods like eat() and breathe() without duplication."
        ),
        "drill_question": "What is the difference between method overriding and method overloading?",
        "answer_key": (
            "Overriding redefines a parent class method in a child class with the same name and "
            "signature, enabling runtime polymorphism. Overloading defines multiple methods with "
            "the same name but different parameter lists in the same class, resolved at compile time."
        ),
    },
    "Polymorphism": {
        "explanation": (
            "Polymorphism allows objects of different classes to respond to the same method call "
            "in their own way. Runtime polymorphism uses method overriding; compile-time "
            "polymorphism uses method overloading."
        ),
        "example": (
            "shapes = [Circle(), Rectangle(), Triangle()] \n"
            "for shape in shapes: print(shape.area()) \n"
            "Each shape computes area() differently, but the calling code is identical — "
            "demonstrating runtime polymorphism through a common interface."
        ),
        "drill_question": "How does polymorphism promote the Open/Closed Principle?",
        "answer_key": (
            "The Open/Closed Principle states classes should be open for extension but closed for "
            "modification. Polymorphism enables adding new shape subclasses (open for extension) "
            "without changing existing code that calls shape.area() (closed for modification), "
            "because new behaviour is added via new subclasses rather than editing existing ones."
        ),
    },
    "Abstraction": {
        "explanation": (
            "Abstraction hides complex implementation details and exposes only the essential "
            "features of an object. In OOP this is achieved through abstract classes and interfaces, "
            "letting programmers work at a high level without worrying about low-level specifics."
        ),
        "example": (
            "A DatabaseConnector abstract class defines connect(), query(), close() with no "
            "implementation. MySQLConnector and PostgreSQLConnector provide concrete implementations. "
            "Application code uses only the abstract interface and works with any database "
            "backend without knowing the underlying details."
        ),
        "drill_question": "What is the difference between abstraction and encapsulation?",
        "answer_key": (
            "Abstraction hides complexity by providing a simplified interface — it focuses on "
            "WHAT an object does. Encapsulation hides internal data by bundling it with methods "
            "and restricting access — it focuses on HOW an object's data is protected. "
            "They complement each other but serve different design goals."
        ),
    },
}


# ─── Fallback Helper ──────────────────────────────────────────────────────────
def _fallback_remediation(concept_name: str, status: str) -> Dict:
    """Return a static fallback when the LLM is unavailable."""
    template = STATIC_FALLBACKS.get(concept_name)
    if not template:
        template = {
            "explanation": (
                f"The concept '{concept_name}' requires further study. "
                "Review your course materials and lecture notes on this topic carefully."
            ),
            "example": (
                f"Search for worked examples of '{concept_name}' in textbooks "
                "or resources like GeeksForGeeks, Wikipedia, or your lecture slides."
            ),
            "drill_question": f"Explain '{concept_name}' in your own words with an example.",
            "answer_key": "Refer to your course notes for the complete answer.",
        }
    return dict(template)  # return a copy


# ─── Stage 5: Remediation Generation ─────────────────────────────────────────
def generate_remediation(
    gap: Dict,
    topic_title: str,
    concept_description: str,
) -> Dict:
    """
    Generate personalised remediation for a single learning gap.
    Tries Gemini API first; falls back to static templates on failure.
    """
    concept_name = gap["concept_name"]
    status       = gap["status"]
    evidence     = gap.get("evidence", "Not mentioned.")

    if not _configure_gemini():
        result = _fallback_remediation(concept_name, status)
        result["source"] = "fallback"
        return result

    prompt = REMEDIATION_PROMPT_TEMPLATE.format(
        concept_name=concept_name,
        status=status,
        topic_title=topic_title,
        concept_description=concept_description,
        student_evidence=evidence,
    )

    try:
        model    = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        raw_text = response.text.strip()

        # Strip markdown fences if present
        raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
        raw_text = re.sub(r"\s*```$", "", raw_text)

        parsed = json.loads(raw_text)
        required_keys = {"explanation", "example", "drill_question", "answer_key"}
        if not required_keys.issubset(parsed.keys()):
            raise ValueError("LLM response missing required keys.")

        parsed["source"] = "gemini"
        logger.info("Gemini remediation generated for: %s", concept_name)
        return parsed

    except json.JSONDecodeError as e:
        logger.warning("Gemini JSON parse error for '%s': %s", concept_name, e)
    except Exception as e:
        logger.warning("Gemini API error for '%s': %s", concept_name, e)

    result = _fallback_remediation(concept_name, status)
    result["source"] = "fallback"
    return result


# ─── Batch Remediation ────────────────────────────────────────────────────────
def generate_all_remediations(gaps: List[Dict], topic: Dict) -> List[Dict]:
    """
    Generate remediation for every gap in the analysis result.
    Returns a list of remediation dicts enriched with gap metadata.
    """
    concept_map  = {c["id"]: c for c in topic["expected_concepts"]}
    remediations = []

    for gap in gaps:
        concept     = concept_map.get(gap["concept_id"], {})
        remediation = generate_remediation(
            gap=gap,
            topic_title=topic["title"],
            concept_description=concept.get("description", ""),
        )
        remediations.append({
            "concept_id":   gap["concept_id"],
            "concept_name": gap["concept_name"],
            "status":       gap["status"],
            "similarity":   gap["similarity"],
            "evidence":     gap["evidence"],
            **remediation,
        })

    return remediations
