"""
Configuration and Domain Knowledge for NLP-Based Learning Gap Detection System.
Paper: "Beyond Marks: NLP-Based Learning Gap Detection and Remediation"
Section V: 8-Stage Methodology
"""

import os
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

# ─── API Configuration ──────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# ─── SBERT Model ─────────────────────────────────────────────────────────────
SBERT_MODEL_NAME = "all-MiniLM-L6-v2"
SBERT_EMBEDDING_DIM = 384

# ─── Tri-State Classification Thresholds ─────────────────────────────────────
THRESHOLD_CORRECT  = 0.70   # cosine similarity >= 0.70 → Correct
THRESHOLD_PARTIAL  = 0.40   # 0.40 <= score < 0.70     → Partial
# score < 0.40              →                             Missing

# ─── Sample Answers for Pre-loaded Topics ────────────────────────────────────
SAMPLE_ANSWERS = {
    "tcp_mechanics": {
        "strong": (
            "TCP ensures reliable data delivery through several fundamental mechanisms. "
            "First, it assigns a unique sequence number to every byte of data sent. These sequence "
            "numbers allow the receiver to reorder segments that arrive out of order and detect missing data, "
            "ensuring the application receives a continuous, ordered byte stream.\n\n"
            "Second, the receiver sends cumulative acknowledgment (ACK) numbers back to confirm receipt. "
            "An ACK number indicates the next expected byte, so the sender knows exactly what has been confirmed.\n\n"
            "Third, if no ACK arrives within the Retransmission Timeout (RTO), TCP assumes the segment was lost "
            "and retransmits it. Fast retransmit also recovers from loss quickly when three duplicate ACKs arrive.\n\n"
            "Finally, TCP implements flow control using a sliding window. The receiver advertises its available "
            "buffer space as the receive window (rwnd), and the sender limits in-flight unacknowledged data to "
            "this value, preventing the sender from overflowing the receiver's buffer."
        ),
        "partial": (
            "TCP is a reliable transport layer protocol that uses sequence numbers so the receiver can put "
            "packets back in the correct order. When packets arrive, the receiver sends back acknowledgments "
            "so the sender knows data was delivered. If a packet gets dropped, the sender will eventually "
            "retransmit it after a timeout period. TCP also has mechanisms to avoid sending too fast."
        ),
        "weak": (
            "TCP is used to send data over the internet reliably. It connects two computers and makes sure "
            "the data gets there without errors. The protocol has headers with various fields. When data is "
            "received, a response is sent back. TCP is more reliable than UDP because it checks that data arrives."
        ),
    },
    "os_scheduling": {
        "strong": (
            "CPU scheduling in modern operating systems manages how processes share the processor. "
            "The primary scheduling criteria include CPU utilization (keeping the processor busy) and throughput "
            "(completing maximum processes per unit time).\n\n"
            "Scheduling can be preemptive, where the OS can forcibly remove a running process via a timer interrupt "
            "or higher-priority arrival, or non-preemptive, where a process holds the CPU until it voluntarily blocks or terminates.\n\n"
            "Round Robin is a preemptive algorithm that gives each process a fixed time quantum in circular order, "
            "ensuring fairness. The quantum size is critical: too small incurs excessive context-switching overhead, while "
            "too large degrades into FCFS behaviour.\n\n"
            "Priority scheduling always runs the highest-priority process first. Its major drawback is starvation, "
            "where low-priority processes wait indefinitely. Aging solves this by gradually increasing the priority "
            "of waiting processes over time."
        ),
        "partial": (
            "CPU scheduling determines which process gets to execute on the CPU. The OS uses algorithms like "
            "Round Robin which gives time slices to processes in a queue. Priority scheduling executes high-priority "
            "jobs first, but low-priority ones might wait a long time. Preemptive scheduling interrupts processes while "
            "non-preemptive lets them run until they finish."
        ),
        "weak": (
            "The OS manages processes with a scheduler. It decides who gets the CPU. Some algorithms are fair and some "
            "are not. The computer has to balance multiple programs running at the same time."
        ),
    },
    "oop_principles": {
        "strong": (
            "Object-Oriented Programming is built on four core principles:\n\n"
            "1. Encapsulation: Bundles attributes and methods into a class and restricts direct access to internal state "
            "using access modifiers (private, protected, public), exposing only a controlled public interface.\n\n"
            "2. Inheritance: Enables a derived child class to acquire attributes and methods of a parent base class, "
            "promoting code reuse. The child can override or extend inherited behaviour.\n\n"
            "3. Polymorphism: Allows objects of different classes to respond to the same method call in distinct ways. "
            "Method overriding provides runtime polymorphism, while overloading provides compile-time polymorphism.\n\n"
            "4. Abstraction: Hides complex implementation details and exposes only essential features via abstract classes "
            "and interfaces, allowing developers to program at a high level."
        ),
        "partial": (
            "OOP is based on classes and objects. Encapsulation means keeping data and methods together in a class. "
            "Inheritance allows a child class to use methods from a parent class without rewriting code. "
            "Polymorphism means the same method name can work differently in different classes. "
            "Abstraction hides details from the user."
        ),
        "weak": (
            "OOP uses objects and classes. A class is like a template for objects. You can create many objects from one class. "
            "Classes can be related to each other. Java and Python are OOP languages."
        ),
    },
}

# ─── Domain Knowledge Base ───────────────────────────────────────────────────
TOPICS = {
    "tcp_mechanics": {
        "key": "tcp_mechanics",
        "title": "TCP Mechanics in Computer Networks",
        "domain": "Computer Networks · Transport Layer",
        "icon": "🌐",
        "description": (
            "Explain the core TCP mechanisms that ensure reliable, ordered, "
            "and flow-controlled data delivery over an IP network."
        ),
        "expected_concepts": [
            {
                "id": "C1",
                "name": "Reliability and Sequence Numbers",
                "description": (
                    "TCP guarantees reliable delivery by assigning a unique sequence number "
                    "to every byte of data. The receiver uses these sequence numbers to "
                    "reorder out-of-order segments and detect missing data, ensuring the "
                    "application receives a complete and ordered byte stream."
                ),
                "keywords": ["sequence number", "reliability", "ordered delivery", "byte stream", "reorder", "segment"],
                "importance": "Essential"
            },
            {
                "id": "C2",
                "name": "Acknowledgments (ACK)",
                "description": (
                    "The receiver sends cumulative acknowledgment numbers back to the sender "
                    "to confirm which bytes have been successfully received. An ACK number "
                    "indicates the next expected byte, allowing the sender to track delivery "
                    "progress and know what has been confirmed."
                ),
                "keywords": ["acknowledgment", "ACK", "cumulative ack", "confirm", "received", "next expected byte"],
                "importance": "Essential"
            },
            {
                "id": "C3",
                "name": "Retransmission on Timeout or Packet Loss",
                "description": (
                    "If the sender does not receive an acknowledgment within a calculated "
                    "retransmission timeout (RTO) period, it assumes the segment was lost and "
                    "retransmits it. TCP also uses fast retransmit triggered by three duplicate "
                    "ACKs to recover from loss quickly without waiting for a timeout."
                ),
                "keywords": ["retransmission", "timeout", "RTO", "packet loss", "duplicate ACK", "fast retransmit", "timer"],
                "importance": "Essential"
            },
            {
                "id": "C4",
                "name": "Flow Control and Windowing",
                "description": (
                    "TCP flow control uses a sliding window mechanism where the receiver "
                    "advertises its available buffer space (receive window) to the sender. "
                    "The sender must not transmit more unacknowledged data than this advertised "
                    "window size, preventing the sender from overwhelming the receiver buffer."
                ),
                "keywords": ["flow control", "window", "sliding window", "receive window", "buffer", "rwnd", "in-flight"],
                "importance": "Essential"
            }
        ],
        "sample_answers": SAMPLE_ANSWERS["tcp_mechanics"]
    },
    "os_scheduling": {
        "key": "os_scheduling",
        "title": "CPU Scheduling in Operating Systems",
        "domain": "Operating Systems · Kernel Architecture",
        "icon": "⚙️",
        "description": (
            "Describe the key CPU scheduling algorithms and concepts that the OS uses "
            "to manage process execution on the processor."
        ),
        "expected_concepts": [
            {
                "id": "C1",
                "name": "Scheduling Criteria (CPU Utilization & Throughput)",
                "description": (
                    "CPU scheduling aims to maximize CPU utilization by keeping the processor "
                    "busy as much as possible, and to maximize throughput by completing as many "
                    "processes per unit time as possible."
                ),
                "keywords": ["CPU utilization", "throughput", "scheduling criteria", "maximize", "performance"],
                "importance": "Essential"
            },
            {
                "id": "C2",
                "name": "Preemptive vs Non-Preemptive Scheduling",
                "description": (
                    "In non-preemptive scheduling a running process holds the CPU until it "
                    "terminates or blocks. Preemptive scheduling allows the OS to forcibly "
                    "remove a process from the CPU (e.g., on a timer interrupt or higher-priority "
                    "arrival), enabling better responsiveness."
                ),
                "keywords": ["preemptive", "non-preemptive", "context switch", "interrupt", "priority", "yield"],
                "importance": "Essential"
            },
            {
                "id": "C3",
                "name": "Round Robin and Time Quantum",
                "description": (
                    "Round Robin scheduling assigns a fixed time slice (quantum) to each ready "
                    "process in circular order. When the quantum expires the process is preempted "
                    "and placed at the end of the ready queue, ensuring fairness."
                ),
                "keywords": ["round robin", "time quantum", "time slice", "circular", "fairness", "queue", "preempted"],
                "importance": "Essential"
            },
            {
                "id": "C4",
                "name": "Priority Scheduling and Starvation",
                "description": (
                    "Priority scheduling runs the highest-priority process first. Low-priority "
                    "processes may wait indefinitely (starvation) if high-priority processes "
                    "keep arriving. Aging — gradually increasing the priority of waiting processes "
                    "— is the standard solution to starvation."
                ),
                "keywords": ["priority", "starvation", "aging", "highest priority", "indefinite wait", "solution"],
                "importance": "Essential"
            }
        ],
        "sample_answers": SAMPLE_ANSWERS["os_scheduling"]
    },
    "oop_principles": {
        "key": "oop_principles",
        "title": "Object-Oriented Programming (OOP) Principles",
        "domain": "Software Engineering · Paradigm & Design",
        "icon": "🧱",
        "description": (
            "Explain the four fundamental principles of Object-Oriented Programming "
            "and how they contribute to software design."
        ),
        "expected_concepts": [
            {
                "id": "C1",
                "name": "Encapsulation",
                "description": (
                    "Encapsulation bundles data (attributes) and the methods that operate on "
                    "that data into a single unit called a class, and restricts direct access "
                    "to internal state through access modifiers (private, protected, public), "
                    "exposing only a controlled public interface."
                ),
                "keywords": ["encapsulation", "class", "private", "public", "data hiding", "access modifier", "attributes"],
                "importance": "Essential"
            },
            {
                "id": "C2",
                "name": "Inheritance",
                "description": (
                    "Inheritance allows a child (derived) class to acquire the attributes and "
                    "methods of a parent (base) class, promoting code reuse. The child class "
                    "can extend or override inherited behaviour to specialise the parent's "
                    "functionality."
                ),
                "keywords": ["inheritance", "parent class", "child class", "base", "derived", "extends", "override", "code reuse"],
                "importance": "Essential"
            },
            {
                "id": "C3",
                "name": "Polymorphism",
                "description": (
                    "Polymorphism lets objects of different classes be treated as objects of "
                    "a common superclass. Method overriding (runtime polymorphism) and method "
                    "overloading (compile-time polymorphism) allow the same interface to behave "
                    "differently depending on the actual object."
                ),
                "keywords": ["polymorphism", "overriding", "overloading", "runtime", "compile-time", "interface", "method"],
                "importance": "Essential"
            },
            {
                "id": "C4",
                "name": "Abstraction",
                "description": (
                    "Abstraction hides complex implementation details and exposes only the "
                    "essential features of an object. In OOP this is achieved through abstract "
                    "classes and interfaces, allowing programmers to work at a high level without "
                    "worrying about low-level implementation specifics."
                ),
                "keywords": ["abstraction", "abstract class", "interface", "hide", "implementation", "essential features"],
                "importance": "Essential"
            }
        ],
        "sample_answers": SAMPLE_ANSWERS["oop_principles"]
    }
}

DEFAULT_TOPIC = "tcp_mechanics"

REMEDIATION_PROMPT_TEMPLATE = """
You are an expert Computer Science tutor. A student's answer was evaluated and 
the concept "{concept_name}" was classified as "{status}" (not fully understood).

Topic: {topic_title}
Concept Description: {concept_description}
Student's Evidence: {student_evidence}

Generate a structured remediation response in EXACTLY this JSON format:
{{
  "explanation": "A clear, simple 2-sentence explanation of the concept that addresses what the student missed.",
  "example": "A concrete, specific Computer Science example that demonstrates the concept in action.",
  "drill_question": "One targeted practice question that tests understanding of this specific concept.",
  "answer_key": "The complete correct answer to the drill question above."
}}

Rules:
- Keep the explanation simple, direct and under 60 words.
- Make the example specific and technical (include real protocols, algorithms, or code context).
- The drill question must be answerable in 2-3 sentences.
- Return ONLY valid JSON, no markdown, no extra text.
"""
