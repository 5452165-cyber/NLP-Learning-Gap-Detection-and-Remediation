"""
Flask Application — Main Entry Point
--------------------------------------
Implements the REST API for the "Beyond Marks" NLP Learning Gap Detection System.

8-Stage Methodology Endpoints:
  GET  /api/health        → Health & model readiness check
  GET  /api/topics        → Stage 1: Return available topics with concepts & samples
  GET  /api/topic/<key>   → Stage 1: Return full topic details
  POST /api/topics        → Stage 1: Add a custom topic
  POST /api/test-concept  → Stage 2: SBERT similarity testing for a single reference concept
  POST /api/preprocess    → Stage 3: NLP text cleaning, tokenization, stopwords, noun phrases
  POST /api/analyze       → Stages 4-6: Semantic analysis, cosine similarity, tri-state classification
  POST /api/remediate     → Stage 7: Personalized remediation generation via LLM or fallbacks
"""

import logging
import sys
import os
import warnings

# Suppress deprecation warnings from google packages (functional on Python 3.10)
warnings.filterwarnings("ignore", category=FutureWarning, module="google")

# ── Fix import path so backend modules are found ──────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, jsonify, request, render_template

from config import TOPICS, DEFAULT_TOPIC, SBERT_MODEL_NAME, SBERT_EMBEDDING_DIM
from nlp_engine import analyze_answer, preprocess_text, test_single_concept, get_sbert_model
from remediation_engine import generate_all_remediations

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ─── Flask App Setup ──────────────────────────────────────────────────────────
app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__), "..", "frontend", "templates"),
    static_folder=os.path.join(os.path.dirname(__file__), "..", "frontend", "static"),
    static_url_path="/static",
)

# Manual CORS headers
@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response

@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_frontend(path):
    """Serve the single-page frontend app."""
    return render_template("index.html")


# ─── API Routes ───────────────────────────────────────────────────────────────

@app.route("/api/health", methods=["GET"])
def health_check():
    """Simple liveness and readiness probe."""
    return jsonify({
        "status": "ok",
        "message": "Beyond Marks API is running.",
        "sbert_model": SBERT_MODEL_NAME,
        "embedding_dim": SBERT_EMBEDDING_DIM,
        "topics_available": len(TOPICS)
    })


# ── Stage 1: Topics & Curriculum Knowledge ────────────────────────────────────

@app.route("/api/topics", methods=["GET", "POST"])
def handle_topics():
    """
    GET: Return all topics with full concept definitions & sample answers.
    POST: Create a new custom topic.
    """
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        key = data.get("key", "").strip().lower().replace(" ", "_")
        title = data.get("title", "").strip()
        description = data.get("description", "").strip()
        concepts = data.get("concepts", [])

        if not key or not title or not concepts:
            return jsonify({"error": "Fields 'key', 'title', and 'concepts' are required."}), 400

        formatted_concepts = []
        for idx, c in enumerate(concepts):
            formatted_concepts.append({
                "id": c.get("id", f"C{idx+1}"),
                "name": c.get("name", f"Concept {idx+1}"),
                "description": c.get("description", ""),
                "keywords": c.get("keywords", []),
                "importance": c.get("importance", "Essential"),
            })

        TOPICS[key] = {
            "key": key,
            "title": title,
            "domain": data.get("domain", "Custom Domain"),
            "icon": data.get("icon", "📚"),
            "description": description,
            "expected_concepts": formatted_concepts,
            "sample_answers": data.get("sample_answers", {
                "strong": "Sample comprehensive answer covering all concepts.",
                "partial": "Sample partial answer missing some details.",
                "weak": "Sample introductory answer."
            })
        }
        logger.info("Custom topic registered: %s", key)
        return jsonify({"message": f"Topic '{title}' registered successfully.", "topic_key": key}), 201

    # GET response: Return full data
    return jsonify({
        "topics": TOPICS,
        "default_topic": DEFAULT_TOPIC,
        "sbert_model": SBERT_MODEL_NAME,
    })


@app.route("/api/topic/<topic_key>", methods=["GET"])
def get_topic(topic_key):
    """Return single topic with full concept breakdown."""
    topic = TOPICS.get(topic_key)
    if not topic:
        return jsonify({"error": f"Topic '{topic_key}' not found."}), 404
    return jsonify(topic)


# ── Stage 2: Single Concept Reference Inspection & Testing ───────────────────

@app.route("/api/test-concept", methods=["POST", "OPTIONS"])
def test_concept_endpoint():
    """
    Stage 2: Test a student fragment directly against a single reference concept.
    Request body (JSON):
      {
        "topic_key": "tcp_mechanics",
        "concept_id": "C1",
        "test_text": "TCP assigns sequence numbers to bytes to reorder them."
      }
    """
    if request.method == "OPTIONS":
        return jsonify({}), 200

    data = request.get_json(silent=True) or {}
    topic_key = data.get("topic_key", DEFAULT_TOPIC)
    concept_id = data.get("concept_id", "")
    test_text = data.get("test_text", "").strip()

    topic = TOPICS.get(topic_key)
    if not topic:
        return jsonify({"error": f"Topic '{topic_key}' not found."}), 404

    if not concept_id:
        return jsonify({"error": "concept_id is required."}), 400

    if not test_text:
        return jsonify({"error": "test_text is required."}), 400

    try:
        result = test_single_concept(topic, concept_id, test_text)
        return jsonify(result), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 422
    except Exception as e:
        logger.exception("Error in /api/test-concept")
        return jsonify({"error": "Internal server error testing concept."}), 500


# ── Stage 3: NLP Preprocessing & Concept Extraction ──────────────────────────

@app.route("/api/preprocess", methods=["POST", "OPTIONS"])
def preprocess_endpoint():
    """
    Stage 3: Tokenize, clean, remove stopwords, extract noun phrases & detect concept keywords.
    Request body (JSON):
      {
        "text": "TCP uses sequence numbers to ensure reliable data stream.",
        "topic_key": "tcp_mechanics"
      }
    """
    if request.method == "OPTIONS":
        return jsonify({}), 200

    data = request.get_json(silent=True) or {}
    text = data.get("text", "")
    topic_key = data.get("topic_key", DEFAULT_TOPIC)
    topic = TOPICS.get(topic_key)

    try:
        preprocessed = preprocess_text(text, topic=topic)
        return jsonify({
            "stage": "Stage 3: NLP Preprocessing & Concept Extraction",
            "topic_key": topic_key,
            **preprocessed
        }), 200
    except Exception as e:
        logger.exception("Error in /api/preprocess")
        return jsonify({"error": "Internal server error during preprocessing."}), 500


# ── Stages 4–6: Semantic Analysis & Tri-State Classification ─────────────────

@app.route("/api/analyze", methods=["POST", "OPTIONS"])
def analyze():
    """
    Stage 3 + 4 + 6: Preprocess student answer and classify concepts.
    Request body (JSON):
      {
        "topic_key":      "tcp_mechanics",
        "student_answer": "TCP uses sequence numbers to ..."
      }
    """
    if request.method == "OPTIONS":
        return jsonify({}), 200

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be JSON."}), 400

    topic_key      = data.get("topic_key", DEFAULT_TOPIC)
    student_answer = data.get("student_answer", "").strip()

    if not student_answer:
        return jsonify({"error": "student_answer is required and cannot be empty."}), 400

    topic = TOPICS.get(topic_key)
    if not topic:
        return jsonify({"error": f"Unknown topic: {topic_key}"}), 400

    try:
        result = analyze_answer(topic, student_answer)
        logger.info(
            "Analysis complete | topic=%s | mastery=%.1f%% | gaps=%d",
            topic_key, result["summary"]["mastery_pct"], result["summary"]["gaps_count"]
        )
        return jsonify(result), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 422
    except Exception as e:
        logger.exception("Unexpected error in /api/analyze")
        return jsonify({"error": "Internal server error during analysis."}), 500


# ── Stage 7: Automated Remediation Generation ─────────────────────────────────

@app.route("/api/remediate", methods=["POST", "OPTIONS"])
def remediate():
    """
    Stage 7: Generate personalized remediation for all gaps.
    Request body (JSON):
      {
        "topic_key": "tcp_mechanics",
        "gaps": [ { concept_id, concept_name, status, similarity, evidence }, ... ]
      }
    """
    if request.method == "OPTIONS":
        return jsonify({}), 200

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be JSON."}), 400

    topic_key = data.get("topic_key", DEFAULT_TOPIC)
    gaps      = data.get("gaps", [])

    topic = TOPICS.get(topic_key)
    if not topic:
        return jsonify({"error": f"Unknown topic: {topic_key}"}), 400

    if not gaps:
        return jsonify({"remediations": [], "message": "No gaps to remediate."}), 200

    try:
        remediations = generate_all_remediations(gaps, topic)
        logger.info(
            "Remediation complete | topic=%s | count=%d",
            topic_key, len(remediations)
        )
        return jsonify({"remediations": remediations}), 200
    except Exception as e:
        logger.exception("Unexpected error in /api/remediate")
        return jsonify({"error": "Internal server error during remediation."}), 500


# ─── Startup ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("  Beyond Marks — NLP Learning Gap Detection System")
    logger.info("  Pre-loading SBERT model (%s) ...", SBERT_MODEL_NAME)
    logger.info("=" * 60)

    # Eagerly load the SBERT model so the first request isn't slow
    get_sbert_model()

    port = int(os.getenv("PORT", 5000))
    logger.info("Starting Flask server on http://localhost:%d", port)
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
