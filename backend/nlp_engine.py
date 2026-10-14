"""
NLP Engine — Stages 3 & 4 of the 8-Stage Methodology
------------------------------------------------------
Stage 3 : NLP Preprocessing & Concept Extraction
Stage 4 : Semantic Analysis & Tri-State Classification (SBERT + Cosine Similarity)

Paper: "Beyond Marks: NLP-Based Learning Gap Detection and Remediation"
"""

import re
import logging
from functools import lru_cache
from typing import List, Dict, Tuple, Optional, Any

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from config import (
    SBERT_MODEL_NAME,
    SBERT_EMBEDDING_DIM,
    THRESHOLD_CORRECT,
    THRESHOLD_PARTIAL,
)

logger = logging.getLogger(__name__)

import tempfile
import os

# ─── NLTK Resource Bootstrap ─────────────────────────────────────────────────
def bootstrap_nltk():
    """Download required NLTK resources safely with fallback for serverless environments."""
    tmp_nltk = os.path.join(tempfile.gettempdir(), "nltk_data")
    if tmp_nltk not in nltk.data.path:
        nltk.data.path.append(tmp_nltk)

    resources = [
        ("tokenizers/punkt",          "punkt"),
        ("tokenizers/punkt_tab",      "punkt_tab"),
        ("corpora/stopwords",         "stopwords"),
    ]
    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            logger.info("Downloading NLTK resource: %s", name)
            try:
                nltk.download(name, quiet=True)
            except Exception:
                try:
                    nltk.download(name, download_dir=tmp_nltk, quiet=True)
                except Exception as e:
                    logger.warning("Could not download NLTK resource %s: %s", name, e)

bootstrap_nltk()

try:
    STOP_WORDS = set(stopwords.words("english"))
except Exception:
    STOP_WORDS = {
        "the", "a", "an", "is", "are", "was", "were", "in", "it", "of", "to",
        "and", "that", "this", "for", "with", "as", "on", "by", "at", "from",
        "be", "or", "which", "so", "but", "not", "have", "has", "had", "can"
    }

# Preserve important technical and domain keywords
DOMAIN_TERMS = {
    "tcp", "ack", "syn", "fin", "rto", "rwnd", "mss", "ip", "cpu",
    "os", "oop", "api", "http", "udp", "dns", "buffer", "class", "base"
}
STOP_WORDS -= DOMAIN_TERMS

# ─── SBERT Model (Singleton) ─────────────────────────────────────────────────
_sbert_model: Optional[SentenceTransformer] = None

def get_sbert_model() -> SentenceTransformer:
    global _sbert_model
    if _sbert_model is None:
        logger.info("Loading SBERT model: %s", SBERT_MODEL_NAME)
        _sbert_model = SentenceTransformer(SBERT_MODEL_NAME)
        logger.info("SBERT model loaded successfully.")
    return _sbert_model


# ─── Stage 3: NLP Preprocessing & Concept Extraction ────────────────────────
def preprocess_text(raw_text: str, topic: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Stage 3: Cleans, tokenizes, removes stopwords, extracts noun phrases,
    and identifies concept keyword alignments against domain concepts.
    """
    if not raw_text or not raw_text.strip():
        return {
            "cleaned": "",
            "tokens": [],
            "raw_token_count": 0,
            "filtered_token_count": 0,
            "stopwords_removed": [],
            "stopwords_removed_count": 0,
            "sentences": [],
            "noun_phrases": [],
            "concept_matches": [],
            "stats": {
                "word_count": 0,
                "sentence_count": 0,
                "unique_tokens": 0,
                "lexical_density_pct": 0,
            }
        }

    # 1. Lowercase
    text = raw_text.lower()

    # 2. Segment sentences on raw text
    sentences = [s.strip() for s in sent_tokenize(raw_text) if s.strip()]

    # 3. Clean characters (preserve alphanumerics, hyphens)
    text_clean = re.sub(r"[^a-z0-9\s\-]", " ", text)
    text_clean = re.sub(r"\s+", " ", text_clean).strip()

    # 4. Tokenize
    tokens_raw = [t for t in word_tokenize(text_clean) if len(t) > 0]
    raw_token_count = len(tokens_raw)

    # 5. Filter stopwords
    tokens_filtered = []
    stopwords_removed = []
    for t in tokens_raw:
        if t in STOP_WORDS:
            stopwords_removed.append(t)
        elif len(t) > 1 and not re.match(r"^[-]+$", t):
            tokens_filtered.append(t)

    # 6. Extract noun phrases and technical terms
    noun_phrases = _extract_noun_phrases(tokens_filtered)

    # 7. Concept keyword alignment (if topic provided)
    concept_matches = []
    text_clean_with_spaces = f" {text_clean} "
    if topic and "expected_concepts" in topic:
        for c in topic["expected_concepts"]:
            c_keywords = c.get("keywords", [])
            matched_kws = []
            for kw in c_keywords:
                kw_clean = kw.lower().strip()
                if kw_clean in text_clean_with_spaces or kw_clean in text.lower():
                    matched_kws.append(kw)
            concept_matches.append({
                "concept_id": c["id"],
                "concept_name": c["name"],
                "matched_keywords": matched_kws,
                "match_count": len(matched_kws),
                "detected": len(matched_kws) > 0,
            })

    # 8. Compute Lexical Stats
    word_count = len(raw_text.strip().split())
    unique_tokens = len(set(tokens_filtered))
    lexical_density = round((len(tokens_filtered) / raw_token_count * 100), 1) if raw_token_count > 0 else 0

    return {
        "cleaned": text_clean,
        "tokens": tokens_filtered,
        "raw_token_count": raw_token_count,
        "filtered_token_count": len(tokens_filtered),
        "stopwords_removed": stopwords_removed[:40],
        "stopwords_removed_count": len(stopwords_removed),
        "sentences": sentences,
        "noun_phrases": noun_phrases[:25],
        "concept_matches": concept_matches,
        "stats": {
            "word_count": word_count,
            "sentence_count": len(sentences),
            "unique_tokens": unique_tokens,
            "lexical_density_pct": lexical_density,
        }
    }


def _extract_noun_phrases(tokens: List[str]) -> List[str]:
    """
    Extracts high-value multi-word phrases and unigrams.
    """
    seen = set()
    phrases = []

    # Bigrams
    for i in range(len(tokens) - 1):
        bigram = f"{tokens[i]} {tokens[i+1]}"
        if bigram not in seen:
            seen.add(bigram)
            phrases.append(bigram)

    # Trigrams
    for i in range(len(tokens) - 2):
        trigram = f"{tokens[i]} {tokens[i+1]} {tokens[i+2]}"
        if trigram not in seen:
            seen.add(trigram)
            phrases.append(trigram)

    # Notable unigrams (technical terms length >= 3)
    for t in tokens:
        if len(t) >= 3 and t not in seen:
            seen.add(t)
            phrases.append(t)

    return phrases


# ─── Stage 2: Single Concept SBERT Similarity Testing ────────────────────────
def test_single_concept(topic: Dict, concept_id: str, student_fragment: str) -> Dict[str, Any]:
    """
    Tests cosine similarity of a student fragment against one specific concept.
    Used for interactive Stage 2 Concept Inspector.
    """
    concept = next((c for c in topic.get("expected_concepts", []) if c["id"] == concept_id), None)
    if not concept:
        raise ValueError(f"Concept '{concept_id}' not found in topic '{topic.get('title')}'.")

    if not student_fragment or not student_fragment.strip():
        raise ValueError("Student fragment cannot be empty.")

    model = get_sbert_model()

    concept_embedding = model.encode([concept["description"]])
    clean_frag = re.sub(r"[^a-z0-9\s\-]", " ", student_fragment.lower()).strip()
    fragment_embedding = model.encode([clean_frag if clean_frag else student_fragment])

    sim = float(cosine_similarity(concept_embedding, fragment_embedding)[0][0])
    sim = max(0.0, min(1.0, sim))  # clamp

    if sim >= THRESHOLD_CORRECT:
        classification = "Correct"
        color = "green"
    elif sim >= THRESHOLD_PARTIAL:
        classification = "Partial"
        color = "yellow"
    else:
        classification = "Missing"
        color = "red"

    return {
        "concept_id": concept["id"],
        "concept_name": concept["name"],
        "concept_description": concept["description"],
        "keywords": concept.get("keywords", []),
        "tested_text": student_fragment,
        "similarity": round(sim, 4),
        "similarity_pct": round(sim * 100, 1),
        "classification": classification,
        "color": color,
        "embedding_model": SBERT_MODEL_NAME,
        "embedding_dim": SBERT_EMBEDDING_DIM,
    }


# ─── Stage 4: Semantic Analysis & Tri-State Classification ───────────────────
def classify_concept(
    concept: Dict,
    student_preprocessed: Dict,
    student_raw: str,
) -> Dict[str, Any]:
    """
    Classifies a single expected concept against the student's answer.
    Classification rules (cosine similarity):
      ≥ THRESHOLD_CORRECT (0.70)  → Correct   (green)
      ≥ THRESHOLD_PARTIAL (0.40)  → Partial   (yellow)
      < THRESHOLD_PARTIAL (0.40)  → Missing   (red)
    """
    model = get_sbert_model()

    # Encode reference concept description
    concept_embedding = model.encode([concept["description"]])

    # Encode student's full cleaned text for overall similarity
    student_embedding = model.encode([student_preprocessed["cleaned"]])
    overall_sim = float(cosine_similarity(concept_embedding, student_embedding)[0][0])

    # Check sentence-level similarity and use the maximum
    sentence_sims = []
    for sent in student_preprocessed["sentences"]:
        sent_clean = re.sub(r"[^a-z0-9\s\-]", " ", sent.lower()).strip()
        if sent_clean:
            sent_emb = model.encode([sent_clean])
            sim = float(cosine_similarity(concept_embedding, sent_emb)[0][0])
            sentence_sims.append((sim, sent))

    max_sentence_sim = max((s for s, _ in sentence_sims), default=0.0)
    best_sentence = next(
        (sent for s, sent in sentence_sims if s == max_sentence_sim), ""
    )

    # Final similarity = max(overall, best sentence)
    final_sim = max(overall_sim, max_sentence_sim)
    final_sim = max(0.0, min(1.0, final_sim))

    # Classify
    if final_sim >= THRESHOLD_CORRECT:
        status = "Correct"
        recommended_action = "✅ Strong understanding demonstrated. Ready to advance."
    elif final_sim >= THRESHOLD_PARTIAL:
        status = "Partial"
        recommended_action = "⚠️ Partial grasp. Review targeted explanation & practice drill."
    else:
        status = "Missing"
        recommended_action = "❌ Concept not demonstrated. Foundational remediation required."

    evidence = best_sentence if max_sentence_sim >= 0.25 else "Not mentioned in student response."

    return {
        "concept_id":         concept["id"],
        "concept_name":       concept["name"],
        "similarity":         round(final_sim, 4),
        "similarity_pct":     round(final_sim * 100, 1),
        "status":             status,
        "evidence":           evidence,
        "recommended_action": recommended_action,
    }


def analyze_answer(topic: Dict, student_answer: str) -> Dict[str, Any]:
    """
    Full Stage 3 + Stage 4 pipeline.
    Returns complete analysis result ready for the dashboard.
    """
    if not student_answer or not student_answer.strip():
        raise ValueError("Student answer cannot be empty.")

    # Stage 3 — Enhanced Preprocessing
    preprocessed = preprocess_text(student_answer, topic=topic)

    # Stage 4 — Classify each expected concept
    results = []
    for concept in topic["expected_concepts"]:
        result = classify_concept(concept, preprocessed, student_answer)
        results.append(result)

    # Overall mastery percentage
    total = len(results)
    correct_count = sum(1 for r in results if r["status"] == "Correct")
    partial_count = sum(1 for r in results if r["status"] == "Partial")
    missing_count = sum(1 for r in results if r["status"] == "Missing")

    # Weighted mastery: Correct=1.0, Partial=0.5, Missing=0.0
    mastery_score = (correct_count + 0.5 * partial_count) / total if total else 0
    mastery_pct = round(mastery_score * 100, 1)

    # Determine gaps (Partial + Missing)
    gaps = [r for r in results if r["status"] in ("Partial", "Missing")]

    return {
        "topic_key": topic.get("key", ""),
        "topic_title": topic["title"],
        "domain": topic.get("domain", "Computer Science"),
        "student_answer": student_answer,
        "preprocessed": preprocessed,
        "concept_results": results,
        "summary": {
            "total": total,
            "correct_count": correct_count,
            "partial_count": partial_count,
            "missing_count": missing_count,
            "mastery_pct": mastery_pct,
            "gaps_count": len(gaps),
        },
        "gaps": gaps,
    }
