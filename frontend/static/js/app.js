/**
 * Beyond Marks — Frontend JavaScript
 * NLP-Based Learning Gap Detection & Remediation
 * Fully connects Stages 1, 2, and 3 directly to backend REST API.
 */

"use strict";

const API_BASE = window.location.origin;

// ── State ─────────────────────────────────────────────────────────────────────
let state = {
  topics:          {},
  selectedTopic:   "tcp_mechanics",
  activeConcept:   null,       // for Stage 2 inspector
  analysisResult:  null,
  remediations:    [],
  preprocessing:   null,
};

// ── DOM References ─────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);

const dom = {
  backendStatusBadge:   $("backendStatusBadge"),

  // Stage 1
  topicGrid:            $("topicGrid"),

  // Stage 2
  conceptTags:          $("conceptTags"),
  conceptsCount:        $("conceptsCount"),

  // Stage 3
  topicDescription:     $("topicDescription"),
  topicMismatchAlert:   $("topicMismatchAlert"),
  mismatchMsg:          $("mismatchMsg"),
  btnFixMismatch:       $("btnFixMismatch"),
  studentAnswer:        $("studentAnswer"),
  wordCount:            $("wordCount"),
  charCount:            $("charCount"),
  btnRunStage3:         $("btnRunStage3"),
  btnSampleStrong:      $("btnSampleStrong"),
  btnSamplePartial:     $("btnSamplePartial"),
  btnSampleWeak:        $("btnSampleWeak"),
  btnSampleClear:       $("btnSampleClear"),
  btnAnalyze:           $("btnAnalyze"),

  // Stage 3 Output Box
  stage3OutputBox:      $("stage3OutputBox"),
  stage3StatsBadge:     $("stage3StatsBadge"),
  s3TokenChips:         $("s3TokenChips"),
  s3PhraseChips:        $("s3PhraseChips"),
  s3ConceptMatches:     $("s3ConceptMatches"),

  // Dashboard Stages 4-8
  dashboardPlaceholder: $("dashboardPlaceholder"),
  dashboardLoading:     $("dashboardLoading"),
  dashboardResults:     $("dashboardResults"),
  loadingMessage:       $("loadingMessage"),
  resultsTopicTitle:    $("resultsTopicTitle"),
  resultsDomainBadge:   $("resultsDomainBadge"),
  resultsSubtitle:      $("resultsSubtitle"),
  btnReanalyze:         $("btnReanalyze"),
  btnReset:             $("btnReset"),

  lstep1: $("lstep1"),
  lstep2: $("lstep2"),
  lstep3: $("lstep3"),
  lstep4: $("lstep4"),
  lstep5: $("lstep5"),

  gaugePct:             $("gaugePct"),
  gaugeArc:             $("gaugeArc"),
  gaugeNeedle:          $("gaugeNeedle"),
  statCorrect:          $("statCorrect"),
  statPartial:          $("statPartial"),
  statMissing:          $("statMissing"),
  statTotal:            $("statTotal"),
  masteryBarCorrect:    $("masteryBarCorrect"),
  masteryBarPartial:    $("masteryBarPartial"),
  masteryLabelCorrect:  $("masteryLabelCorrect"),
  masteryLabelPartial:  $("masteryLabelPartial"),
  masteryLabelMissing:  $("masteryLabelMissing"),

  diagTableBody:        $("diagTableBody"),
  remLoading:           $("remLoading"),
  remediationCards:     $("remediationCards"),

  // Stage 2 Modal
  conceptModalBackdrop: $("conceptModalBackdrop"),
  modalConceptTitle:    $("modalConceptTitle"),
  modalConceptDesc:     $("modalConceptDesc"),
  modalConceptKeywords: $("modalConceptKeywords"),
  sandboxInput:         $("sandboxInput"),
  btnSandboxTest:       $("btnSandboxTest"),
  sandboxResultBox:     $("sandboxResultBox"),
  sandboxSimScore:      $("sandboxSimScore"),
  sandboxStatusPill:    $("sandboxStatusPill"),
  btnModalClose:        $("btnModalClose"),
  btnModalOk:           $("btnModalOk"),
};

// ── Initialization ─────────────────────────────────────────────────────────────
async function init() {
  bindEvents();
  await checkHealth();
  await fetchTopics();
}

// ── Event Bindings ─────────────────────────────────────────────────────────────
function bindEvents() {
  dom.studentAnswer.addEventListener("input", onAnswerInput);
  dom.btnRunStage3.addEventListener("click", () => runStage3Preprocess());
  dom.btnAnalyze.addEventListener("click", runFullAnalysis);
  dom.btnReanalyze.addEventListener("click", runFullAnalysis);
  dom.btnReset.addEventListener("click", () => resetDashboard(false));

  dom.btnSampleStrong.addEventListener("click", () => loadSample("strong"));
  dom.btnSamplePartial.addEventListener("click", () => loadSample("partial"));
  dom.btnSampleWeak.addEventListener("click", () => loadSample("weak"));
  dom.btnSampleClear.addEventListener("click", () => {
    dom.studentAnswer.value = "";
    onAnswerInput();
    dom.stage3OutputBox.style.display = "none";
    dom.topicMismatchAlert.style.display = "none";
  });

  dom.btnFixMismatch.addEventListener("click", () => loadSample("strong"));

  // Stage 2 Modal
  dom.btnModalClose.addEventListener("click", closeModal);
  dom.btnModalOk.addEventListener("click", closeModal);
  dom.conceptModalBackdrop.addEventListener("click", e => {
    if (e.target === dom.conceptModalBackdrop) closeModal();
  });
  dom.btnSandboxTest.addEventListener("click", testSandboxConcept);
  dom.sandboxInput.addEventListener("keydown", e => {
    if (e.key === "Enter") testSandboxConcept();
  });
}

// ── Backend Health Probe ──────────────────────────────────────────────────────
async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/api/health`);
    const data = await res.json();
    if (data.status === "ok") {
      dom.backendStatusBadge.textContent = `🟢 Backend Live · SBERT ${data.sbert_model} (${data.embedding_dim}-d)`;
      dom.backendStatusBadge.style.background = "rgba(16,185,129,.25)";
      dom.backendStatusBadge.style.color = "#a7f3d0";
    }
  } catch (err) {
    dom.backendStatusBadge.textContent = "🔴 Backend Disconnected";
    dom.backendStatusBadge.style.background = "rgba(239,68,68,.3)";
    dom.backendStatusBadge.style.color = "#fecaca";
  }
}

// ── STAGE 1: Fetch & Render Topics (Connected to Backend) ─────────────────────
async function fetchTopics() {
  try {
    const res = await fetch(`${API_BASE}/api/topics`);
    const data = await res.json();
    state.topics = data.topics || {};
    state.selectedTopic = data.default_topic || Object.keys(state.topics)[0] || "tcp_mechanics";

    renderTopicGrid();
    selectTopic(state.selectedTopic, true); // initial load
  } catch (err) {
    console.error("Failed to fetch topics from backend:", err);
    showAlert("Failed to connect to backend topics API.", "error");
  }
}

function renderTopicGrid() {
  dom.topicGrid.innerHTML = "";
  Object.values(state.topics).forEach(topic => {
    const btn = document.createElement("button");
    btn.className = "topic-btn" + (topic.key === state.selectedTopic ? " active" : "");
    btn.dataset.key = topic.key;
    btn.innerHTML = `
      <span class="topic-icon">${topic.icon}</span>
      <div class="topic-info">
        <div class="topic-name">${topic.title}</div>
        <div class="topic-meta">
          <span class="topic-domain-pill">${topic.domain || "Computer Science"}</span>
          <span>${topic.expected_concepts ? topic.expected_concepts.length : 4} expected concepts</span>
        </div>
      </div>
    `;
    btn.addEventListener("click", () => selectTopic(topic.key));
    dom.topicGrid.appendChild(btn);
  });
}

async function selectTopic(key, isInitial = false) {
  state.selectedTopic = key;

  // Update UI buttons
  dom.topicGrid.querySelectorAll(".topic-btn").forEach(btn => {
    btn.classList.toggle("active", btn.dataset.key === key);
  });

  const topic = state.topics[key];
  if (!topic) return;

  // Stage 1 -> Stage 2: Render Expected Concepts
  renderConceptTags(topic);

  // Stage 1 -> Stage 3: Update Description & Prompt
  dom.topicDescription.innerHTML = `
    <strong>Objective:</strong> ${topic.description}
  `;

  // Auto-load Strong sample on topic switch so the student answer is never mismatched
  if (isInitial || !dom.studentAnswer.value.trim() || isTopicMismatched(dom.studentAnswer.value, key)) {
    loadSample("strong", false);
  } else {
    checkTopicMismatch();
  }

  // Run Stage 3 NLP Preprocessing
  await runStage3Preprocess();

  if (!isInitial) {
    resetDashboard(true);
  }
}

// ── STAGE 2: Render & Inspect Concepts (Connected to Backend) ────────────────
function renderConceptTags(topic) {
  dom.conceptsCount.textContent = `${topic.expected_concepts.length} concepts`;
  dom.conceptTags.innerHTML = "";

  const colors = ["#6366f1", "#8b5cf6", "#ec4899", "#f59e0b", "#10b981"];

  topic.expected_concepts.forEach((c, idx) => {
    const color = colors[idx % colors.length];
    const tag = document.createElement("div");
    tag.className = "concept-tag";
    tag.style.borderColor = color + "66";
    tag.style.animationDelay = `${idx * 0.06}s`;
    tag.innerHTML = `
      <span style="color:${color};font-weight:800">${c.id}</span>
      <span>${c.name}</span>
      <span style="font-size:.65rem;color:${color}">🔍</span>
    `;
    tag.title = "Click to inspect reference definition & test SBERT embedding";
    tag.addEventListener("click", () => openConceptModal(c));
    dom.conceptTags.appendChild(tag);
  });
}

function openConceptModal(concept) {
  state.activeConcept = concept;
  dom.modalConceptTitle.textContent = `[${concept.id}] ${concept.name}`;
  dom.modalConceptDesc.textContent  = concept.description;

  // Render keywords
  dom.modalConceptKeywords.innerHTML = "";
  (concept.keywords || []).forEach(kw => {
    const el = document.createElement("span");
    el.className = "modal-kw-tag";
    el.textContent = kw;
    dom.modalConceptKeywords.appendChild(el);
  });

  // Suggest a test sentence from current text or default
  const curAnswer = dom.studentAnswer.value.trim();
  const sentences = curAnswer.split(/[.!?]\s+/).filter(Boolean);
  const matchedSentence = sentences.find(s =>
    (concept.keywords || []).some(k => s.toLowerCase().includes(k.toLowerCase()))
  );

  dom.sandboxInput.value = matchedSentence || `TCP guarantees reliable delivery using ${concept.name.toLowerCase()}.`;
  dom.sandboxResultBox.style.display = "none";
  dom.conceptModalBackdrop.style.display = "flex";
}

function closeModal() {
  dom.conceptModalBackdrop.style.display = "none";
}

// Stage 2 Quick-Test: calls backend /api/test-concept
async function testSandboxConcept() {
  if (!state.activeConcept) return;

  const testText = dom.sandboxInput.value.trim();
  if (!testText) {
    showAlert("Please enter a sentence to test.", "warning");
    return;
  }

  dom.btnSandboxTest.disabled = true;
  dom.btnSandboxTest.textContent = "Computing…";

  try {
    const res = await fetch(`${API_BASE}/api/test-concept`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        topic_key:  state.selectedTopic,
        concept_id: state.activeConcept.id,
        test_text:  testText,
      }),
    });

    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Failed to test concept");

    dom.sandboxResultBox.style.display = "flex";
    dom.sandboxSimScore.textContent = `${data.similarity_pct}%`;

    const statusPill = dom.sandboxStatusPill;
    statusPill.className = `status-pill status-${data.classification.toLowerCase()}`;
    const icon = { Correct: "✅", Partial: "⚠️", Missing: "❌" }[data.classification] || "•";
    statusPill.textContent = `${icon} ${data.classification} (${data.color})`;

  } catch (err) {
    console.error("Sandbox test error:", err);
    showAlert(`Error: ${err.message}`, "error");
  } finally {
    dom.btnSandboxTest.disabled = false;
    dom.btnSandboxTest.textContent = "Test SBERT";
  }
}

// ── STAGE 3: Response Ingestion & Live Preprocessing (Backend Connected) ─────
function onAnswerInput() {
  const text = dom.studentAnswer.value;
  const words = text.trim().split(/\s+/).filter(Boolean).length;
  dom.wordCount.textContent = words;
  dom.charCount.textContent = text.length;

  checkTopicMismatch();
}

function checkTopicMismatch() {
  const text = dom.studentAnswer.value.toLowerCase();
  const curKey = state.selectedTopic;

  const isMismatch = isTopicMismatched(text, curKey);
  if (isMismatch) {
    dom.topicMismatchAlert.style.display = "flex";
    if (curKey === "tcp_mechanics" && (text.includes("oop") || text.includes("class") || text.includes("encapsulation"))) {
      dom.mismatchMsg.textContent = "Your text discusses Object-Oriented Programming (OOP), but TCP Mechanics is selected.";
    } else if (curKey === "os_scheduling" && (text.includes("tcp") || text.includes("packet") || text.includes("sequence number"))) {
      dom.mismatchMsg.textContent = "Your text discusses TCP Mechanics, but CPU Scheduling is selected.";
    } else {
      dom.mismatchMsg.textContent = "The current answer appears to discuss concepts from a different topic.";
    }
  } else {
    dom.topicMismatchAlert.style.display = "none";
  }
}

function isTopicMismatched(text, curKey) {
  const t = text.toLowerCase();
  if (curKey === "tcp_mechanics") {
    return t.includes("encapsulation") || t.includes("polymorphism") || (t.includes("class") && t.includes("inheritance"));
  }
  if (curKey === "os_scheduling") {
    return (t.includes("tcp") && t.includes("sequence")) || t.includes("retransmission timeout") || t.includes("encapsulation");
  }
  if (curKey === "oop_principles") {
    return (t.includes("tcp") && t.includes("acknowledgment")) || t.includes("round robin") || t.includes("quantum");
  }
  return false;
}

function loadSample(strength, autoPreprocess = true) {
  const topic = state.topics[state.selectedTopic];
  if (!topic || !topic.sample_answers) return;

  const sample = topic.sample_answers[strength];
  if (sample) {
    dom.studentAnswer.value = sample.trim();
    onAnswerInput();
    dom.topicMismatchAlert.style.display = "none";
    if (autoPreprocess) {
      runStage3Preprocess();
    }
  }
}

// Stage 3 Live Preprocessing: calls POST /api/preprocess
async function runStage3Preprocess() {
  const text = dom.studentAnswer.value.trim();
  if (!text) {
    dom.stage3OutputBox.style.display = "none";
    return;
  }

  try {
    const res = await fetch(`${API_BASE}/api/preprocess`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text:      text,
        topic_key: state.selectedTopic,
      }),
    });

    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Preprocessing failed");

    state.preprocessing = data;
    renderStage3Output(data);

  } catch (err) {
    console.error("Stage 3 preprocessing error:", err);
  }
}

function renderStage3Output(data) {
  dom.stage3OutputBox.style.display = "flex";

  const stats = data.stats || {};
  dom.stage3StatsBadge.textContent =
    `${stats.word_count} words · ${data.filtered_token_count} content tokens · ${data.stopwords_removed_count} stopwords removed · ${stats.lexical_density_pct}% density`;

  // 1. Filtered Tokens
  dom.s3TokenChips.innerHTML = "";
  (data.tokens || []).slice(0, 20).forEach(tok => {
    const span = document.createElement("span");
    span.className = "token-chip";
    span.textContent = tok;
    dom.s3TokenChips.appendChild(span);
  });
  if ((data.tokens || []).length > 20) {
    const more = document.createElement("span");
    more.className = "token-chip";
    more.textContent = `+${data.tokens.length - 20} more…`;
    dom.s3TokenChips.appendChild(more);
  }

  // 2. Extracted Noun Phrases
  dom.s3PhraseChips.innerHTML = "";
  const phrases = (data.noun_phrases || []).filter(p => p.includes(" "));
  phrases.slice(0, 8).forEach(phrase => {
    const span = document.createElement("span");
    span.className = "phrase-chip";
    span.textContent = phrase;
    dom.s3PhraseChips.appendChild(span);
  });

  // 3. Concept Keyword Matches
  dom.s3ConceptMatches.innerHTML = "";
  (data.concept_matches || []).forEach(cm => {
    const span = document.createElement("span");
    span.className = `match-chip ${cm.detected ? "match-chip-detected" : "match-chip-missing"}`;
    span.innerHTML = `
      <strong>${cm.concept_id}</strong> ${cm.detected ? "✓ " + cm.matched_keywords.slice(0, 2).join(", ") : "✕ Not detected"}
    `;
    dom.s3ConceptMatches.appendChild(span);
  });
}

// ── STAGES 4–8: Full Pipeline Execution (Connected to Backend) ────────────────
async function runFullAnalysis() {
  const answer = dom.studentAnswer.value.trim();
  if (!answer) {
    showAlert("Please enter or load a student answer before analyzing.", "warning");
    return;
  }
  if (answer.split(/\s+/).length < 8) {
    showAlert("Response is too brief. Please enter at least 8-10 words for accurate semantic evaluation.", "warning");
    return;
  }

  showLoading();
  dom.btnAnalyze.disabled = true;

  try {
    // Stage 3 + 4: Analyze
    await animateLoadingStep("lstep1", "Stage 3: Tokenizing, removing stopwords & extracting phrases…");
    await animateLoadingStep("lstep2", "Stage 4: Encoding reference & student dense embeddings via SBERT…");
    await animateLoadingStep("lstep3", "Stage 4: Calculating cosine similarity matrix for all concepts…");

    const analysisRes = await fetch(`${API_BASE}/api/analyze`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({
        topic_key:      state.selectedTopic,
        student_answer: answer,
      }),
    });

    if (!analysisRes.ok) {
      const err = await analysisRes.json();
      throw new Error(err.error || "Analysis failed");
    }

    state.analysisResult = await analysisRes.json();
    await animateLoadingStep("lstep4", "Stages 5–6: Tri-state classification & learning gap mapping…");

    // Stage 7: Remediation
    await animateLoadingStep("lstep5", "Stage 7: Generating personalized explanation, CS example & drill question…");

    const remRes = await fetch(`${API_BASE}/api/remediate`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({
        topic_key: state.selectedTopic,
        gaps:      state.analysisResult.gaps,
      }),
    });

    if (!remRes.ok) throw new Error("Remediation generation failed");

    const remData = await remRes.json();
    state.remediations = remData.remediations || [];

    // Stage 8: Visual Dashboard
    await sleep(350);
    renderDashboard();

  } catch (err) {
    console.error("Full analysis error:", err);
    showAlert(`Error: ${err.message}`, "error");
    resetDashboard(true);
  } finally {
    dom.btnAnalyze.disabled = false;
  }
}

// ── STAGE 8: Dashboard Rendering ──────────────────────────────────────────────
function renderDashboard() {
  const result = state.analysisResult;
  const topic  = state.topics[state.selectedTopic];

  dom.dashboardLoading.style.display = "none";
  dom.dashboardResults.style.display = "block";

  dom.resultsDomainBadge.textContent = topic.domain || "Computer Science";
  dom.resultsTopicTitle.textContent  = topic.title;
  dom.resultsSubtitle.textContent    =
    `${result.summary.total} concepts evaluated · ${result.summary.correct_count} Correct · ${result.summary.partial_count} Partial · ${result.summary.missing_count} Missing`;

  renderMastery(result.summary);
  renderDiagTable(result.concept_results);
  renderRemediationCards(state.remediations);

  // Scroll dashboard smoothly into view on smaller displays
  if (window.innerWidth < 1100) {
    dom.dashboardResults.scrollIntoView({ behavior: "smooth" });
  }
}

function renderMastery(summary) {
  const pct = summary.mastery_pct;

  dom.statCorrect.textContent = summary.correct_count;
  dom.statPartial.textContent = summary.partial_count;
  dom.statMissing.textContent = summary.missing_count;
  dom.statTotal.textContent   = summary.total;

  const arcLen = 251.2;
  const filled = arcLen * (pct / 100);
  dom.gaugeArc.style.strokeDashoffset = arcLen - filled;

  const angle = -90 + (pct / 100) * 180;
  const rad   = angle * (Math.PI / 180);
  const cx = 100, cy = 100, r = 68;
  const nx = cx + r * Math.sin(rad);
  const ny = cy - r * Math.cos(rad);
  dom.gaugeNeedle.setAttribute("x2", nx.toFixed(1));
  dom.gaugeNeedle.setAttribute("y2", ny.toFixed(1));

  const gColor = pct >= 70 ? "#10b981" : pct >= 40 ? "#f59e0b" : "#ef4444";
  dom.gaugePct.textContent = `${pct}%`;
  dom.gaugePct.setAttribute("fill", gColor);
  dom.gaugeArc.setAttribute("stroke", pct >= 70 ? "url(#gaugeGradient)" : (pct >= 40 ? "#f59e0b" : "#ef4444"));

  const total = summary.total;
  const correctPct = (summary.correct_count / total * 100).toFixed(1);
  const partialPct = (summary.partial_count / total * 100).toFixed(1);
  const missingPct = (summary.missing_count / total * 100).toFixed(1);

  requestAnimationFrame(() => {
    dom.masteryBarCorrect.style.width = correctPct + "%";
    dom.masteryBarPartial.style.width = partialPct + "%";
    dom.masteryLabelCorrect.textContent = `${correctPct}% Correct`;
    dom.masteryLabelPartial.textContent = `${partialPct}% Partial`;
    dom.masteryLabelMissing.textContent = `${missingPct}% Missing`;
  });
}

function renderDiagTable(conceptResults) {
  dom.diagTableBody.innerHTML = "";

  conceptResults.forEach(cr => {
    const statusCls = {
      Correct: "status-correct",
      Partial: "status-partial",
      Missing: "status-missing",
    }[cr.status] || "status-missing";

    const statusIcon = { Correct: "✅", Partial: "⚠️", Missing: "❌" }[cr.status];
    const simColor = cr.similarity >= 0.70 ? "#10b981" :
                     cr.similarity >= 0.40 ? "#f59e0b" : "#ef4444";
    const simPct   = Math.round(cr.similarity * 100);

    const row = document.createElement("tr");
    row.innerHTML = `
      <td>
        <span class="concept-id">${cr.concept_id}</span>
        <span class="concept-name-cell">${escHtml(cr.concept_name)}</span>
      </td>
      <td>
        <span class="evidence-text">${escHtml(cr.evidence)}</span>
      </td>
      <td>
        <div class="sim-bar-wrap">
          <div class="sim-bar">
            <div class="sim-bar-fill" style="width:${simPct}%;background:${simColor}"></div>
          </div>
          <span class="sim-value" style="color:${simColor}">${cr.similarity_pct}%</span>
        </div>
      </td>
      <td>
        <span class="status-pill ${statusCls}">${statusIcon} ${cr.status}</span>
      </td>
      <td style="font-size:.76rem;color:var(--color-text-muted)">${escHtml(cr.recommended_action)}</td>
    `;
    dom.diagTableBody.appendChild(row);
  });
}

function renderRemediationCards(remediations) {
  dom.remediationCards.innerHTML = "";

  if (!remediations || remediations.length === 0) {
    dom.remediationCards.innerHTML = `
      <div class="card" style="padding:22px;text-align:center;color:var(--color-correct)">
        <p style="font-size:1.6rem">🎉</p>
        <p style="font-weight:700;margin-top:6px">Outstanding! Full Concept Mastery Demonstrated.</p>
        <p style="font-size:.78rem;color:var(--color-text-muted);margin-top:4px">All expected concepts met the threshold (≥0.70). No gaps flagged.</p>
      </div>
    `;
    return;
  }

  remediations.forEach((rem, idx) => {
    const cardClass = rem.status === "Missing" ? "rem-card-missing" : "rem-card-partial";
    const statusPill = rem.status === "Missing"
      ? `<span class="status-pill status-missing">❌ Missing Gap</span>`
      : `<span class="status-pill status-partial">⚠️ Partial Gap</span>`;
    const sourceLabel = rem.source === "gemini"
      ? "✨ Personalized by Gemini AI"
      : "📚 Domain Knowledge Base Fallback";

    const card = document.createElement("div");
    card.className = `rem-card ${cardClass}`;
    card.style.animationDelay = `${idx * 0.1}s`;
    card.innerHTML = `
      <div class="rem-card-header">
        <span class="concept-id">${escHtml(rem.concept_id)}</span>
        <span class="rem-concept-name">${escHtml(rem.concept_name)}</span>
        ${statusPill}
        <span class="rem-similarity">sim: ${(rem.similarity * 100).toFixed(1)}%</span>
      </div>
      <div class="rem-card-body">

        <div>
          <div class="rem-section-title">💡 2-Sentence Targeted Explanation</div>
          <div class="rem-explanation">${escHtml(rem.explanation)}</div>
        </div>

        <div>
          <div class="rem-section-title">🔬 Concrete CS Example</div>
          <div class="rem-example">${escHtml(rem.example)}</div>
        </div>

        <div class="rem-drill">
          <div class="rem-section-title">🎯 Targeted Drill Question</div>
          <div class="drill-question">${escHtml(rem.drill_question)}</div>
          <button class="btn-show-answer" onclick="toggleAnswer(this)">Show Answer Key</button>
          <div class="drill-answer">${escHtml(rem.answer_key)}</div>
        </div>

        <div class="rem-source">${sourceLabel}</div>

      </div>
    `;
    dom.remediationCards.appendChild(card);
  });
}

// ── Loading & Utilities ───────────────────────────────────────────────────────
function showLoading() {
  dom.dashboardPlaceholder.style.display = "none";
  dom.dashboardResults.style.display     = "none";
  dom.dashboardLoading.style.display     = "block";

  ["lstep1","lstep2","lstep3","lstep4","lstep5"].forEach(id => {
    $(id).classList.remove("active","done");
    const labels = {
      lstep1: "Preprocessing text (Stage 3)",
      lstep2: "Generating SBERT embeddings (Stage 4)",
      lstep3: "Computing cosine similarity matrix",
      lstep4: "Tri-State classification & Gap mapping (Stages 5-6)",
      lstep5: "Generating personalized remediation (Stage 7)",
    };
    $(id).textContent = `⏳ ${labels[id]}`;
  });
}

async function animateLoadingStep(stepId, message) {
  const el = $(stepId);
  const stepIdx = ["lstep1","lstep2","lstep3","lstep4","lstep5"].indexOf(stepId);
  if (stepIdx > 0) {
    const prev = $(`lstep${stepIdx}`);
    prev.classList.remove("active");
    prev.classList.add("done");
    prev.textContent = "✅ " + prev.textContent.replace(/^[⏳🔄✅]\s*/, "");
  }
  el.classList.add("active");
  el.textContent = "🔄 " + message;
  dom.loadingMessage.textContent = message;
  await sleep(400);
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

function toggleAnswer(btn) {
  const answerEl = btn.nextElementSibling;
  const visible  = answerEl.style.display === "block";
  answerEl.style.display = visible ? "none" : "block";
  btn.textContent = visible ? "Show Answer Key" : "Hide Answer Key";
}

function resetDashboard(silent = false) {
  dom.dashboardResults.style.display     = "none";
  dom.dashboardLoading.style.display     = "none";
  dom.dashboardPlaceholder.style.display = "block";
  if (!silent) {
    dom.studentAnswer.value = "";
    onAnswerInput();
    dom.stage3OutputBox.style.display = "none";
    dom.topicMismatchAlert.style.display = "none";
  }
  state.analysisResult = null;
  state.remediations   = [];
}

function escHtml(str) {
  if (!str) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;")
    .replace(/\n/g, "<br/>");
}

function showAlert(message, type = "info") {
  const color = type === "error" ? "#ef4444" : type === "warning" ? "#f59e0b" : "#6366f1";
  const old   = document.getElementById("__alert__");
  if (old) old.remove();

  const el = document.createElement("div");
  el.id = "__alert__";
  el.style.cssText = `
    position:fixed;top:68px;right:20px;z-index:99999;
    background:#fff;border:2px solid ${color};border-radius:10px;
    padding:10px 16px;max-width:380px;box-shadow:0 8px 24px rgba(0,0,0,.15);
    font-size:.8rem;color:${color};font-weight:700;
    animation:slideUp .3s ease;
  `;
  el.textContent = message;
  document.body.appendChild(el);
  setTimeout(() => el.remove(), 4500);
}

// ── Boot ──────────────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", init);
