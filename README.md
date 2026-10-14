# Beyond Marks — NLP Learning Gap Detection System
### Paper: "Beyond Marks: NLP-Based Learning Gap Detection and Remediation"

## Quick Start

### 1. Add your Gemini API Key
Edit `.env` and replace `your_gemini_api_key_here` with your key from https://aistudio.google.com/app/apikey

```
GEMINI_API_KEY=your_actual_key
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt -i https://pypi.org/simple/
```

### 3. Run the Server
```bash
python backend/app.py
```

### 4. Open the App
Navigate to: **http://localhost:5000**

---

## Project Structure
```
Data_final_project/
├── .env                        ← API keys & config
├── requirements.txt
├── backend/
│   ├── app.py                  ← Flask REST API (main entry point)
│   ├── config.py               ← Domain knowledge & thresholds
│   ├── nlp_engine.py           ← Stages 3 & 4: SBERT + classification
│   └── remediation_engine.py   ← Stage 5: LLM remediation generation
└── frontend/
    ├── templates/
    │   └── index.html          ← Single-page dashboard
    └── static/
        ├── css/styles.css      ← Full stylesheet
        └── js/app.js           ← Frontend logic & API calls
```

## 8-Stage Methodology (Section V)
| Stage | Description | Implementation |
|-------|-------------|----------------|
| 1 | Topic & Domain Selection | `config.py` TOPICS dict |
| 2 | Expected Concept Definition | Pre-loaded in config |
| 3 | NLP Preprocessing | `nlp_engine.py::preprocess_text()` |
| 4 | SBERT Semantic Analysis | `nlp_engine.py::classify_concept()` |
| 5 | Gap Mapping & Remediation | `remediation_engine.py` |
| 6 | Visual Dashboard | Mastery gauge & progress bars |
| 7 | Diagnostic Table | Table I from paper |
| 8 | Remediation Cards | LLM-generated explanations |

## Classification Thresholds
- ✅ **Correct**: Cosine similarity ≥ 0.70
- ⚠️ **Partial**: Cosine similarity 0.40–0.69  
- ❌ **Missing**: Cosine similarity < 0.40

## Topics Available
1. 🌐 TCP Mechanics in Computer Networks
2. ⚙️ CPU Scheduling in Operating Systems
3. 🧱 Object-Oriented Programming Principles
