# 🛡️ Compliance Monitor — OpenEnv Environment

<div align="center">

### Team RunTimers

**Gargi Monga · Anushka Pandey**

**Meta × Scaler OpenEnv Hackathon — Round 1**
**Deadline:** 8 April 2026, 11:59 PM IST

</div>

---

## 📋 Problem Statement

Companies store compliance rules in PDF documents that are rarely reviewed until violations occur. This leads to legal risk, audit failures, and operational inefficiencies.

This project builds an **AI-powered compliance monitoring system** that:

* Extracts structured rules from policy PDFs using LLMs
* Scans company records (employees, contracts, transactions)
* Detects policy violations automatically
* Explains violations in plain English
* Assigns severity levels (Low / Medium / High / Critical)
* Suggests actionable fixes
* Detects contradictions between policies
* Tracks compliance trends over time

**Target Impact:** Reduce legal risk and automate compliance monitoring for real-world organisations.

---

## 🚀 Overview

This is a **combined system** integrating two components:

| Component              | Owner          | Description                                  |
| ---------------------- | -------------- | -------------------------------------------- |
| 🌍 OpenEnv Environment | Gargi Monga    | Agent simulation (`reset`, `step`, `state`)  |
| 🧠 Data Pipeline       | Anushka Pandey | PDF → rules → DB → violations → explanations |

### 🔗 Integration

The components are connected via `merge_bridge.py`.

* OpenEnv fetches real data from:

  * `/openenv/records`
  * `/openenv/rules`
* Dummy data is automatically replaced with real database records
* No additional code changes required

---

## 🧩 System Architecture

```
HuggingFace Space
│
├── OpenEnv Server (server.py) — Port 7860
│   ├── /reset
│   ├── /step
│   ├── /state
│   └── /tasks
│
├── Data Pipeline (pipeline_server.py) — Port 7861
│   ├── /openenv/records
│   ├── /openenv/rules
│   ├── /scan
│   ├── /ingest/pdf
│   └── /violations
│
└── Shared
    ├── SQLite Database
    ├── PDF Ingestion Pipeline
    └── Trend Tracker
```

---

## ⚙️ OpenEnv Environment

### Action Space (6 Actions)

* `check_record`
* `flag_violation`
* `assign_severity`
* `generate_explanation`
* `suggest_fix`
* `resolve_conflict`

### Reward Function

```
R = detection(+0.4) + severity(+0.2) + explanation(+0.2) + fix(+0.2)
    - false_positive_penalty(-0.1)
    + conflict_resolution(+0.3)
```

### Observation Space

```json
{
  "records": [...],
  "rules": [...],
  "violations": [...],
  "conflicts": [...],
  "checked_record_ids": [...],
  "episode_step": 0,
  "max_steps": 60,
  "done": false,
  "total_reward": 0.0
}
```

---

## 📊 Tasks & Evaluation

| Task        | Difficulty | Description                          |
| ----------- | ---------- | ------------------------------------ |
| task_easy   | Easy       | Single record vs single rule         |
| task_medium | Medium     | Multi-record multi-rule              |
| task_hard   | Hard       | Full database + conflicting policies |

### Evaluation Criteria

* Real-world utility (30%)
* Task & grader quality (25%)
* Environment design (20%)
* Code quality & spec compliance (15%)
* Creativity & novelty (10%)

---

## 🧠 Data Pipeline

* PDF ingestion & rule extraction
* SQLite company database
* Rule-based violation detection
* LLM-powered explanations
* Severity scoring
* Fix suggestion engine
* Policy conflict detection
* Compliance trend tracking
* Real-time dashboard

---

## 📁 Project Structure

```
RunTimers/
│
├── server.py
├── pipeline_server.py
├── environment.py
├── merge_bridge.py
├── openenv.yaml
├── inference.py
├── validate.py
├── Dockerfile
├── requirements.txt
│
├── database/
├── pipeline/
└── static/
```

---

## 🔌 API Endpoints

### OpenEnv (Port 7860)

* POST `/reset`
* POST `/step`
* GET `/state`
* GET `/tasks`
* GET `/health`

### Data Pipeline (Port 7861)

* GET `/openenv/records`
* GET `/openenv/rules`
* POST `/scan`
* POST `/ingest/pdf`
* POST `/scan/conflicts`
* GET `/violations`
* GET `/trend`
* GET `/health`

---

## 🛠️ Setup & Run

### Local (Two Terminals)

**Terminal 1 — OpenEnv**

```bash
pip install -r requirements.txt
python server.py
```

**Terminal 2 — Data Pipeline**

```bash
python pipeline_server.py
```

---

### Docker

```bash
docker-compose up --build
```

---

## 🔑 Environment Variables

| Variable     | Purpose         |
| ------------ | --------------- |
| API_BASE_URL | LLM API         |
| MODEL_NAME   | Model           |
| HF_TOKEN     | API Key         |
| DB_PATH      | Database        |
| PORT         | Server port     |
| PIPELINE_URL | Integration URL |

---

## 🔗 Integration Code

```python
from merge_bridge import load_real_data, MERGE_AVAILABLE

if MERGE_AVAILABLE:
    records, rules = load_real_data(task_id)
```

---

## ✅ Pre-Submission Checklist

* OpenEnv spec valid (`openenv.yaml`)
* Docker builds successfully
* All endpoints working
* 3 tasks with deterministic graders
* `inference.py` runs correctly
* HF Space responds to `/reset`

---

## 🏆 Why This Stands Out

* Real-world enterprise use case
* Fully OpenEnv compliant
* Hybrid system (rule-based + LLM)
* Non-sparse reward design
* Multi-step reasoning tasks
* End-to-end working pipeline

---

<div align="center">

### Team RunTimers

**Gargi Monga · Anushka Pandey**

</div>
