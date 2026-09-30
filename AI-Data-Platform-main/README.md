# 🤖 Agentic AI - Multi-Agent Data & Analytics Platform

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.2+-purple.svg)](https://langchain-ai.github.io/langgraph/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.37+-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An enterprise-grade, multi-agent AI system for intelligent PostgreSQL querying, autonomous ETL data pipelines, sandboxed execution, and real-time visualization built with **LangGraph**, **PostgreSQL**, **Streamlit**, **sqlglot**, and **Docker**.

---

## 📋 Table of Contents
- [Architecture & Workflow](#-architecture--workflow)
- [Key Features & Production Upgrades](#-key-features--production-upgrades)
- [Project Structure](#-project-structure)
- [Prerequisites & Environment Configuration](#-prerequisites--environment-configuration)
- [Quickstart: Local & Docker Deployment](#-quickstart-local--docker-deployment)
- [Security & Sandboxing Deep Dive](#-security--sandboxing-deep-dive)
- [Full-Stack Observability](#-full-stack-observability)
- [Testing & Verification](#-testing--verification)

---

## 🏗️ Architecture & Workflow

The platform implements a **hierarchical multi-agent state machine** built on LangGraph:

```
                                  ┌─────────────────────────────┐
                                  │    Data Agent (Router)      │
                                  │   Routes SQL vs. ETL Query  │
                                  └──────────────┬──────────────┘
                                                 │
                        ┌────────────────────────┴────────────────────────┐
                        ▼                                                 ▼
             ┌─────────────────────┐                           ┌─────────────────────┐
             │  SQL Analyst Agent  │                           │  ETL Analyst Agent  │
             └──────────┬──────────┘                           └──────────┬──────────┘
                        │                                                 │
         ├─► 1. Question Curation                              ├─► 1. Intent & Tool Selection
         ├─► 2. Schema Metadata Injection                      ├─► 2. API Extraction (REST)
         ├─► 3. SQL Query Generation                           ├─► 3. Dynamic Pandas Code Gen
         ├─► 4. Dual AST + LLM Judge Safety Check              ├─► 4. Docker / AST Sandboxed Exec
         ├─► 5. PostgreSQL Query Execution                     └─► 5. Multi-Format Load (CSV/Parquet)
         │      └─► [Error Trace] ──► Self-Healing (3 Retries)
         ├─► 6. Final Human-Readable Summary
         └─► 7. Automated Plotly Visualization
```

---

## ✨ Key Features & Production Upgrades

### 1. 🛡️ Dual-Layer SQL Safety Guardrails
- **Deterministic AST Parsing (`sqlglot`)**: Mathematically validates the SQL abstract syntax tree to guarantee single, read-only `SELECT` queries.
- **Strict Mutation Blocking**: Blocks 8+ DDL/DML mutation keywords (`DROP`, `DELETE`, `UPDATE`, `INSERT`, `ALTER`, `TRUNCATE`, `CREATE`, `GRANT`, `REVOKE`) and dangerous system functions (`pg_sleep`, `pg_read_file`).
- **LLM-as-Judge Node**: Secondary structured verification using Pydantic `JudgeSchema`.

### 2. 🔄 Self-Healing / Auto-Repair Loop
- Built with a closed-loop LangGraph cycle. If PostgreSQL returns an execution or schema error, the error traceback is dynamically routed back into the LLM context to self-correct syntax, column names, or table joins (up to 3 automated retries).

### 3. 🐳 Docker Resource & AST Execution Sandboxing
- **Container Isolation**: Dedicated `Dockerfile.sandbox` executes generated Pandas transformations under an unprivileged user (`UID 1000`).
- **Resource Constraints**: `--memory=512m`, `--cpus=1.0`, `--pids-limit=64`, and `--network=none`.
- **Execution Timeout**: 15-second subprocess termination to kill accidental infinite loops.
- **In-Process Fallback**: AST security analyzer (`validate_python_ast`) blocks dangerous imports (`os`, `sys`, `subprocess`, `shutil`) and unsafe builtins.

### 4. 📊 Full-Stack Streamlit Dashboard
- **Live Agent Tracing**: Real-time `st.status` indicators tracking router decisions, safety checks, and execution nodes.
- **Automated Plotly Charts**: Automatically generates interactive visualizations for analytical queries.
- **Interactive Tables & CSV Export**: In-browser dataframe rendering and 1-click CSV downloads.

### 5. 🔭 Unified LLM Observability
- Integrated with **LangSmith**, **Langfuse**, and **Pydantic Logfire** for token usage, latency tracing, and OpenTelemetry instrumentation.

---

## 📁 Project Structure

```
├── agents/
│   ├── data_agent.py               # Main router agent with observability callbacks
│   ├── sql_analyst.py              # Self-healing SQL agent with AST validation & retries
│   └── etl_analyst.py              # ReAct ETL agent with tool calling
├── Models/
│   └── schema.py                   # Pydantic state schemas (Logfire instrumented)
├── utils/
│   ├── database.py                 # PostgreSQL connection & lifecycle manager
│   ├── etl_tools.py                # Safe ETL tools with Docker sandboxing & AST checks
│   ├── sql_validator.py            # Deterministic SQL AST parser (sqlglot)
│   ├── observability.py            # Unified LangSmith, Langfuse & Logfire manager
│   └── llm_pick.py                 # Dynamic multi-tier LLM selector
├── data/                           # Data storage for CSV, JSON & Parquet exports
├── tests/
│   ├── test_improvements.py        # Safety & sandboxing unit tests
│   └── test_observability.py       # Telemetry integration tests
├── app.py                          # Full-stack interactive Streamlit web dashboard
├── Dockerfile                      # Production application container
├── Dockerfile.sandbox              # Unprivileged isolated ETL execution sandbox
├── docker-compose.yml              # Multi-container Postgres + App orchestration
├── pyproject.toml                  # Project metadata & dependencies
└── requirements.txt                # Pip requirements
```

---

## ⚙️ Prerequisites & Environment Configuration

### 1. Requirements
- Python 3.12+
- PostgreSQL database
- (Optional) Docker & Docker Compose

### 2. Environment Setup (`.env`)
Create a `.env` file in the root directory (see `.env.example`):

```env
# LLM Providers
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key

# PostgreSQL Database
host=localhost
port=5432
user=postgres
password=your_password
database=postgres

# Observability (Optional)
LANGCHAIN_TRACING_V2=false
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_PROJECT=ai-data-agent

LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
LANGFUSE_SECRET_KEY=your_langfuse_secret_key
LANGFUSE_HOST=https://cloud.langfuse.com

LOGFIRE_TOKEN=your_logfire_token
```

---

## 🚀 Quickstart: Local & Docker Deployment

### Option A: Run Locally with Streamlit

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start the interactive web dashboard
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

### Option B: Run with Docker Compose (Recommended)

Spins up PostgreSQL and the AI Data Agent application in a single command:

```bash
docker-compose up --build
```

---

## 🔒 Security & Sandboxing Deep Dive

| Protection Layer | Technology | Enforced Rule |
|---|---|---|
| **SQL Safety** | `sqlglot` AST Parser | Blocks all DDL/DML, requires single read-only `SELECT` |
| **SQL Verification** | Pydantic `JudgeSchema` | LLM-as-Judge evaluation on query intent |
| **Python AST Sandbox** | Python `ast` module | Prohibits `os`, `sys`, `subprocess`, `shutil`, `eval`, `open` |
| **Docker Resource Quotas** | Docker cgroups | Caps memory at 512 MB, CPU at 1.0 core, PIDs at 64 |
| **Network Security** | Docker isolated network | Disables outbound networking (`--network=none`) during execution |
| **Subprocess Timeout** | Python `subprocess` | 15-second strict timeout on script execution |

---

## 🔭 Full-Stack Observability

- **LangSmith**: Enable `LANGCHAIN_TRACING_V2=true` in `.env` for graph execution visualizations.
- **Langfuse**: Set `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` for token cost and latency monitoring.
- **Pydantic Logfire**: Native OpenTelemetry instrumentation for all state schemas and validation flows.

---

## 🧪 Testing & Verification

Run the automated test suite to verify AST safety guardrails and sandboxing:

```bash
python -m unittest discover tests
```

---

## 📄 License
This project is licensed under the MIT License.
