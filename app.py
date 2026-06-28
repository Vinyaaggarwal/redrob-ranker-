# -*- coding: utf-8 -*-
import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
import streamlit as st
import json
import numpy as np
import re
import io
import base64
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime
from sentence_transformers import SentenceTransformer

st.set_page_config(
    page_title="Redrob · Talent Intelligence Dashboard",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Initialize session state variables
if "scored_list" not in st.session_state:
    st.session_state["scored_list"] = []
if "shortlisted" not in st.session_state:
    st.session_state["shortlisted"] = []
if "comparison_pool" not in st.session_state:
    st.session_state["comparison_pool"] = []
if "view_resume" not in st.session_state:
    st.session_state["view_resume"] = None

# ── CSS Design System (Linear / Stripe inspired) ──────────────────────────────
CSS = """
<style>
/* Font import */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* Typography & Base Reset */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0F1A17 !important;
    font-family: 'Inter', sans-serif !important;
    color: #FFF4E1 !important;
}

/* Hide Streamlit chrome */
[data-testid="stHeader"] { display: none !important; }
[data-testid="stSidebar"] { display: none !important; }

/* Wider, more breathable page container */
.block-container {
    padding: 32px 48px !important;
    max-width: 1500px !important;
    margin: 0 auto !important;
}
[data-testid="stVerticalBlock"] { gap: 0 !important; }

/* ── Sticky Left Column ────────────────────────────── */
[data-testid="stAppViewContainer"] > section [data-testid="stVerticalBlock"] > div > [data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-of-type(1) {
    position: sticky !important;
    top: 24px !important;
    align-self: start !important;
    height: calc(100vh - 48px) !important;
    overflow-y: auto !important;
    padding-right: 10px !important;
}
[data-testid="stAppViewContainer"] > section [data-testid="stVerticalBlock"] > div > [data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-of-type(1)::-webkit-scrollbar { width: 4px; }
[data-testid="stAppViewContainer"] > section [data-testid="stVerticalBlock"] > div > [data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-of-type(1)::-webkit-scrollbar-thumb {
    background: #428475;
    border-radius: 4px;
}

/* ── Header Banner — 25% taller, more premium ─────── */
.header-banner {
    background: linear-gradient(135deg, #1A312C 0%, #2B5247 50%, #1A312C 100%);
    border: 1px solid #428475;
    border-radius: 16px;
    padding: 22px 28px;
    margin: 0 0 28px 0;
    box-shadow:
        0 4px 6px -1px rgba(0, 0, 0, 0.25),
        0 10px 30px -5px rgba(26, 49, 44, 0.5),
        inset 0 1px 0 rgba(137, 215, 183, 0.08);
    display: flex;
    align-items: center;
}
.banner-content {
    display: flex;
    align-items: center;
    gap: 18px;
}
.banner-logo {
    width: 44px;
    height: 44px;
    background: rgba(137, 215, 183, 0.12);
    border: 1px solid rgba(137, 215, 183, 0.3);
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #89D7B7;
    flex-shrink: 0;
}
/* Dashboard title — +7px, bolder */
.banner-title {
    font-size: 26px;
    font-weight: 800;
    color: #FFF4E1;
    letter-spacing: -0.03em;
    line-height: 1.15;
}
.banner-subtitle {
    font-size: 13px;
    color: #89D7B7;
    margin-top: 4px;
    font-weight: 500;
    letter-spacing: 0.01em;
}

/* ── Panel / Section Headings — +3px ──────────────── */
.panel-title {
    font-size: 13px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: #89D7B7;
    margin-bottom: 16px;
    padding-bottom: 8px;
    border-bottom: 1px solid rgba(66, 132, 117, 0.3);
}

/* ── Inputs & Form Controls ───────────────────────── */
.stTextArea textarea {
    background-color: #1A312C !important;
    border: 1px solid #428475 !important;
    border-radius: 10px !important;
    color: #FFF4E1 !important;
    font-size: 14px !important;
    padding: 12px 14px !important;
    line-height: 1.55 !important;
}
.stTextArea textarea:focus {
    border-color: #89D7B7 !important;
    box-shadow: 0 0 0 3px rgba(137, 215, 183, 0.15) !important;
}

/* Slider labels slightly larger */
.stSlider {
    margin-bottom: 18px !important;
}
[data-testid="stSlider"] label {
    font-size: 13px !important;
    font-weight: 500 !important;
    color: #A2C0B9 !important;
}

/* ── Buttons ───────────────────────────────────────── */
button[data-testid="baseButton-secondary"] {
    background-color: #1A312C !important;
    color: #FFF4E1 !important;
    border: 1px solid #428475 !important;
    border-radius: 10px !important;
    padding: 8px 14px !important;
    font-size: 13.5px !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.12) !important;
}
button[data-testid="baseButton-secondary"]:hover {
    border-color: #89D7B7 !important;
    color: #89D7B7 !important;
    background-color: rgba(137, 215, 183, 0.08) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(137, 215, 183, 0.15) !important;
}
button[data-testid="baseButton-secondary"]:active {
    transform: translateY(0) !important;
}

button[data-testid="baseButton-primary"] {
    background-color: #428475 !important;
    color: #FFF4E1 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 22px !important;
    font-size: 15px !important;
    font-weight: 700 !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
    letter-spacing: 0.01em;
}
button[data-testid="baseButton-primary"]:hover {
    background-color: #52A090 !important;
    box-shadow: 0 6px 16px rgba(66, 132, 117, 0.35) !important;
    transform: translateY(-1px) !important;
}

/* ── Candidate Card list ───────────────────────────── */
.candidate-list {
    display: flex;
    flex-direction: column;
    gap: 24px;
}

/* ── Candidate Card — deeper padding ──────────────── */
.candidate-card {
    background-color: #132521;
    border: 1px solid #2D5047;
    border-radius: 16px;
    padding: 26px 28px;
    transition: all 0.28s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    box-shadow:
        0 4px 6px -1px rgba(0, 0, 0, 0.15),
        0 2px 4px -1px rgba(0, 0, 0, 0.08);
}
.candidate-card:hover {
    border-color: #89D7B7;
    transform: translateY(-3px);
    box-shadow:
        0 16px 28px -6px rgba(0, 0, 0, 0.35),
        0 0 0 1px rgba(137, 215, 183, 0.2);
}
.candidate-card.shortlisted-card {
    border-left: 5px solid #89D7B7 !important;
    box-shadow: 0 4px 16px rgba(137, 215, 183, 0.18) !important;
}

/* ── Card Header ───────────────────────────────────── */
.card-top {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 16px;
}
/* Candidate name — +3px, bolder */
.candidate-name {
    font-size: 21px;
    font-weight: 700;
    color: #FFF4E1;
    margin-bottom: 4px;
    line-height: 1.2;
}
/* Candidate title/company — slightly larger */
.candidate-title {
    font-size: 14px;
    color: #A2C0B9;
    font-weight: 400;
}
.score-badge-container {
    display: flex;
    align-items: center;
    gap: 10px;
    flex-shrink: 0;
}
.confidence-badge {
    background-color: #1A312C;
    border: 1px solid #428475;
    color: #89D7B7;
    padding: 4px 10px;
    border-radius: 9999px;
    font-size: 11px;
    font-weight: 600;
}
.score-badge {
    font-size: 15px;
    font-weight: 800;
    padding: 5px 14px;
    border-radius: 8px;
    border: 1px solid transparent;
    letter-spacing: -0.01em;
}
.score-badge.high {
    background-color: rgba(137, 215, 183, 0.12);
    color: #89D7B7;
    border-color: rgba(137, 215, 183, 0.35);
}
.score-badge.mid {
    background-color: rgba(245, 158, 11, 0.12);
    color: #FBBF24;
    border-color: rgba(245, 158, 11, 0.3);
}
.score-badge.low {
    background-color: rgba(162, 192, 185, 0.1);
    color: #A2C0B9;
    border-color: rgba(162, 192, 185, 0.25);
}

/* ── Progress Bar ──────────────────────────────────── */
.progress-container { margin-bottom: 20px; }
.progress-labels {
    display: flex;
    justify-content: space-between;
    font-size: 12px;
    color: #A2C0B9;
    margin-bottom: 6px;
    font-weight: 500;
}
.progress-bar-bg {
    height: 6px;
    background-color: #1A312C;
    border-radius: 9999px;
    overflow: hidden;
}
.progress-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, #428475, #89D7B7);
    border-radius: 9999px;
}

/* ── Metadata Grid ─────────────────────────────────── */
.meta-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 14px;
    background-color: #1A312C;
    padding: 14px 18px;
    border-radius: 10px;
    border: 1px solid #2D5047;
    margin-bottom: 20px;
}
.meta-item {
    display: flex;
    flex-direction: column;
    gap: 3px;
}
/* Meta labels — +1px */
.meta-label {
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #6D9E94;
}
/* Meta values — +1px */
.meta-value {
    font-size: 13px;
    font-weight: 600;
    color: #FFF4E1;
}

/* ── Skill Chips ───────────────────────────────────── */
.skill-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 20px;
}
.skill-chip {
    font-size: 12px;
    font-weight: 500;
    padding: 5px 12px;
    border-radius: 9999px;
    background-color: #1A312C;
    border: 1px solid #2D5047;
    color: #A2C0B9;
    transition: all 0.15s ease;
}
.skill-chip:hover {
    border-color: #89D7B7;
    color: #FFF4E1;
}
.skill-chip.core, .skill-chip.matched-core {
    background-color: rgba(137, 215, 183, 0.1);
    border-color: rgba(137, 215, 183, 0.45);
    color: #89D7B7;
}
.skill-chip.core:hover, .skill-chip.matched-core:hover {
    background-color: rgba(137, 215, 183, 0.2);
    border-color: #89D7B7;
}
.skill-chip.preferred, .skill-chip.matched-pref {
    background-color: rgba(245, 158, 11, 0.1);
    border-color: rgba(245, 158, 11, 0.4);
    color: #FBBF24;
}
.skill-chip.preferred:hover, .skill-chip.matched-pref:hover {
    background-color: rgba(245, 158, 11, 0.2);
    border-color: #FBBF24;
}
.skill-chip.other {
    background-color: #1A312C;
    border: 1px solid #2D5047;
    color: #A2C0B9;
}
.skill-chip.missing {
    background-color: rgba(239, 68, 68, 0.04);
    border-color: rgba(239, 68, 68, 0.2);
    color: #FCA5A5;
    text-decoration: line-through;
    opacity: 0.8;
}

/* ── Details / Expansion panel ─────────────────────── */
details.card-details {
    border-top: 1px solid #2D5047;
    margin-top: 16px;
    padding-top: 14px;
}
details.card-details summary {
    list-style: none;
    font-size: 13px;
    font-weight: 600;
    color: #89D7B7;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: flex-end;
    gap: 4px;
    outline: none;
    user-select: none;
}
details.card-details summary::-webkit-details-marker { display: none; }
.details-body {
    display: flex;
    gap: 24px;
    margin-top: 16px;
}
.details-left {
    flex: 1.4;
    display: flex;
    flex-direction: column;
    gap: 14px;
}
.details-right {
    flex: 0.8;
    display: flex;
    justify-content: center;
    align-items: center;
    border-left: 1px solid #2D5047;
    padding-left: 20px;
}
.insight-section {
    display: flex;
    flex-direction: column;
    gap: 5px;
}
/* Insight headers — +1px */
.insight-header {
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #6D9E94;
    margin-bottom: 3px;
}
/* Insight rows — +1px */
.insight-row {
    font-size: 13px;
    color: #FFF4E1;
    line-height: 1.5;
    display: flex;
    gap: 7px;
}
.insight-bullet { color: #89D7B7; }
.summary-text {
    font-size: 13px;
    color: #A2C0B9;
    line-height: 1.6;
}

/* ── Empty States ──────────────────────────────────── */
.empty-illustration {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 340px;
    border: 2px dashed #2D5047;
    border-radius: 14px;
    color: #6D9E94;
    text-align: center;
    padding: 24px;
}
.empty-headline {
    font-size: 17px;
    font-weight: 700;
    color: #A2C0B9;
    margin-top: 14px;
    margin-bottom: 6px;
}
.empty-desc {
    font-size: 14px;
    color: #6D9E94;
    max-width: 340px;
    line-height: 1.55;
}

/* ── Results Header Bar ────────────────────────────── */
.res-header-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 20px;
}
.res-count-badge {
    background-color: #1A312C;
    border: 1px solid #428475;
    color: #89D7B7;
    font-size: 12px;
    font-weight: 600;
    padding: 4px 12px;
    border-radius: 9999px;
}

/* ── Dashboard Metrics Row ─────────────────────────── */
.metrics-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-bottom: 28px;
}
.metric-card {
    background-color: #132521;
    border: 1px solid #2D5047;
    border-radius: 12px;
    padding: 16px 18px;
    text-align: center;
    transition: all 0.22s ease-in-out;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
}
.metric-card:hover {
    border-color: #89D7B7;
    transform: translateY(-3px);
    box-shadow: 0 8px 20px rgba(137, 215, 183, 0.15);
}
/* Metric label — slightly larger */
.metric-label {
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #6D9E94;
}
/* Metric value — slightly larger */
.metric-value {
    font-size: 22px;
    font-weight: 800;
    color: #FFF4E1;
    margin-top: 6px;
    letter-spacing: -0.02em;
}

/* ── Resume Viewer Drawer ──────────────────────────── */
.resume-modal {
    background-color: #1A312C;
    border: 1px solid #428475;
    border-radius: 16px;
    padding: 24px 28px;
    margin-bottom: 24px;
    box-shadow: 0 12px 30px -6px rgba(0, 0, 0, 0.45);
}
.resume-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 20px;
    border-bottom: 1px solid #2D5047;
    padding-bottom: 14px;
}
.resume-title {
    font-size: 22px;
    font-weight: 700;
    color: #FFF4E1;
}
.resume-subtitle {
    font-size: 14.5px;
    color: #A2C0B9;
}
.resume-section-title {
    font-size: 12.5px;
    font-weight: 700;
    color: #89D7B7;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-top: 20px;
    margin-bottom: 12px;
    border-bottom: 1px solid #428475;
    padding-bottom: 4px;
}
.timeline-container {
    display: flex;
    flex-direction: column;
    gap: 14px;
}
.timeline-item {
    position: relative;
    border-left: 2px solid #428475;
    padding-left: 16px;
    padding-bottom: 8px;
}
.timeline-item:last-child {
    border-left: none;
    padding-bottom: 0;
}
.timeline-dot {
    width: 9px;
    height: 9px;
    background-color: #89D7B7;
    border-radius: 50%;
    position: absolute;
    left: -5.5px;
    top: 4px;
}
.timeline-header {
    font-size: 14.5px;
    font-weight: 600;
    color: #FFF4E1;
}
.timeline-sub {
    font-size: 12px;
    color: #A2C0B9;
    margin-bottom: 4px;
}
.timeline-desc {
    font-size: 13px;
    color: #C8DDD8;
    line-height: 1.5;
}

/* Signals Grid */
.signals-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
}
.signal-card {
    background-color: #132521;
    border: 1px solid #428475;
    border-radius: 8px;
    padding: 10px 12px;
}
.signal-label {
    font-size: 10.5px;
    color: #A2C0B9;
    text-transform: uppercase;
    font-weight: 600;
    letter-spacing: 0.04em;
}
.signal-value {
    font-size: 14px;
    font-weight: 700;
    color: #FFF4E1;
    margin-top: 3px;
}

/* ── Comparison Table ──────────────────────────────── */
.comparison-table {
    width: 100%;
    border-collapse: collapse;
    background-color: #1A312C;
    border: 1px solid #428475;
    border-radius: 12px;
    overflow: hidden;
    margin-bottom: 20px;
}
.comparison-table th, .comparison-table td {
    padding: 12px 14px;
    text-align: left;
    border-bottom: 1px solid #2D5047;
    font-size: 13px;
}
.comparison-table th {
    background-color: #132521;
    color: #89D7B7;
    font-size: 11px;
    text-transform: uppercase;
    font-weight: 700;
    letter-spacing: 0.05em;
}
.comparison-table td.label-col {
    font-weight: 600;
    color: #A2C0B9;
    background-color: rgba(26, 49, 44, 0.45);
}
.comparison-table td.val-col {
    color: #FFF4E1;
}

/* ── Side Panel (comparison / resume wrappers) ─────── */
.side-panel {
    background-color: #132521;
    border: 1px solid #2D5047;
    border-radius: 14px;
    padding: 22px 24px;
    margin-bottom: 24px;
    box-shadow: 0 4px 18px rgba(0, 0, 0, 0.2);
}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

# ── Load Assets ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_assets():
    if not os.path.exists("candidate_embeddings.npz") or not os.path.exists("candidate_ids.json"):
        return None
    try:
        emb = np.load("candidate_embeddings.npz")["embeddings"]
        with open("candidate_ids.json") as f:
            ids = json.load(f)
        honey = set()
        if os.path.exists("honeypots.json"):
            with open("honeypots.json") as f:
                honey = set(json.load(f))
        model = SentenceTransformer("all-MiniLM-L6-v2")
        return emb, ids, honey, model
    except Exception as e:
        return None

@st.cache_data(show_spinner=False)
def load_non_tech_ids():
    s = set()
    if not os.path.exists("candidates.jsonl"): return s
    with open("candidates.jsonl","r",encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            cm = re.search(r'"candidate_id":\s*"(CAND_\d{7})"', line)
            tm = re.search(r'"current_title":\s*"([^"]*)"', line)
            if cm and tm:
                title_lower = tm.group(1).lower()
                non_tech_keywords = [
                    "civil engineer","mechanical engineer","electrical engineer","structural engineer",
                    "chemical engineer","aerospace engineer","automobile engineer","industrial engineer",
                    "hr manager","human resources","hr executive","talent acquisition","recruiter",
                    "graphic designer","ui designer","fashion designer","interior designer",
                    "sales executive","sales manager","account manager","business development",
                    "operations manager","operations executive","supply chain","logistics",
                    "marketing manager","marketing executive","brand manager","content writer",
                    "project manager","program manager","finance manager","accountant",
                    "chartered accountant","ca ","customer support","customer success",
                    "customer service","teacher","professor","lecturer","business analyst",
                    "qa engineer","quality assurance","test engineer","software tester",
                ]
                if any(nt in title_lower for nt in non_tech_keywords):
                    s.add(cm.group(1))
    return s

def load_jd():
    if os.path.exists("job_description.txt"):
        with open("job_description.txt","r",encoding="utf-8") as f:
            return f.read()
    return ""

assets = load_assets()

# ── Dynamic skill lookup and constants ────────────────────────────────────────
SERVICE_FIRMS = [
    "tcs","tata consultancy","infosys","wipro","accenture",
    "cognizant","capgemini","tech mahindra","hcl","l&t","lnt",
    "mindtree","mphasis"
]
CORE_SKILLS_MAPPING = {
    "embeddings": ["embeddings","dense retrieval","rag","search engine","retrieval","semantic search","indexing","vector search"],
    "vector_databases": ["vector db","vector database","faiss","pinecone","milvus","qdrant","weaviate","chroma","elasticsearch","opensearch"],
    "hybrid_search": ["hybrid search","bm25","sparse search","keyword search"],
    "python": ["python","py","django","flask","fastapi"],
    "evaluation_frameworks": ["ndcg","mrr","map","evaluation","recall","precision","benchmark"]
}
PREFERRED_SKILLS_MAPPING = {
    "llm_finetuning": ["fine-tuning","lora","qlora","peft","sft","rlhf","transformers","huggingface","llama","mistral"],
    "learning_to_rank": ["learning to rank","ltr","xgboost","lightgbm","catboost","neural ranking"],
    "distributed_systems": ["distributed","spark","ray","kubernetes","docker"],
    "hr_tech": ["hr-tech","recruitment","marketplace","talent acquisition"],
    "open_source": ["open-source","open source","github","contributions"]
}

# ── Feature Scoring Functions (matching rank.py exactly) ──────────────────────
def is_service_firm(company_name):
    return any(f in company_name.lower() for f in SERVICE_FIRMS)

def is_non_tech_profile(cand):
    history = cand.get("career_history", [])
    if not history: return True
    GENUINE = [
        "software engineer","ml engineer","ai engineer","machine learning",
        "data engineer","data scientist","nlp engineer","backend engineer",
        "frontend engineer","fullstack","devops engineer","sre","platform engineer",
        "infrastructure engineer","cloud engineer","research engineer","applied scientist",
        "search engineer","recommendation","computer vision","deep learning engineer",
        "developer","programmer","software developer","android engineer","ios engineer",
    ]
    tech_count = sum(
        1 for j in history
        if any(kw in j.get("title","").lower() for kw in GENUINE)
    )
    return tech_count == 0

def evaluate_skills(cand):
    skills = cand.get("skills", [])
    if not skills: return 0.0
    total_dur = sum(s.get("duration_months", 0) for s in skills)
    avg_dur = total_dur / len(skills)
    stuffer_pen = 0.5 if len(skills) > 20 and avg_dur < 12 else 1.0
    
    def cat_score_fn(mapping):
        scores = []
        for cat, syns in mapping.items():
            best = 0.0
            for s in skills:
                sn = s.get("name","").lower()
                if any(syn in sn for syn in syns):
                    pv = {"beginner":0.4,"intermediate":0.7,"advanced":0.9,"expert":1.0}.get(s.get("proficiency","beginner"),0.4)
                    dur = s.get("duration_months",0)
                    dm = 1.0 if dur>=24 else (0.5 if dur<6 else 0.5+0.5*(dur-6)/18)
                    em = 1.0 + min(0.3, np.log1p(s.get("endorsements",0))/10)
                    best = max(best, pv*dm*em)
            scores.append(min(1.0, best))
        return float(np.mean(scores)) if scores else 0.0
    core_avg = cat_score_fn(CORE_SKILLS_MAPPING)
    pref_avg = cat_score_fn(PREFERRED_SKILLS_MAPPING)
    return float((0.7*core_avg + 0.3*pref_avg) * stuffer_pen)

def evaluate_experience(cand):
    profile = cand.get("profile", {})
    history = cand.get("career_history", [])
    if is_non_tech_profile(cand): return 0.02
    yrs = profile.get("years_of_experience", 0)
    sf = 1.0 if 5<=yrs<=9 else (0.8 if yrs in (4,10) else (0.6 if yrs in (3,11) else (0.4 if yrs in (2,12) else 0.1)))
    ai_kw = ["ml","machine learning","deep learning","nlp","embeddings","retrieval","vector","llm","transformers","pytorch","tensorflow","ai engineer","data scientist","search","rag"]
    ai_months = sum(j.get("duration_months",0) for j in history if any(kw in j.get("title","").lower() or kw in j.get("description","").lower() for kw in ai_kw))
    ai_yrs = ai_months/12
    ai_s = 1.0 if ai_yrs>=4 else (0.8 if ai_yrs>=3 else (0.5 if ai_yrs>=2 else (0.2 if ai_yrs>=1 else 0.0)))
    return float(0.5*sf + 0.5*ai_s)

def evaluate_project_relevance(cand):
    history = cand.get("career_history", [])
    if not history: return 0.0
    kws = ["ranking","retrieval","vector search","recommendation system","search engine","rag","information retrieval","hybrid search"]
    rel = sum(j.get("duration_months",0) for j in history if any(kw in j.get("description","").lower() or kw in j.get("title","").lower() for kw in kws))
    return min(1.0, rel/48.0)

def evaluate_velocity_and_stability(cand):
    history = cand.get("career_history", [])
    if not history: return 0.5
    if all(j.get("company") and is_service_firm(j.get("company","")) for j in history): return 0.05
    total = sum(j.get("duration_months",0) for j in history)
    avg = total/len(history)
    stab = 1.0 if avg>=36 else (0.1 if avg<12 else (0.4 if avg<18 else 0.4+0.6*(avg-18)/18))
    senior = any(any(sk in j.get("title","").lower() for sk in ["senior","lead","staff","principal","head","director","architect"]) for j in history)
    junior = any(any(jk in j.get("title","").lower() for jk in ["junior","associate","intern","trainee","fresher"]) for j in history)
    growth = 1.0 if junior and senior else (0.8 if senior else 0.6)
    return float(0.5*stab + 0.5*growth)

def evaluate_behavior(cand):
    sig = cand.get("redrob_signals", {})
    otw = 1.0 if sig.get("open_to_work_flag") else 0.8
    rr = sig.get("recruiter_response_rate", 0.5)
    rt = sig.get("avg_response_time_hours", 24)
    tm = 1.0 if rt<=24 else (0.5 if rt>=72 else 0.5+0.5*(72-rt)/48)
    eng = rr*tm
    comp = sig.get("interview_completion_rate", 0.5)
    return float(0.4*otw + 0.3*eng + 0.3*comp)

# ── Radar Chart Base64 Encoder ────────────────────────────────────────────────
def make_radar_base64(scores):
    labels = ['Semantic','Skills','Exp','Projects','Velocity','Behavior']
    vals   = [scores.get(k,0) for k in ['semantic','skills','experience','projects','velocity','behavior']]
    N = len(labels)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    vals   = vals + vals[:1]
    angles = angles + angles[:1]

    fig, ax = plt.subplots(figsize=(1.8, 1.8), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor('#1A312C')
    ax.set_facecolor('#1A312C')

    ax.plot(angles, vals, color='#89D7B7', linewidth=1.5)
    ax.fill(angles, vals, color='#89D7B7', alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, color="#FFF4E1", size=6, fontfamily="sans-serif")
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels([])
    ax.set_ylim(0, 1)
    ax.spines['polar'].set_color('#428475')
    ax.grid(color='#428475', linewidth=0.5)

    for spine in ax.spines.values():
        spine.set_color('#428475')

    plt.tight_layout(pad=0)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', facecolor='#1A312C')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

# ── Header Banner ─────────────────────────────────────────────────────────────
st.markdown("""<div class="header-banner">
<div class="banner-content">
<div class="banner-logo">
<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
<path d="M12 2a10 10 0 0 0-10 10v2a10 10 0 0 0 10 10h1v-4a4 4 0 0 0-8 0v-2a6 6 0 0 1 12 0v2a4 4 0 0 0-8 0v4h1a10 10 0 0 0 10-10v-2a10 10 0 0 0-10-10z"/>
<circle cx="9" cy="12" r="1"/>
<circle cx="15" cy="12" r="1"/>
</svg>
</div>
<div>
<div class="banner-title">Redrob Talent Intelligence Dashboard</div>
<div class="banner-subtitle">Semantic Ranking Engine & Sandbox Demo</div>
</div>
</div>
</div>""", unsafe_allow_html=True)

# Extra breathing room between header and content
st.markdown('<div style="height: 8px"></div>', unsafe_allow_html=True)

# Main columns — gap injected via CSS override below
left_col, right_col = st.columns([32, 68], gap="large")

# Inject a wider explicit gap via CSS targeting the column wrapper
st.markdown("""
<style>
/* Increase the gap between the two main columns to ~40px */
[data-testid="stAppViewContainer"] > section [data-testid="stVerticalBlock"] > div > [data-testid="stHorizontalBlock"] {
    gap: 40px !important;
    column-gap: 40px !important;
}
</style>
""", unsafe_allow_html=True)


with left_col:
    st.markdown('<div class="panel-title">Job Requirements</div>', unsafe_allow_html=True)
    jd_val = load_jd()
    jd_input = st.text_area("Job Description Details", value=jd_val, height=280, label_visibility="collapsed")
    
    st.markdown('<div style="height: 36px"></div>', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Hiring Score Weighting</div>', unsafe_allow_html=True)
    
    w_sem  = st.slider("Semantic Relevancy", 0.0, 1.0, 0.30, 0.05)
    w_sk   = st.slider("Skill Assessment Match", 0.0, 1.0, 0.20, 0.05)
    w_exp  = st.slider("Applied AI/ML Years", 0.0, 1.0, 0.20, 0.05)
    w_pr   = st.slider("Domain Projects", 0.0, 1.0, 0.10, 0.05)
    w_vel  = st.slider("Job Stability / Career Growth", 0.0, 1.0, 0.10, 0.05)
    w_beh  = st.slider("Response Rate / Engagement", 0.0, 1.0, 0.10, 0.05)
    
    total_w = w_sem + w_sk + w_exp + w_pr + w_vel + w_beh
    if total_w == 0:
        st.stop()
        
    weights = {
        "semantic": w_sem / total_w,
        "skills": w_sk / total_w,
        "experience": w_exp / total_w,
        "projects": w_pr / total_w,
        "velocity": w_vel / total_w,
        "behavior": w_beh / total_w,
    }
    
    st.markdown('<div style="height: 8px"></div>', unsafe_allow_html=True)
    search_btn = st.button("Score & Rank Candidates")

with right_col:
    if assets is None:
        st.markdown("""
        <div class="empty-illustration">
          <div style="font-size: 36px;">&#9881;</div>
          <div class="empty-headline">Missing Embeddings File</div>
          <div class="empty-desc">Please run <code>python precompute_embeddings.py</code> first to prepare the candidate dataset.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        candidate_embeddings, candidate_ids, honeypots, model = assets
        non_tech_ids = load_non_tech_ids()
        
        # Determine if search should trigger
        should_search = search_btn or ("scored_list" in st.session_state)
        
        if search_btn:
            with st.spinner("Processing candidate matching model..."):
                # Clean prompt embedding
                jd_emb = model.encode(jd_input, convert_to_numpy=True).astype(np.float32)
                ce = candidate_embeddings.astype(np.float32)
                jd_norm = jd_emb / np.linalg.norm(jd_emb)
                
                ce_norms = np.linalg.norm(ce, axis=1, keepdims=True)
                ce_norms[ce_norms == 0] = 1.0
                ce = ce / ce_norms
                
                raw_similarities = np.dot(ce, jd_norm)
                
                # Exclude honeypots / non-tech profiles
                for idx, cid in enumerate(candidate_ids):
                    if cid in honeypots: raw_similarities[idx] = -9999.0
                    if cid in non_tech_ids: raw_similarities[idx] = -8888.0
                
                # Fetch candidate detail structures
                top_indices = np.argsort(raw_similarities)[::-1][:20]
                top_ids_set = set(candidate_ids[i] for i in top_indices)
                
                parsed_profiles = {}
                with open("candidates.jsonl", "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip(): continue
                        cand_match = re.search(r'"candidate_id":\s*"(CAND_\d{7})"', line)
                        if cand_match and cand_match.group(1) in top_ids_set:
                            parsed_profiles[cand_match.group(1)] = json.loads(line)
                
                # Apply Multi-dimensional Score weights
                scored_list = []
                for i in top_indices:
                    cid = candidate_ids[i]
                    cand = parsed_profiles.get(cid)
                    if not cand: continue
                    
                    scores_dict = {
                        "semantic": max(0.0, float(raw_similarities[i])),
                        "skills": evaluate_skills(cand),
                        "experience": evaluate_experience(cand),
                        "projects": evaluate_project_relevance(cand),
                        "velocity": evaluate_velocity_and_stability(cand),
                        "behavior": evaluate_behavior(cand),
                    }
                    
                    final_match = sum(weights[k] * scores_dict[k] for k in weights)
                    final_match = round(float(max(0.0, min(1.0, final_match))), 4)
                    
                    scored_list.append({
                        "id": cid,
                        "cand": cand,
                        "scores": scores_dict,
                        "final_score": final_match
                    })
                
                scored_list.sort(key=lambda x: (-x["final_score"], x["id"]))
                st.session_state["scored_list"] = scored_list[:10]
        
        # Display candidates
        if "scored_list" in st.session_state and st.session_state["scored_list"]:
            ranked_list = st.session_state["scored_list"]
            
            # Helper function for missing skills
            def get_missing_skills(cand):
                cand_skills_lower = {s.get("name", "").lower() for s in cand.get("skills", [])}
                missing = []
                for cat, syns in CORE_SKILLS_MAPPING.items():
                    has_skill = any(any(syn in cs for cs in cand_skills_lower) or any(cs in syn for cs in cand_skills_lower) for syn in syns)
                    if not has_skill:
                        missing.append(cat.replace("_", " ").title())
                for cat, syns in PREFERRED_SKILLS_MAPPING.items():
                    has_skill = any(any(syn in cs for cs in cand_skills_lower) or any(cs in syn for cs in cand_skills_lower) for syn in syns)
                    if not has_skill:
                        missing.append(cat.replace("_", " ").title())
                return missing

            # 1. Summary Metrics Row
            total_cand = len(ranked_list)
            top_score = f"{ranked_list[0]['final_score'] * 100:.1f}%" if total_cand > 0 else "--"
            avg_score = f"{np.mean([x['final_score'] for x in ranked_list]) * 100:.1f}%" if total_cand > 0 else "--"
            num_shortlisted = len(st.session_state.get("shortlisted", []))
            num_compared = len(st.session_state.get("comparison_pool", []))
            
            st.markdown(f"""
            <div class="metrics-row">
              <div class="metric-card">
                <div class="metric-label">Candidates Evaluated</div>
                <div class="metric-value">{total_cand}</div>
              </div>
              <div class="metric-card">
                <div class="metric-label">Top Match</div>
                <div class="metric-value" style="color: #89D7B7;">{top_score}</div>
              </div>
              <div class="metric-card">
                <div class="metric-label">Average Match</div>
                <div class="metric-value" style="color: #FFF4E1;">{avg_score}</div>
              </div>
              <div class="metric-card">
                <div class="metric-label">Shortlisted</div>
                <div class="metric-value" style="color: #89D7B7;">{num_shortlisted} / {total_cand}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # 2. Search & Filter Bar
            st.markdown('<div class="panel-title" style="margin-top: 20px; margin-bottom: 8px;">Search & Filter Pool</div>', unsafe_allow_html=True)
            search_query = st.text_input("Search Candidate pool...", placeholder="🔍 Filter by candidate name, company, title, or skills...", label_visibility="collapsed", key="cand_search_input")
            
            col_f1, col_f2 = st.columns([2, 2])
            with col_f1:
                notice_filter = st.selectbox("Max Notice Period", ["All", "Immediate (0-15 days)", "30 Days", "60 Days", "90 Days"], index=0, key="notice_filter_select")
            with col_f2:
                shortlist_filter = st.checkbox("Show Shortlisted Only", value=False, key="shortlist_filter_check")

            # 3. Apply Filters
            filtered_list = ranked_list
            if search_query:
                sq = search_query.lower()
                filtered_list = [
                    x for x in filtered_list
                    if sq in x["cand"]["profile"].get("anonymized_name", "").lower() or
                       sq in x["cand"]["profile"].get("current_title", "").lower() or
                       sq in x["cand"]["profile"].get("current_company", "").lower() or
                       any(sq in s.get("name", "").lower() for s in x["cand"].get("skills", []))
                ]
            
            if notice_filter != "All":
                days_limit = {"Immediate (0-15 days)": 15, "30 Days": 30, "60 Days": 60, "90 Days": 90}[notice_filter]
                filtered_list = [x for x in filtered_list if x["cand"].get("redrob_signals", {}).get("notice_period_days", 30) <= days_limit]
            
            if shortlist_filter:
                filtered_list = [x for x in filtered_list if x["id"] in st.session_state["shortlisted"]]


            # 5. Side-by-side Candidate Comparison Drawer
            if len(st.session_state["comparison_pool"]) >= 1:
                comp_ids = st.session_state["comparison_pool"]
                comp_items = [x for x in ranked_list if x["id"] in comp_ids]
                
                if len(comp_items) == 1:
                    st.info("💡 Candidate added to comparison pool. Select a second candidate below to view side-by-side comparison matrix!")
                
                st.markdown('<div class="panel-title" style="margin-top: 16px;">⚖️ Candidate Comparison Matrix</div>', unsafe_allow_html=True)
                
                header_row = "<tr><th class='label-col'>Feature</th>"
                score_row = "<tr><td class='label-col'>Match Score</td>"
                exp_row = "<tr><td class='label-col'>Experience</td>"
                loc_row = "<tr><td class='label-col'>Location</td>"
                notice_row = "<tr><td class='label-col'>Notice Period</td>"
                skills_row = "<tr><td class='label-col'>Key Skills</td>"
                strengths_row = "<tr><td class='label-col'>Why Ranked</td>"
                missing_row = "<tr><td class='label-col'>Missing Skills</td>"
                
                for item in comp_items:
                    cname = item["cand"]["profile"].get("anonymized_name", "Candidate")
                    cscore = f"{item['final_score']*100:.1f}%"
                    cexp = f"{item['cand']['profile'].get('years_of_experience', 0):.1f} Years"
                    cloc = item["cand"]["profile"].get("location", "Noida, India")
                    cnotice = f"{item['cand'].get('redrob_signals', {}).get('notice_period_days', 30)} Days"
                    c_skills = ", ".join([s.get("name", "") for s in item["cand"].get("skills", [])[:5]])
                    
                    c_strengths = "No major highlights"
                    if item["scores"]["projects"] > 0.5: c_strengths = "Strong vector search/projects"
                    elif item["scores"]["skills"] > 0.6: c_strengths = "Good technical matching"
                    
                    c_missing = ", ".join(get_missing_skills(item["cand"])[:3]) or "None"
                    
                    header_row += f"<th>{cname}</th>"
                    score_row += f"<td style='font-weight: 700; color: #818CF8;'>{cscore}</td>"
                    exp_row += f"<td>{cexp}</td>"
                    loc_row += f"<td>{cloc}</td>"
                    notice_row += f"<td>{cnotice}</td>"
                    skills_row += f"<td>{c_skills}</td>"
                    strengths_row += f"<td>{c_strengths}</td>"
                    missing_row += f"<td style='color: #FCA5A5;'>{c_missing}</td>"
                    
                header_row += "</tr>"
                score_row += "</tr>"
                exp_row += "</tr>"
                loc_row += "</tr>"
                notice_row += "</tr>"
                skills_row += "</tr>"
                strengths_row += "</tr>"
                missing_row += "</tr>"
                
                table_html = f"""
                <table class="comparison-table">
                  {header_row}
                  {score_row}
                  {exp_row}
                  {loc_row}
                  {notice_row}
                  {skills_row}
                  {strengths_row}
                  {missing_row}
                </table>
                """
                st.markdown(table_html, unsafe_allow_html=True)
                
                comp_cols = st.columns(len(comp_items) + 1)
                with comp_cols[0]:
                    st.write("")
                for idx, item in enumerate(comp_items):
                    with comp_cols[idx + 1]:
                        if st.button(f"Remove {item['cand']['profile'].get('anonymized_name').split()[0]} ❌", key=f"rm_comp_btn_{item['id']}"):
                            st.session_state["comparison_pool"].remove(item["id"])
                            st.rerun()

            # 6. Render Candidate Cards
            st.markdown(f"""
            <div class="res-header-bar" style="margin-top: 24px;">
              <span class="panel-title" style="margin-bottom: 0;">Verified Candidates</span>
              <span class="res-count-badge">{len(filtered_list)} Matches Shown</span>
            </div>
            """, unsafe_allow_html=True)
            
            if not filtered_list:
                st.info("No candidates match your active search and filter criteria.")
            else:
                for idx, item in enumerate(filtered_list):
                    cand = item["cand"]
                    scores = item["scores"]
                    score = item["final_score"]
                    
                    prof = cand.get("profile", {})
                    sigs = cand.get("redrob_signals", {})
                    skills = cand.get("skills", [])
                    
                    name = prof.get("anonymized_name", "Anonymous Candidate")
                    title = prof.get("current_title", "Software Engineer")
                    company = prof.get("current_company", "Product Company")
                    years = prof.get("years_of_experience", 0)
                    location = prof.get("location", "Noida, India")
                    notice = sigs.get("notice_period_days", 30)
                    
                    agreement = 1.0 - abs(scores.get("semantic", 0) - score)
                    completeness = (float(bool(cand.get("career_history"))) + float(bool(skills))) / 2.0
                    confidence = min(1.0, max(0.1, 0.7 * agreement + 0.3 * completeness))
                    
                    score_pct = score * 100
                    conf_pct = confidence * 100
                    score_cls = "high" if score_pct >= 70 else ("mid" if score_pct >= 45 else "low")
                    
                    skill_chips_html = ""
                    for s in skills[:12]:
                        sname = s.get("name", "")
                        sn = sname.lower()
                        is_core = False
                        is_pref = False
                        for cat, syns in CORE_SKILLS_MAPPING.items():
                            if any(syn in sn for syn in syns):
                                is_core = True
                                break
                        if not is_core:
                            for cat, syns in PREFERRED_SKILLS_MAPPING.items():
                                if any(syn in sn for syn in syns):
                                    is_pref = True
                                    break
                        
                        if is_core:
                            chip_class = "skill-chip core"
                        elif is_pref:
                            chip_class = "skill-chip preferred"
                        else:
                            chip_class = "skill-chip"
                        skill_chips_html += f'<span class="{chip_class}">{sname}</span>'
                    if len(skills) > 12:
                        skill_chips_html += f'<span class="skill-chip">+{len(skills)-12} more</span>'
                    
                    radar_base64 = make_radar_base64(scores)
                    
                    why_bullets = []
                    if score >= 0.7:
                        why_bullets.append("Excellent overall relevancy score based on multi-dimensional weightings.")
                    if scores.get("semantic", 0) >= 0.7:
                        why_bullets.append("Strong semantic match to job description requirements.")
                    if scores.get("skills", 0) >= 0.6:
                        why_bullets.append("Matches core technology stack required for this job.")
                    if 5 <= years <= 9:
                        why_bullets.append("Ideal seniority level (5-9 years experience envelope).")
                    if scores.get("projects", 0) >= 0.5:
                        why_bullets.append("Hands-on product development history in search & retrieval.")
                    if scores.get("velocity", 0) >= 0.6:
                        why_bullets.append("Solid career progression and tenure metrics.")
                    if scores.get("behavior", 0) >= 0.7:
                        why_bullets.append("Excellent response rate and completion profile on recruiter signals.")
                    
                    why_html = "".join([f'<div class="insight-row"><span class="insight-bullet">&#10003;</span>{b}</div>' for b in why_bullets]) if why_bullets else '<div class="insight-row"><span class="insight-bullet">&#10003;</span>Matches target role profile parameters.</div>'
                    
                    missing_list = get_missing_skills(cand)
                    missing_chips_html = ""
                    for m in missing_list[:4]:
                        missing_chips_html += f'<span class="skill-chip missing">{m}</span>'
                    if not missing_chips_html:
                        missing_chips_html = '<span style="font-size: 11px; color: #89D7B7;">No major missing technical skill sets.</span>'
                    
                    risks = []
                    hist = cand.get("career_history", [])
                    avg_tenure = sum(j.get("duration_months", 0) for j in hist) / len(hist) if hist else 0
                    if avg_tenure < 18 and hist: risks.append(f"Average role tenure of {avg_tenure:.1f} months suggests high attrition risk.")
                    if notice > 60: risks.append(f"Extended notice period of {notice} days may delay onboarding timeline.")
                    risks_html = "".join([f'<div class="insight-row"><span class="insight-bullet" style="color:#F59E0B">!</span>{r}</div>' for r in risks])
                    risks_block = f'<div class="insight-section"><div class="insight-header">Risk Markers</div>{risks_html}</div>' if risks else ""
                    
                    summary_val = prof.get("summary", "No background statement provided.")
                    
                    is_shortlisted = item["id"] in st.session_state["shortlisted"]
                    is_in_comparison = item["id"] in st.session_state["comparison_pool"]
                    
                    card_classes = ["candidate-card"]
                    if is_shortlisted:
                        card_classes.append("shortlisted-card")
                    if is_in_comparison:
                        card_classes.append("in-comparison-card")
                    card_class = " ".join(card_classes)
                    
                    # Clean newlines from dynamic data variables to prevent breaking st.markdown HTML parsing
                    summary_val_clean = summary_val.replace('\n\n', '<br/><br/>').replace('\n', '<br/>').replace('\r', '')
                    why_html_clean = why_html.replace('\n', '').replace('\r', '')
                    missing_chips_html_clean = missing_chips_html.replace('\n', '').replace('\r', '')
                    risks_block_clean = risks_block.replace('\n', '').replace('\r', '') if risks_block else ""

                    # ── Candidate Card Shell with inline expandable details ───────────
                    card_html = f"""<div class="{card_class}">
                    <div class="card-top">
                    <div>
                    <div class="candidate-name">{name}</div>
                    <div class="candidate-title">{title} &middot; {company}</div>
                    </div>
                    <div class="score-badge-container">
                    <span class="confidence-badge">{conf_pct:.0f}% confidence</span>
                    <span class="score-badge {score_cls}">{score_pct:.1f}% Match</span>
                    </div>
                    </div>
                    <div class="progress-container">
                    <div class="progress-labels">
                    <span>Match Level</span>
                    <span>{score_pct:.1f}%</span>
                    </div>
                    <div class="progress-bar-bg">
                    <div class="progress-bar-fill" style="width: {score_pct}%;"></div>
                    </div>
                    </div>
                    <div class="meta-grid">
                    <div class="meta-item">
                    <span class="meta-label">Experience</span>
                    <span class="meta-value">{years:.1f} Years</span>
                    </div>
                    <div class="meta-item">
                    <span class="meta-label">Location</span>
                    <span class="meta-value">{location}</span>
                    </div>
                    <div class="meta-item">
                    <span class="meta-label">Notice Period</span>
                    <span class="meta-value">{notice} Days</span>
                    </div>
                    </div>
                    <div class="skill-chips">
                    {skill_chips_html}
                    </div>
                    <details class="card-details">
                    <summary>
                    <span>Show Technical Profile Details</span>
                    <span style="font-size: 10px;">▼</span>
                    </summary>
                    <div class="details-body">
                    <div class="details-left">
                    <div class="summary-text">
                    <div class="insight-header" style="margin-bottom: 4px;">Profile Summary</div>
                    {summary_val_clean}
                    </div>
                    <div class="insight-section" style="margin-top: 12px;">
                    <div class="insight-header" style="color: #89D7B7;">Why Ranked</div>
                    {why_html_clean}
                    </div>
                    <div class="insight-section" style="margin-top: 12px;">
                    <div class="insight-header" style="color: #FCA5A5;">Missing Core Areas</div>
                    <div style="display: flex; flex-wrap: wrap; gap: 6px; margin-top: 4px;">
                    {missing_chips_html_clean}
                    </div>
                    </div>
                    {risks_block_clean}
                    </div>
                    <div class="details-right">
                    <img src="data:image/png;base64,{radar_base64}" alt="Radar Chart" />
                    </div>
                    </div>
                    </details>
                    </div>"""

                    # Collapse all newlines in HTML to prevent st.markdown parser from interpreting lines as markdown block breaks
                    card_html_clean = card_html.replace('\n', ' ').replace('\r', '')
                    st.markdown(card_html_clean, unsafe_allow_html=True)


                    # ── Part 3: Action buttons ────────────────────────────────
                    btn_col1, btn_col2, btn_col3 = st.columns(3)

                    with btn_col1:
                        shortlist_label = "⭐ Remove Shortlist" if is_shortlisted else "⭐ Shortlist"
                        if st.button(shortlist_label, key=f"short_btn_{item['id']}"):
                            if is_shortlisted:
                                st.session_state["shortlisted"].remove(item["id"])
                                st.toast(f"Removed {name} from shortlist.", icon="⭐")
                            else:
                                st.session_state["shortlisted"].append(item["id"])
                                st.toast(f"Added {name} to shortlist!", icon="⭐")
                            st.rerun()

                    with btn_col2:
                        compare_label = "⚖️ Remove Compare" if is_in_comparison else "⚖️ Compare"
                        if st.button(compare_label, key=f"comp_btn_{item['id']}"):
                            if is_in_comparison:
                                st.session_state["comparison_pool"].remove(item["id"])
                                st.toast(f"Removed {name} from comparison pool.", icon="⚖️")
                            else:
                                if len(st.session_state["comparison_pool"]) >= 4:
                                    st.toast("⚠️ Compare up to 4 candidates.", icon="⚠️")
                                else:
                                    st.session_state["comparison_pool"].append(item["id"])
                                    st.toast(f"Added {name} to comparison pool!", icon="⚖️")
                            st.rerun()

                    with btn_col3:
                        is_viewing = st.session_state["view_resume"] == item["id"]
                        view_label = "👁️ Hide Details" if is_viewing else "👁️ View Details"
                        if st.button(view_label, key=f"view_btn_{item['id']}"):
                            if is_viewing:
                                st.session_state["view_resume"] = None
                            else:
                                st.session_state["view_resume"] = item["id"]
                            st.rerun()

                    # ── Part 4: Resume drawer — appears directly below this card ────────
                    if st.session_state.get("view_resume") == item["id"]:
                        vcand = item["cand"]
                        vprof = vcand.get("profile", {})
                        vsigs = vcand.get("redrob_signals", {})
                        vhistory = vcand.get("career_history", [])
                        vedu = vcand.get("education", [])
                        vlanguages = vcand.get("languages", [])
                        vcertifications = vcand.get("certifications", [])

                        timeline_html = '<div class="timeline-container">'
                        for job in vhistory:
                            jtitle = job.get("title", "Software Engineer")
                            jcomp = job.get("company", "Company")
                            jstart = job.get("start_date", "")
                            jend = job.get("end_date", "Present") if job.get("end_date") else "Present"
                            jdur = job.get("duration_months", 0)
                            jdesc = job.get("description", "No description provided.")
                            timeline_html += f"""
                            <div class="timeline-item">
                              <div class="timeline-dot"></div>
                              <div class="timeline-header">{jtitle} at {jcomp}</div>
                              <div class="timeline-sub">{jstart} - {jend} ({jdur} months)</div>
                              <div class="timeline-desc">{jdesc}</div>
                            </div>
                            """
                        timeline_html += '</div>'

                        edu_html = ""
                        if vedu:
                            for edu in vedu:
                                deg = edu.get("degree", "B.S.")
                                field = edu.get("field_of_study", "Computer Science")
                                inst = edu.get("institution", "University")
                                grade = edu.get("grade", "")
                                grade_str = f"({grade})" if grade else ""
                                edu_html += f'<div style="font-size:13px;color:#FFF4E1;margin-bottom:6px;"><strong>{deg} in {field}</strong><br/><span style="color:#A2C0B9;">{inst} {grade_str}</span></div>'
                        else:
                            edu_html = '<div style="font-size:12px;color:#6D9E94;">No education history provided.</div>'

                        salary_min = vsigs.get("expected_salary_range_inr_lpa", {}).get("min", "--")
                        salary_max = vsigs.get("expected_salary_range_inr_lpa", {}).get("max", "--")
                        salary_str = f"{salary_min} - {salary_max} LPA" if salary_min != "--" else "Not Specified"
                        mode = vsigs.get("preferred_work_mode", "flexible").title()
                        relocate = "Yes" if vsigs.get("willing_to_relocate") else "No"
                        active = vsigs.get("last_active_date", "--")
                        resp_rate = vsigs.get("recruiter_response_rate", 0) * 100
                        resp_time = vsigs.get("avg_response_time_hours", 24)
                        comp_rate = vsigs.get("interview_completion_rate", 0) * 100

                        cert_html = ""
                        if vcertifications:
                            for cert in vcertifications:
                                if isinstance(cert, dict):
                                    c_name = cert.get("name", "Certification")
                                    c_issuer = cert.get("issuer", "")
                                    c_year = cert.get("year", "")
                                    issuer_str = f" &bull; {c_issuer}" if c_issuer else ""
                                    year_str = f" ({c_year})" if c_year else ""
                                    cert_html += f'<div style="font-size:12px;color:#C8DDD8;margin-bottom:4px;">&bull; <strong>{c_name}</strong>{issuer_str}{year_str}</div>'
                                else:
                                    cert_html += f'<div style="font-size:12px;color:#C8DDD8;margin-bottom:4px;">&bull; {cert}</div>'
                        else:
                            cert_html = '<div style="font-size:12px;color:#6D9E94;">No certifications listed.</div>'

                        lang_html = ""
                        if vlanguages:
                            for l in vlanguages:
                                lang_html += f'<span class="confidence-badge" style="margin-right:6px;margin-bottom:6px;display:inline-block;">{l.get("language")} ({l.get("proficiency")})</span>'
                        else:
                            lang_html = '<div style="font-size:12px;color:#6D9E94;">No languages listed.</div>'

                        resume_html = f"""
                        <div class="resume-modal">
                          <div class="resume-header">
                            <div>
                              <div class="resume-title">{vprof.get("anonymized_name", "Anonymous Candidate")}</div>
                              <div class="resume-subtitle">{vprof.get("current_title", "Software Engineer")} &middot; {vprof.get("current_company", "Product Company")}</div>
                            </div>
                            <div style="font-size:14px;font-weight:700;color:#89D7B7;text-align:right;">
                              Match Score: {item['final_score']*100:.1f}%
                            </div>
                          </div>
                          <div style="display:flex;gap:24px;flex-wrap:wrap;">
                            <div style="flex:1.6;min-width:300px;">
                              <div class="resume-section-title">Career History</div>
                              {timeline_html}
                              <div class="resume-section-title" style="margin-top:20px;">Education</div>
                              {edu_html}
                            </div>
                            <div style="flex:1.2;min-width:240px;">
                              <div class="resume-section-title">Engagement &amp; Salary Signals</div>
                              <div class="signals-grid">
                                <div class="signal-card"><div class="signal-label">Expected Salary</div><div class="signal-value">{salary_str}</div></div>
                                <div class="signal-card"><div class="signal-label">Work Mode</div><div class="signal-value">{mode}</div></div>
                                <div class="signal-card"><div class="signal-label">Willing to Relocate</div><div class="signal-value">{relocate}</div></div>
                                <div class="signal-card"><div class="signal-label">Last Active</div><div class="signal-value">{active}</div></div>
                                <div class="signal-card"><div class="signal-label">Recruiter Response</div><div class="signal-value">{resp_rate:.0f}% ({resp_time:.1f} hrs)</div></div>
                                <div class="signal-card"><div class="signal-label">Interview Completion</div><div class="signal-value">{comp_rate:.0f}%</div></div>
                              </div>
                              <div class="resume-section-title" style="margin-top:20px;">Certifications</div>
                              {cert_html}
                              <div class="resume-section-title" style="margin-top:20px;">Languages</div>
                              <div>{lang_html}</div>
                            </div>
                          </div>
                        </div>
                        """
                        st.html(resume_html)
                        if st.button("Close Details View ✖", key=f"close_resume_{item['id']}", type="primary"):
                            st.session_state["view_resume"] = None
                            st.rerun()

                    st.markdown('<div style="height: 16px;"></div>', unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="empty-illustration">
              <div style="font-size: 40px; color: #89D7B7;">◈</div>
              <div class="empty-headline">Talent Intelligence Pool Idle</div>
              <div class="empty-desc">Configure your job description weights on the left side, then click <strong>Score & Rank Candidates</strong> to initialize shortlists.</div>
            </div>
            """, unsafe_allow_html=True)
