import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
import streamlit as st
import json
import numpy as np
import re
import matplotlib.pyplot as plt
from datetime import datetime
from sentence_transformers import SentenceTransformer

# Set page config
st.set_page_config(
    page_title="Redrob Talent Intelligence Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
SERVICE_FIRMS = [
    "tcs", "tata consultancy", "infosys", "wipro", "accenture", 
    "cognizant", "capgemini", "tech mahindra", "hcl", "l&t", "lnt", 
    "mindtree", "mphasis"
]

# Titles that are clearly non-tech / non-AI roles (Tier 5 disguised candidates)
NON_TECH_TITLES = [
    "civil engineer", "mechanical engineer", "electrical engineer", "structural engineer",
    "chemical engineer", "aerospace engineer", "automobile engineer", "industrial engineer",
    "hr manager", "human resources", "hr executive", "talent acquisition", "recruiter",
    "graphic designer", "ui designer", "fashion designer", "interior designer",
    "sales executive", "sales manager", "account manager", "business development",
    "operations manager", "operations executive", "supply chain", "logistics",
    "marketing manager", "marketing executive", "brand manager", "content writer",
    "project manager", "program manager",
    "finance manager", "accountant", "chartered accountant", "ca ",
    "customer support", "customer success", "customer service",
    "teacher", "professor", "lecturer",
]

TECH_TITLE_KEYWORDS = [
    "engineer", "developer", "programmer", "scientist", "analyst",
    "architect", "researcher", "ml ", "ai ", "nlp", "data ",
    "software", "backend", "frontend", "fullstack", "devops", "sre",
    "platform", "infrastructure", "cloud", "security", "systems",
]

CORE_SKILLS_MAPPING = {
    "embeddings-based retrieval": ["embeddings", "dense retrieval", "rag", "search engine", "retrieval", "semantic search", "information retrieval", "search system", "indexing", "vector search"],
    "vector databases": ["vector db", "vector database", "faiss", "pinecone", "milvus", "qdrant", "weaviate", "chroma", "elasticsearch", "opensearch"],
    "hybrid search": ["hybrid search", "bm25", "sparse search", "keyword search", "lexical search"],
    "python": ["python", "py", "django", "flask", "fastapi"],
    "evaluation frameworks": ["ndcg", "mrr", "map", "evaluation", "ranking evaluation", "recall", "precision", "benchmark", "evaluation framework"]
}

PREFERRED_SKILLS_MAPPING = {
    "llm fine-tuning": ["fine-tuning", "lora", "qlora", "peft", "sft", "rlhf", "pre-training", "transformers", "huggingface", "llama", "mistral", "deepspeed", "fsdp"],
    "learning-to-rank": ["learning to rank", "ltr", "xgboost", "lightgbm", "catboost", "neural ranking"],
    "distributed systems": ["distributed", "spark", "ray", "kubernetes", "docker", "scale", "clustering"],
    "hr-tech": ["hr-tech", "recruiting tech", "recruitment", "marketplace", "talent acquisition"],
    "open-source": ["open-source", "open source", "github", "contributions", "pull requests"]
}

# Helper functions (matching rank.py exactly for fidelity)
def is_service_firm(company_name):
    name = company_name.lower()
    return any(firm in name for firm in SERVICE_FIRMS)

def is_non_tech_profile(cand):
    profile = cand.get("profile", {})
    history = cand.get("career_history", [])
    current_title = profile.get("current_title", "").lower()
    if not history:
        return True
    GENUINE_TECH_KEYWORDS = [
        "software engineer", "ml engineer", "ai engineer", "machine learning",
        "data engineer", "data scientist", "nlp engineer", "backend engineer",
        "frontend engineer", "fullstack", "devops engineer", "sre", "platform engineer",
        "infrastructure engineer", "cloud engineer", "research engineer", "applied scientist",
        "search engineer", "recommendation", "computer vision", "deep learning engineer",
        "developer", "programmer", "software developer", "android engineer", "ios engineer",
        "security engineer", "systems engineer", "network engineer",
        "data analyst", "research scientist", "business intelligence",
        "product engineer", "solutions engineer",
    ]
    tech_job_count = 0
    for job in history:
        title = job.get("title", "").lower()
        is_non_tech = any(nt in title for nt in NON_TECH_TITLES)
        if is_non_tech:
            continue
        if any(kw in title for kw in GENUINE_TECH_KEYWORDS):
            tech_job_count += 1
    total_jobs = len(history)
    is_current_non_tech = any(nt in current_title for nt in NON_TECH_TITLES)
    tech_ratio = tech_job_count / total_jobs
    if tech_job_count == 0:
        return True
    if is_current_non_tech and tech_ratio < 0.25:
        return True
    return False

def evaluate_skills(cand):
    skills = cand.get("skills", [])
    signals = cand.get("redrob_signals", {})
    assessment_scores = signals.get("skill_assessment_scores", {})
    if not skills: return 0.0
    total_duration = sum(s.get("duration_months", 0) for s in skills)
    avg_duration = total_duration / len(skills) if skills else 0
    stuffer_penalty = 0.5 if len(skills) > 20 and avg_duration < 12 else 1.0
    
    core_scores = []
    for cat, synonyms in CORE_SKILLS_MAPPING.items():
        cat_score = 0.0
        for s in skills:
            s_name = s.get("name", "").lower()
            if any(syn in s_name for syn in synonyms):
                prof = s.get("proficiency", "beginner")
                prof_val = {"beginner": 0.4, "intermediate": 0.7, "advanced": 0.9, "expert": 1.0}.get(prof, 0.4)
                dur = s.get("duration_months", 0)
                dur_mult = 1.0 if dur >= 24 else (0.5 if dur < 6 else 0.5 + 0.5 * (dur - 6) / 18.0)
                ends = s.get("endorsements", 0)
                ends_mult = 1.0 + min(0.3, np.log1p(ends) / 10.0)
                assess_score = 0.0
                for skill_key, score in assessment_scores.items():
                    if skill_key.lower() in s_name or s_name in skill_key.lower():
                        assess_score = max(assess_score, score)
                assess_mult = 1.0 + (assess_score / 100.0) * 0.3 if assess_score > 0 else 1.0
                cat_score = max(cat_score, prof_val * dur_mult * ends_mult * assess_mult)
        core_scores.append(min(1.0, cat_score))
    core_avg = np.mean(core_scores) if core_scores else 0.0
    
    pref_scores = []
    for cat, synonyms in PREFERRED_SKILLS_MAPPING.items():
        cat_score = 0.0
        for s in skills:
            s_name = s.get("name", "").lower()
            if any(syn in s_name for syn in synonyms):
                prof = s.get("proficiency", "beginner")
                prof_val = {"beginner": 0.4, "intermediate": 0.7, "advanced": 0.9, "expert": 1.0}.get(prof, 0.4)
                dur = s.get("duration_months", 0)
                dur_mult = 1.0 if dur >= 12 else (0.5 if dur < 3 else 0.5 + 0.5 * (dur - 3) / 9.0)
                cat_score = max(cat_score, prof_val * dur_mult)
        pref_scores.append(min(1.0, cat_score))
    pref_avg = np.mean(pref_scores) if pref_scores else 0.0
    
    if is_non_tech_profile(cand):
        return float((0.7 * core_avg + 0.3 * pref_avg) * stuffer_penalty * 0.1)
    return float((0.7 * core_avg + 0.3 * pref_avg) * stuffer_penalty)

def evaluate_experience(cand):
    profile = cand.get("profile", {})
    history = cand.get("career_history", [])
    
    # Hard disqualifier for non-tech profiles
    if is_non_tech_profile(cand):
        return 0.02
    
    profile_years = profile.get("years_of_experience", 0)
    if 5 <= profile_years <= 9: seniority_fit = 1.0
    elif profile_years == 4 or profile_years == 10: seniority_fit = 0.8
    elif profile_years == 3 or profile_years == 11: seniority_fit = 0.6
    elif profile_years == 2 or profile_years == 12: seniority_fit = 0.4
    else: seniority_fit = 0.1
        
    ai_keywords = ["ml", "machine learning", "deep learning", "nlp", "natural language processing", "embeddings", "retrieval", "vector", "llm", "transformers", "pytorch", "tensorflow", "ai engineer", "data scientist", "computer vision", "recommender", "search", "rag"]
    ai_months = sum(job.get("duration_months", 0) for job in history if any(kw in job.get("title", "").lower() or kw in job.get("description", "").lower() for kw in ai_keywords))
    ai_years = ai_months / 12.0
    
    ai_score = 1.0 if ai_years >= 4.0 else (0.8 if ai_years >= 3.0 else (0.5 if ai_years >= 2.0 else (0.2 if ai_years >= 1.0 else 0.0)))
    
    research_jobs = sum(1 for job in history if any(kw in job.get("title", "").lower() for kw in ["researcher", "research assistant", "fellow", "scientist", "academic"]))
    eng_jobs = sum(1 for job in history if any(kw in job.get("title", "").lower() for kw in ["engineer", "developer", "programmer", "architect"]))
    research_penalty = 0.2 if research_jobs > 0 and eng_jobs == 0 else 1.0
    
    coding_penalty = 1.0
    if history and history[0].get("is_current"):
        title = history[0].get("title", "").lower()
        dur = history[0].get("duration_months", 0)
        if any(m in title for m in ["manager", "director", "architect", "lead"]) and dur > 18:
            desc = history[0].get("description", "").lower()
            if not any(code in desc for code in ["code", "programming", "python", "ship", "implement", "deploy", "build"]):
                coding_penalty = 0.8
                
    return float((0.5 * seniority_fit + 0.5 * ai_score) * research_penalty * coding_penalty)

def evaluate_project_relevance(cand):
    history = cand.get("career_history", [])
    if not history: return 0.0
    project_keywords = ["ranking", "retrieval", "vector search", "recommendation system", "search engine", "rag", "information retrieval", "hybrid search"]
    relevant_months = sum(job.get("duration_months", 0) for job in history if any(kw in job.get("description", "").lower() or kw in job.get("title", "").lower() for kw in project_keywords))
    return min(1.0, float(relevant_months / 48.0))

def evaluate_velocity_and_stability(cand):
    history = cand.get("career_history", [])
    if not history: return 0.5
    if all(comp.get("company") and is_service_firm(comp.get("company")) for comp in history):
        return 0.05
    total_months = sum(job.get("duration_months", 0) for job in history)
    avg_tenure = total_months / len(history) if history else 0
    stability = 1.0 if avg_tenure >= 36 else (0.1 if avg_tenure < 12 else (0.4 if avg_tenure < 18 else 0.4 + 0.6 * (avg_tenure - 18) / 18.0))
    has_senior = any(any(sk in job.get("title", "").lower() for sk in ["senior", "lead", "staff", "principal", "head", "director", "architect"]) for job in history)
    has_junior = any(any(jk in job.get("title", "").lower() for jk in ["junior", "associate", "intern", "trainee", "fresher"]) for job in history)
    growth = 1.0 if has_junior and has_senior else (0.8 if has_senior else 0.6)
    return float(0.5 * stability + 0.5 * growth)

def evaluate_behavior(cand):
    signals = cand.get("redrob_signals", {})
    open_to_work = 1.0 if signals.get("open_to_work_flag") else 0.8
    response_rate = signals.get("recruiter_response_rate", 0.5)
    resp_time = signals.get("avg_response_time_hours", 24)
    time_mult = 1.0 if resp_time <= 24 else (0.5 if resp_time >= 72 else 0.5 + 0.5 * (72 - resp_time) / 48.0)
    engagement = response_rate * time_mult
    completion = signals.get("interview_completion_rate", 0.5)
    
    last_act_str = signals.get("last_active_date", "2025-01-01")
    days_since = 365
    try:
        last_act_dt = datetime.strptime(last_act_str, "%Y-%m-%d")
        days_since = (datetime(2026, 6, 26) - last_act_dt).days
    except ValueError: pass
    recency = 1.0 if days_since <= 30 else (0.8 if days_since <= 90 else (0.5 if days_since <= 180 else 0.2))
    
    conns = signals.get("connection_count", 0)
    conns_score = min(1.0, np.log1p(conns) / np.log1p(500))
    ends = signals.get("endorsements_received", 0)
    ends_score = min(1.0, np.log1p(ends) / np.log1p(100))
    
    return float(0.2 * open_to_work + 0.2 * engagement + 0.2 * completion + 0.2 * recency + 0.1 * conns_score + 0.1 * ends_score)

def make_radar_chart(scores, name):
    labels = ['Semantic', 'Skills', 'Experience', 'Projects', 'Velocity', 'Behavior']
    values = [scores['semantic'], scores['skills'], scores['experience'], scores['projects'], scores['velocity'], scores['behavior']]
    
    # Number of variables
    num_vars = len(labels)
    
    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    
    # The plot is a circle, so we need to "complete the loop" by appending the start value
    values += values[:1]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(3, 3), subplot_kw=dict(polar=True))
    
    # Draw one axe per variable + add labels
    plt.xticks(angles[:-1], labels, color='grey', size=8)
    
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=7)
    plt.ylim(0, 1)
    
    # Plot data
    ax.plot(angles, values, color='#6C5CE7', linewidth=2, linestyle='solid')
    
    # Fill area
    ax.fill(angles, values, color='#a29bfe', alpha=0.4)
    
    # Fine-tune visual options
    ax.spines['polar'].set_color('#dfe6e9')
    ax.grid(color='#dfe6e9')
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#f9f9fb')
    
    return fig

# Load data helper functions
@st.cache_resource
def load_assets():
    if not os.path.exists("candidate_embeddings.npz") or not os.path.exists("candidate_ids.json"):
        return None
    try:
        embeddings_archive = np.load("candidate_embeddings.npz")
        candidate_embeddings = embeddings_archive["embeddings"]
        with open("candidate_ids.json", "r") as f:
            candidate_ids = json.load(f)
        with open("honeypots.json", "r") as f:
            honeypots = set(json.load(f))
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return candidate_embeddings, candidate_ids, honeypots, model
    except Exception as e:
        st.error(f"Error loading assets: {e}")
        return None

def load_jd_and_candidates():
    # Load JD text
    jd_text = ""
    if os.path.exists("job_description.txt"):
        with open("job_description.txt", "r", encoding="utf-8") as f:
            jd_text = f.read()
    return jd_text

# Main UI layout
st.markdown("""
<div style='background-color:#6C5CE7;padding:1.5rem;border-radius:10px;margin-bottom:2rem;color:white;'>
    <h1 style='margin:0;font-size:2.2rem;'>🤖 Redrob Talent Intelligence Dashboard</h1>
    <p style='margin:0.5rem 0 0 0;font-size:1.1rem;opacity:0.9;'>Semantic Ranking Engine & Sandbox Demo</p>
</div>
""", unsafe_allow_html=True)

assets = load_assets()
if assets is None:
    st.warning("⚠️ Precomputed embeddings not found. Please run `precompute_embeddings.py` first to generate embeddings and IDs files.")
else:
    candidate_embeddings, candidate_ids, honeypots, model = assets
    jd_text_default = load_jd_and_candidates()
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📋 Search Configuration")
        jd_input = st.text_area("Job Description", value=jd_text_default, height=350)
        
        st.subheader("⚖️ Custom Slider Weights")
        w_semantic = st.slider("Semantic Similarity Weight", 0.0, 1.0, 0.35, 0.05)
        w_skills = st.slider("Skills Overlap Weight", 0.0, 1.0, 0.20, 0.05)
        w_experience = st.slider("Experience Fit Weight", 0.0, 1.0, 0.15, 0.05)
        w_projects = st.slider("Project Relevance Weight", 0.0, 1.0, 0.10, 0.05)
        w_velocity = st.slider("Career Velocity Weight", 0.0, 1.0, 0.10, 0.05)
        w_behavior = st.slider("Behavioral Signals Weight", 0.0, 1.0, 0.10, 0.05)
        
        # Calculate sum and normalize
        total_w = w_semantic + w_skills + w_experience + w_projects + w_velocity + w_behavior
        if total_w == 0:
            st.error("Total weight cannot be zero.")
            st.stop()
            
        st.caption(f"Raw weight sum: {total_w:.2f} (values will be normalized to sum to 100%)")
        
        # Normalize weights
        ns_semantic = w_semantic / total_w
        ns_skills = w_skills / total_w
        ns_experience = w_experience / total_w
        ns_projects = w_projects / total_w
        ns_velocity = w_velocity / total_w
        ns_behavior = w_behavior / total_w
        
    with col2:
        st.subheader("🏆 Candidate Shortlist (Top 10)")
        
        if st.button("🚀 Find & Rank Candidates"):
            with st.spinner("Embedding Job Description and computing similarities..."):
                jd_embedding = model.encode(jd_input, convert_to_numpy=True).astype(np.float32)
                
                # Perform fast dot-product
                cand_emb_f32 = candidate_embeddings.astype(np.float32)
                jd_emb_f32 = jd_embedding / np.linalg.norm(jd_embedding)
                cand_emb_norms = np.linalg.norm(cand_emb_f32, axis=1, keepdims=True)
                cand_emb_norms[cand_emb_norms == 0] = 1.0
                cand_emb_f32 = cand_emb_f32 / cand_emb_norms
                
                semantic_scores = np.dot(cand_emb_f32, jd_emb_f32)
                
                # Set honeypot semantic scores to low
                for idx, cid in enumerate(candidate_ids):
                    if cid in honeypots:
                        semantic_scores[idx] = -9999.0
                        
                top_indices = np.argsort(semantic_scores)[::-1][:20]
                top_ids_set = set(candidate_ids[idx] for idx in top_indices)
                
                # Stream profiles for top candidates
                candidate_profiles = {}
                with open("candidates.jsonl", "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        cid_match = re.search(r'"candidate_id":\s*"(CAND_\d{7})"', line)
                        if cid_match:
                            cid = cid_match.group(1)
                            if cid in top_ids_set:
                                cand = json.loads(line)
                                candidate_profiles[cid] = cand
                                
                results = []
                for idx in top_indices:
                    cid = candidate_ids[idx]
                    cand = candidate_profiles.get(cid)
                    if not cand:
                        continue
                        
                    sem = max(0.0, float(semantic_scores[idx]))
                    sk = evaluate_skills(cand)
                    ex = evaluate_experience(cand)
                    pr = evaluate_project_relevance(cand)
                    vl = evaluate_velocity_and_stability(cand)
                    bh = evaluate_behavior(cand)
                    
                    if is_non_tech_profile(cand):
                        final_score = 0.001
                    else:
                        final_score = (
                            ns_semantic * sem +
                            ns_skills * sk +
                            ns_experience * ex +
                            ns_projects * pr +
                            ns_velocity * vl +
                            ns_behavior * bh
                        )
                    
                    results.append({
                        "id": cid,
                        "cand": cand,
                        "scores": {
                            "semantic": sem,
                            "skills": sk,
                            "experience": ex,
                            "projects": pr,
                            "velocity": vl,
                            "behavior": bh
                        },
                        "final_score": final_score
                    })
                    
                # Sort
                results.sort(key=lambda x: (-x["final_score"], x["id"]))
                
                # Cache results in session state
                st.session_state["results"] = results[:10]
                
        if "results" in st.session_state:
            results = st.session_state["results"]
            for i, res in enumerate(results):
                cid = res["id"]
                cand = res["cand"]
                scores = res["scores"]
                final_score = res["final_score"]
                
                profile = cand.get("profile", {})
                signals = cand.get("redrob_signals", {})
                
                # Render candidate container
                with st.container():
                    st.markdown(f"---")
                    c_col1, c_col2 = st.columns([3, 1.2])
                    
                    with c_col1:
                        st.markdown(f"### Rank {i+1}: {profile.get('anonymized_name')} ({cid})")
                        st.markdown(f"**Current Role:** {profile.get('current_title')} at *{profile.get('current_company')}* | **Experience:** {profile.get('years_of_experience')} years")
                        st.markdown(f"**Location:** {profile.get('location')}, {profile.get('country')} | **Notice Period:** {signals.get('notice_period_days')} days")
                        st.write(f"**Headline:** {profile.get('headline')}")
                        st.write(f"**Summary:** {profile.get('summary')}")
                        
                        # Calculate Strengths & Gaps
                        strengths = []
                        gaps = []
                        risks = []
                        
                        if scores['projects'] > 0.6:
                            strengths.append("Demonstrated hands-on experience building search, ranking, or retrieval systems.")
                        if scores['skills'] > 0.7:
                            strengths.append("Broad skill alignment with core AI engineer target inventory (NLP, Vector search).")
                        if scores['experience'] > 0.8:
                            strengths.append("Matches ideal 5-9 years experience envelope for founding AI roles.")
                            
                        if scores['skills'] < 0.4:
                            gaps.append("Weak core technical skills in search/retrieval (NDCG metrics, vector indexing).")
                        if scores['experience'] < 0.4:
                            gaps.append("Experience total falls outside of targeted 5-9 years seniority level.")
                            
                        # Career Velocity Risks
                        avg_tenure = sum(job.get("duration_months", 0) for job in cand.get("career_history", [])) / len(cand.get("career_history", [])) if cand.get("career_history") else 0
                        if avg_tenure < 18:
                            risks.append(f"Job stability concern: average job duration is only {avg_tenure:.1f} months.")
                            
                        # Notice period concern
                        if signals.get("notice_period_days", 0) > 90:
                            risks.append(f"Long notice period: {signals.get('notice_period_days')} days could affect hiring timelines.")
                            
                        if strengths:
                            st.markdown("**✅ Strengths:**")
                            for s in strengths: st.write(f"- {s}")
                        if gaps:
                            st.markdown("**⚠️ Gaps:**")
                            for g in gaps: st.write(f"- {g}")
                        if risks:
                            st.markdown("**🚩 Risks & Concerns:**")
                            for r in risks: st.write(f"- {r}")
                            
                        # Custom targeted interview questions
                        questions = []
                        if scores['projects'] < 0.5:
                            questions.append("Can you explain how you would design a retrieval framework from scratch on a CPU-only environment?")
                        else:
                            questions.append("You mentioned building ranking/search pipelines. How did you optimize query latency and resolve embedding drift?")
                        if scores['skills'] < 0.5:
                            questions.append("Could you walk us through the differences in optimization between BM25 search and dense vector search?")
                        else:
                            questions.append("How did you set up your offline evaluation suite (NDCG, MAP, Recall@10)? How did offline-to-online metrics correlate?")
                            
                        st.markdown("**📝 Custom Interview Kit:**")
                        for q in questions: st.write(f"- *\"{q}\"*")
                            
                    with c_col2:
                        st.markdown(f"<div style='text-align:center;'><span style='font-size:2.5rem;font-weight:bold;color:#6C5CE7;'>{final_score*100:.1f}%</span><br/><span style='font-size:0.9rem;color:grey;'>Match Score</span></div>", unsafe_allow_html=True)
                        
                        # Completeness & confidence
                        completeness = (
                            float(bool(cand.get("career_history"))) +
                            float(bool(cand.get("skills"))) +
                            float(bool(signals.get("recruiter_response_rate") is not None)) +
                            float(bool(profile.get("years_of_experience") is not None)) +
                            float(bool(profile.get("summary")))
                        ) / 5.0
                        score_agreement = 1.0 - abs(scores['semantic'] - final_score)
                        confidence = 0.6 * score_agreement + 0.4 * completeness
                        confidence_val = min(1.0, max(0.1, confidence))
                        
                        st.markdown(f"<div style='text-align:center;margin-top:0.5rem;'><span style='font-weight:bold;color:#10AC84;'>{confidence_val*100:.0f}% Confidence</span></div>", unsafe_allow_html=True)
                        
                        # Draw radar chart
                        fig = make_radar_chart(scores, profile.get("anonymized_name"))
                        st.pyplot(fig)
                        plt.close(fig)
        else:
            st.info("💡 Click the button above to run the hybrid ranker and generate the recruiter shortlist.")
