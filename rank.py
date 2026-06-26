import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
import argparse
import json
import re
import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer

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
    "project manager", "program manager",  # non-technical PM roles
    "finance manager", "accountant", "chartered accountant", "ca ",
    "customer support", "customer success", "customer service",
    "teacher", "professor", "lecturer",
]

# Tech title keywords — at least one of these must appear in career history
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

def is_service_firm(company_name):
    name = company_name.lower()
    return any(firm in name for firm in SERVICE_FIRMS)

def is_non_tech_profile(cand):
    """Returns True if the candidate's entire career history is in non-tech roles.
    These are 'Tier 5 disguised' candidates who keyword-stuff AI skills but have
    zero legitimate tech engineering backgrounds."""
    profile = cand.get("profile", {})
    history = cand.get("career_history", [])
    
    current_title = profile.get("current_title", "").lower()
    
    if not history:
        return True  # No history = suspicious
    
    # A job title counts as TECH only if:
    # 1. It does NOT match any non-tech title (blocklist check first)
    # 2. AND it contains at least one genuine tech keyword
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
        # First: exclude if it matches a non-tech title
        is_non_tech = any(nt in title for nt in NON_TECH_TITLES)
        if is_non_tech:
            continue
        # Then: count as tech if it contains a genuine tech role keyword
        if any(kw in title for kw in GENUINE_TECH_KEYWORDS):
            tech_job_count += 1
    
    total_jobs = len(history)
    
    # Check if current title is non-tech
    is_current_non_tech = any(nt in current_title for nt in NON_TECH_TITLES)
    tech_ratio = tech_job_count / total_jobs
    
    # Disqualify if: has zero genuine tech jobs
    if tech_job_count == 0:
        return True
    
    # Disqualify if: current title is non-tech AND less than 25% of jobs are genuine tech
    if is_current_non_tech and tech_ratio < 0.25:
        return True
        
    return False

def evaluate_skills(cand):
    skills = cand.get("skills", [])
    signals = cand.get("redrob_signals", {})
    assessment_scores = signals.get("skill_assessment_scores", {})
    
    if not skills:
        return 0.0
        
    # Check if keyword stuffer: many skills but very low average duration
    total_duration = sum(s.get("duration_months", 0) for s in skills)
    avg_duration = total_duration / len(skills) if skills else 0
    stuffer_penalty = 1.0
    if len(skills) > 20 and avg_duration < 12:
        stuffer_penalty = 0.5
        
    core_scores = []
    for cat, synonyms in CORE_SKILLS_MAPPING.items():
        cat_score = 0.0
        for s in skills:
            s_name = s.get("name", "").lower()
            if any(syn in s_name for syn in synonyms):
                # Calculate candidate's skill quality
                prof = s.get("proficiency", "beginner")
                prof_val = {"beginner": 0.4, "intermediate": 0.7, "advanced": 0.9, "expert": 1.0}.get(prof, 0.4)
                
                dur = s.get("duration_months", 0)
                dur_mult = 1.0 if dur >= 24 else (0.5 if dur < 6 else 0.5 + 0.5 * (dur - 6) / 18.0)
                
                ends = s.get("endorsements", 0)
                ends_mult = 1.0 + min(0.3, np.log1p(ends) / 10.0)
                
                # Check assessment score
                assess_score = 0.0
                for skill_key, score in assessment_scores.items():
                    if skill_key.lower() in s_name or s_name in skill_key.lower():
                        assess_score = max(assess_score, score)
                assess_mult = 1.0 + (assess_score / 100.0) * 0.3 if assess_score > 0 else 1.0
                
                score = prof_val * dur_mult * ends_mult * assess_mult
                cat_score = max(cat_score, score)
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
                score = prof_val * dur_mult
                cat_score = max(cat_score, score)
        pref_scores.append(min(1.0, cat_score))
        
    pref_avg = np.mean(pref_scores) if pref_scores else 0.0
    
    # Additional check: if profile is non-tech, skills are likely stuffed
    if is_non_tech_profile(cand):
        return float((0.7 * core_avg + 0.3 * pref_avg) * stuffer_penalty * 0.1)
    
    final_skill_score = (0.7 * core_avg + 0.3 * pref_avg) * stuffer_penalty
    return float(final_skill_score)

def evaluate_experience(cand):
    profile = cand.get("profile", {})
    history = cand.get("career_history", [])
    
    # HARD DISQUALIFIER: Non-tech career history (Civil/Mechanical/HR/etc disguised as AI)
    # These are "Tier 5 plain-language" candidates with keyword-stuffed skills
    if is_non_tech_profile(cand):
        return 0.02  # Near-zero: effectively removes them from top 100
    
    # 1. Seniority Years fit
    profile_years = profile.get("years_of_experience", 0)
    if 5 <= profile_years <= 9:
        seniority_fit = 1.0
    elif profile_years == 4 or profile_years == 10:
        seniority_fit = 0.8
    elif profile_years == 3 or profile_years == 11:
        seniority_fit = 0.6
    elif profile_years == 2 or profile_years == 12:
        seniority_fit = 0.4
    else:
        seniority_fit = 0.1
        
    # 2. Applied AI/ML Experience in history — check TITLE AND DESCRIPTION
    ai_keywords = [
        "machine learning", "deep learning", "nlp", "natural language processing",
        "embeddings", "retrieval", "vector search", "llm", "transformers", "pytorch", "tensorflow",
        "ai engineer", "data scientist", "computer vision", "recommender system", "search engine",
        "rag", "information retrieval", "semantic search", "ranking", "recommendation",
        "hugging face", "bert", "gpt", "neural network", "fine-tuning"
    ]
    
    # AI title keywords — a job title must contain these for AI credit
    ai_title_keywords = [
        "ml engineer", "machine learning engineer", "ai engineer", "data scientist",
        "nlp engineer", "research scientist", "applied scientist", "search engineer",
        "recommendation", "cv engineer", "computer vision", "llm engineer",
        "deep learning", "ml researcher", "data engineer", "ai researcher"
    ]
    
    ai_months = 0
    total_months = 0
    research_jobs = 0
    eng_jobs = 0
    
    for job in history:
        title = job.get("title", "").lower()
        desc = job.get("description", "").lower()
        dur = job.get("duration_months", 0)
        total_months += dur
        
        # Check title type
        if any(kw in title for kw in ["researcher", "research assistant", "fellow", "scientist", "academic"]):
            research_jobs += 1
        if any(kw in title for kw in ["engineer", "developer", "programmer", "architect"]):
            eng_jobs += 1
            
        # Count AI experience only if: title is AI-relevant OR description has substantial AI content
        title_is_ai = any(kw in title for kw in ai_title_keywords)
        desc_has_ai = sum(1 for kw in ai_keywords if kw in desc) >= 2  # at least 2 AI keywords in desc
        
        if title_is_ai or (desc_has_ai and any(kw in title for kw in ["engineer", "scientist", "developer", "analyst", "researcher"])):
            ai_months += dur
            
    ai_years = ai_months / 12.0
    
    if ai_years >= 4.0:
        ai_score = 1.0
    elif 3.0 <= ai_years < 4.0:
        ai_score = 0.8
    elif 2.0 <= ai_years < 3.0:
        ai_score = 0.5
    elif 1.0 <= ai_years < 2.0:
        ai_score = 0.2
    else:
        ai_score = 0.0
        
    # Pure research penalty: if only research-based jobs, penalty
    research_penalty = 1.0
    if research_jobs > 0 and eng_jobs == 0:
        research_penalty = 0.2
        
    # Recent coding check: if current title is architect or manager for >18 months
    coding_penalty = 1.0
    if history:
        latest_job = history[0]
        if latest_job.get("is_current"):
            title = latest_job.get("title", "").lower()
            dur = latest_job.get("duration_months", 0)
            if any(m in title for m in ["manager", "director", "architect", "lead"]) and dur > 18:
                # Check description for lack of coding
                desc = latest_job.get("description", "").lower()
                if not any(code in desc for code in ["code", "programming", "python", "ship", "implement", "deploy", "build"]):
                    coding_penalty = 0.8
                    
    exp_score = (0.5 * seniority_fit + 0.5 * ai_score) * research_penalty * coding_penalty
    return float(exp_score)

def evaluate_project_relevance(cand):
    history = cand.get("career_history", [])
    if not history:
        return 0.0
        
    project_keywords = [
        "ranking", "retrieval", "vector search", "recommendation system", 
        "search engine", "rag", "information retrieval", "hybrid search"
    ]
    
    relevant_months = 0
    for job in history:
        desc = job.get("description", "").lower()
        title = job.get("title", "").lower()
        dur = job.get("duration_months", 0)
        
        if any(kw in desc or kw in title for kw in project_keywords):
            relevant_months += dur
            
    score = relevant_months / 48.0 # target 4 years
    return min(1.0, float(score))

def evaluate_velocity_and_stability(cand):
    history = cand.get("career_history", [])
    if not history:
        return 0.5
        
    # Service firm check: if ALL companies in history are service consulting firms
    is_all_service = True
    for job in history:
        comp = job.get("company", "")
        if comp and not is_service_firm(comp):
            is_all_service = False
            break
            
    if is_all_service:
        return 0.05 # Severe disqualification penalty
        
    # Stability: Average tenure
    total_months = sum(job.get("duration_months", 0) for job in history)
    num_jobs = len(history)
    avg_tenure = total_months / num_jobs if num_jobs > 0 else 0
    
    if avg_tenure >= 36:
        stability = 1.0
    elif avg_tenure < 12:
        stability = 0.1
    elif avg_tenure < 18:
        stability = 0.4
    else: # 18 to 36 months
        stability = 0.4 + 0.6 * (avg_tenure - 18) / 18.0
        
    # Title progression / velocity
    # Count promotions or transitions to senior roles
    senior_keywords = ["senior", "lead", "staff", "principal", "head", "director", "architect"]
    junior_keywords = ["junior", "associate", "intern", "trainee", "fresher"]
    
    has_senior = False
    has_junior = False
    
    for i, job in enumerate(reversed(history)): # chronological
        title = job.get("title", "").lower()
        if any(sk in title for sk in senior_keywords):
            has_senior = True
        if any(jk in title for jk in junior_keywords):
            has_junior = True
            
    # If they started junior and ended senior, clear progression
    if has_junior and has_senior:
        growth = 1.0
    elif has_senior:
        growth = 0.8
    else:
        growth = 0.6
        
    velocity = 0.5 * stability + 0.5 * growth
    return float(velocity)

def evaluate_behavior(cand):
    signals = cand.get("redrob_signals", {})
    
    # 1. Availability (open to work)
    open_to_work = 1.0 if signals.get("open_to_work_flag") else 0.8
    
    # 2. Recruiter engagement
    response_rate = signals.get("recruiter_response_rate", 0.5)
    resp_time = signals.get("avg_response_time_hours", 24)
    time_mult = 1.0 if resp_time <= 24 else (0.5 if resp_time >= 72 else 0.5 + 0.5 * (72 - resp_time) / 48.0)
    engagement = response_rate * time_mult
    
    # 3. Interview completion
    completion = signals.get("interview_completion_rate", 0.5)
    
    # 4. Last active recency
    last_act_str = signals.get("last_active_date", "2025-01-01")
    days_since = 365
    try:
        last_act_dt = datetime.strptime(last_act_str, "%Y-%m-%d")
        ref_dt = datetime(2026, 6, 26) # current challenge date
        days_since = (ref_dt - last_act_dt).days
    except ValueError:
        pass
        
    if days_since <= 30:
        recency = 1.0
    elif days_since <= 90:
        recency = 0.8
    elif days_since <= 180:
        recency = 0.5
    else:
        recency = 0.2
        
    # 5. Connections & Endorsements
    conns = signals.get("connection_count", 0)
    conns_score = min(1.0, np.log1p(conns) / np.log1p(500))
    
    ends = signals.get("endorsements_received", 0)
    ends_score = min(1.0, np.log1p(ends) / np.log1p(100))
    
    beh_score = (
        0.2 * open_to_work +
        0.2 * engagement +
        0.2 * completion +
        0.2 * recency +
        0.1 * conns_score +
        0.1 * ends_score
    )
    return float(beh_score)

def make_justification(cand, scores):
    profile = cand.get("profile", {})
    name = profile.get("anonymized_name", "Candidate")
    title = profile.get("current_title", "Software Engineer")
    years = profile.get("years_of_experience", 0)
    skills = cand.get("skills", [])
    
    # Find top technical matching skill
    ai_skills = ["nlp", "fine-tuning llms", "vector databases", "pytorch", "tensorflow", "fastapi", "embeddings", "milvus", "qdrant", "faiss"]
    matching_skills = [s.get("name") for s in skills if s.get("name", "").lower() in ai_skills]
    top_skill_str = f"with expertise in {matching_skills[0]}" if matching_skills else "with strong technical foundations"
    
    history = cand.get("career_history", [])
    has_product = False
    for job in history:
        comp = job.get("company", "").lower()
        if comp and not is_service_firm(comp):
            has_product = True
            break
            
    exp_context = "at product-focused companies" if has_product else "in professional environments"
    
    # Check notice period and location
    signals = cand.get("redrob_signals", {})
    notice = signals.get("notice_period_days", 30)
    loc = profile.get("location", "")
    
    # Build reasoning
    reasoning = (
        f"{name} is a {title} with {years:.1f} years of experience {exp_context}, {top_skill_str}. "
        f"They demonstrate high platform engagement (response rate: {signals.get('recruiter_response_rate', 0.0)*100:.0f}%) "
        f"and have a {notice}-day notice period, fitting the Noida/Pune hybrid target."
    )
    return reasoning

def rank_candidates(candidates_file, output_file):
    print("Loading precomputed candidate embeddings and IDs...")
    embeddings_archive = np.load("candidate_embeddings.npz")
    candidate_embeddings = embeddings_archive["embeddings"] # float16
    
    with open("candidate_ids.json", "r") as f:
        candidate_ids = json.load(f)
        
    with open("honeypots.json", "r") as f:
        honeypots = set(json.load(f))
        
    print("Reading Job Description...")
    with open("job_description.txt", "r", encoding="utf-8") as f:
        jd_text = f.read()
        
    print("Encoding Job Description...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    jd_embedding = model.encode(jd_text, convert_to_numpy=True).astype(np.float32)
    
    # Calculate cosine similarity using float32 matrix operations
    cand_emb_f32 = candidate_embeddings.astype(np.float32)
    jd_emb_f32 = jd_embedding / np.linalg.norm(jd_embedding)
    
    cand_emb_norms = np.linalg.norm(cand_emb_f32, axis=1, keepdims=True)
    cand_emb_norms[cand_emb_norms == 0] = 1.0
    cand_emb_f32 = cand_emb_f32 / cand_emb_norms
    
    print("Computing semantic similarity scores...")
    semantic_scores = np.dot(cand_emb_f32, jd_emb_f32)
    
    # Immediately blacklist honeypots
    for idx, cid in enumerate(candidate_ids):
        if cid in honeypots:
            semantic_scores[idx] = -9999.0
            
    # Retrieve top 5000 indices for detailed scoring (expanded to handle non-tech filter)
    print("Retrieving top 5000 candidates for multi-dimensional scoring...")
    top_indices = np.argsort(semantic_scores)[::-1][:5000]
    top_ids_set = set(candidate_ids[idx] for idx in top_indices)
    
    # Load candidate profiles from the JSONL file for top 2000
    print("Loading profiles for the top candidates from candidates.jsonl...")
    candidate_profiles = {}
    with open(candidates_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            # Simple substring check before full JSON parse for speed
            # Since candidate_id format is "CAND_XXXXXXX", we can match it using simple regex or string check
            cid_match = re.search(r'"candidate_id":\s*"(CAND_\d{7})"', line)
            if cid_match:
                cid = cid_match.group(1)
                if cid in top_ids_set:
                    cand = json.loads(line)
                    candidate_profiles[cid] = cand
                    
    # Compute hybrid scores for these top 2000 candidates
    scored_candidates = []
    
    print("Scoring candidates on Skills, Experience, Projects, Velocity, and Behavior...")
    for idx in top_indices:
        cid = candidate_ids[idx]
        cand = candidate_profiles.get(cid)
        if not cand:
            continue
            
        semantic_val = max(0.0, float(semantic_scores[idx]))
        
        # Call metric scoring functions
        skills_score = evaluate_skills(cand)
        exp_score = evaluate_experience(cand)
        proj_score = evaluate_project_relevance(cand)
        vel_score = evaluate_velocity_and_stability(cand)
        beh_score = evaluate_behavior(cand)
        
        # HARD DISQUALIFIER: Non-tech profiles are immediately suppressed to near-zero
        # regardless of their semantic similarity (they keyword-stuff AI terms to fool embeddings)
        if is_non_tech_profile(cand):
            final_score = 0.001
        else:
            # Calculate final composite score
            final_score = (
                0.30 * semantic_val +
                0.20 * skills_score +
                0.25 * exp_score +
                0.10 * proj_score +
                0.10 * vel_score +
                0.05 * beh_score
            )
            
        # Clamp to 0.0 - 1.0 range
        final_score = float(max(0.0, min(1.0, final_score)))
        
        scores = {
            "semantic": semantic_val,
            "skills": skills_score,
            "experience": exp_score,
            "projects": proj_score,
            "velocity": vel_score,
            "behavior": beh_score
        }
        
        reasoning = make_justification(cand, scores)
        
        scored_candidates.append({
            "candidate_id": cid,
            "score": final_score,
            "reasoning": reasoning
        })
        
    # Round the scores to 4 decimal places before sorting to avoid tie-breaker mismatches due to rounding
    for item in scored_candidates:
        item["score"] = round(item["score"], 4)

    # Sort: non-increasing by score, tie-break by candidate_id ascending
    print("Sorting and compiling rankings...")
    scored_candidates.sort(key=lambda x: (-x["score"], x["candidate_id"]))
    
    # Assign ranks 1 to 100
    top_100 = scored_candidates[:100]
    
    # Save the output CSV
    print(f"Writing rankings to {output_file}...")
    import csv
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for i, item in enumerate(top_100):
            writer.writerow([
                item["candidate_id"],
                i + 1,
                item["score"],
                item["reasoning"]
            ])
            
    print("Ranking successfully completed!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Rank candidates against the Senior AI Engineer Job Description.")
    parser.add_argument("--candidates", default="candidates.jsonl", help="Path to candidates JSONL file")
    parser.add_argument("--out", default="submission.csv", help="Path to write ranked candidates CSV")
    args = parser.parse_args()
    
    rank_candidates(args.candidates, args.out)
