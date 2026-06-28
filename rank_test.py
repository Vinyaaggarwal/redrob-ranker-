import json
import re
import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer

# Non-tech titles that should be penalized/filtered
NON_TECH_TITLE_KEYWORDS = [
    "civil", "mechanical", "marketing", "operations", "accountant", 
    "hr manager", "human resources", "graphic designer", "sales", 
    "customer support", "business analyst", "content writer", "project manager"
]

def is_non_tech_title(title):
    t = title.lower()
    return any(kw in t for kw in NON_TECH_TITLE_KEYWORDS)

def test_ranking():
    print("Loading data...")
    embeddings_archive = np.load("candidate_embeddings.npz")
    candidate_embeddings = embeddings_archive["embeddings"]
    
    with open("candidate_ids.json", "r") as f:
        candidate_ids = json.load(f)
        
    with open("honeypots.json", "r") as f:
        honeypots = set(json.load(f))
        
    with open("job_description.txt", "r", encoding="utf-8") as f:
        jd_text = f.read()
        
    print("Encoding JD...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    jd_embedding = model.encode(jd_text, convert_to_numpy=True).astype(np.float32)
    
    # Calculate similarity
    cand_emb_f32 = candidate_embeddings.astype(np.float32)
    jd_emb_f32 = jd_embedding / np.linalg.norm(jd_embedding)
    cand_emb_norms = np.linalg.norm(cand_emb_f32, axis=1, keepdims=True)
    cand_emb_norms[cand_emb_norms == 0] = 1.0
    cand_emb_f32 = cand_emb_f32 / cand_emb_norms
    
    semantic_scores = np.dot(cand_emb_f32, jd_emb_f32)
    
    # Apply honeypot filter
    for idx, cid in enumerate(candidate_ids):
        if cid in honeypots:
            semantic_scores[idx] = -9999.0
            
    # Quick scan of profiles to filter out non-tech titles in Top 5000
    top_indices = np.argsort(semantic_scores)[::-1][:5000]
    top_ids_set = set(candidate_ids[idx] for idx in top_indices)
    
    # Load profiles
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
                    
    # Score
    results = []
    for idx in top_indices:
        cid = candidate_ids[idx]
        cand = candidate_profiles.get(cid)
        if not cand:
            continue
            
        profile = cand["profile"]
        title = profile.get("current_title", "")
        
        # Calculate scores
        sem = max(0.0, float(semantic_scores[idx]))
        
        # Title penalty check
        title_penalty = 1.0
        if is_non_tech_title(title):
            title_penalty = 0.02 # Huge penalty to filter them out of top ranks
            
        # Experience seniority & AI/ML scoring
        profile_years = profile.get("years_of_experience", 0)
        history = cand.get("career_history", [])
        
        # Check if they have an actual AI/ML title or general software title
        title_lower = title.lower()
        is_ai_ml_title = any(kw in title_lower for kw in ["ai", "ml", "machine learning", "data scientist", "nlp"])
        
        # Base composite score
        # (For this test, let's just see how title penalty shifts the rankings)
        score = sem * title_penalty
        
        results.append({
            "id": cid,
            "name": profile.get("anonymized_name"),
            "title": title,
            "exp": profile_years,
            "score": score,
            "sem": sem
        })
        
    results.sort(key=lambda x: -x["score"])
    
    print("\nTop 10 candidates with strict Title Filter:")
    for i, res in enumerate(results[:10]):
        print(f"Rank {i+1}: {res['id']} ({res['name']}) - Title: {res['title']}, Exp: {res['exp']} yrs, Sem Score: {res['sem']:.4f}, Final Score: {res['score']:.4f}")

if __name__ == '__main__':
    test_ranking()
