import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def make_candidate_profile_text(cand):
    profile = cand.get("profile", {})
    skills = cand.get("skills", [])
    history = cand.get("career_history", [])
    
    parts = []
    # Headline and current details
    parts.append(f"Candidate is a {profile.get('current_title', '')} at {profile.get('current_company', '')}.")
    parts.append(f"Headline: {profile.get('headline', '')}")
    parts.append(f"Summary: {profile.get('summary', '')}")
    
    # Skills details
    skill_strings = []
    for s in skills:
        name = s.get("name", "")
        prof = s.get("proficiency", "")
        dur = s.get("duration_months", 0)
        skill_strings.append(f"{name} ({prof}, {dur}m)")
    if skill_strings:
        parts.append("Skills: " + ", ".join(skill_strings))
        
    # Career History details
    history_strings = []
    for job in history:
        comp = job.get("company", "")
        title = job.get("title", "")
        dur = job.get("duration_months", 0)
        desc = job.get("description", "")
        history_strings.append(f"{title} at {comp} ({dur}m): {desc}")
    if history_strings:
        parts.append("Career: " + " | ".join(history_strings))
        
    return " ".join(parts)

def precompute():
    candidates_path = "candidates.jsonl"
    out_npy_path = "candidate_embeddings.npz"
    out_ids_path = "candidate_ids.json"
    
    if not os.path.exists(candidates_path):
        print(f"Error: {candidates_path} not found in workspace.")
        return
        
    print("Loading SentenceTransformer model 'all-MiniLM-L6-v2'...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("Reading candidates and compiling text profiles...")
    texts = []
    ids = []
    
    with open(candidates_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, total=100000, desc="Parsing candidates"):
            if not line.strip():
                continue
            cand = json.loads(line)
            texts.append(make_candidate_profile_text(cand))
            ids.append(cand["candidate_id"])
            
    print(f"Loaded {len(texts)} candidates. Generating embeddings (this may take 10-15 minutes on CPU)...")
    
    # We use a batch size of 256 for fast encoding on CPU
    embeddings = model.encode(
        texts, 
        batch_size=256, 
        show_progress_bar=True, 
        convert_to_numpy=True
    )
    
    # Cast to float16 to save space (half precision, ~76MB instead of 153MB)
    embeddings_f16 = embeddings.astype(np.float16)
    
    print(f"Saving compressed embeddings to {out_npy_path}...")
    np.savez_compressed(out_npy_path, embeddings=embeddings_f16)
    
    print(f"Saving candidate IDs to {out_ids_path}...")
    with open(out_ids_path, "w", encoding="utf-8") as f:
        json.dump(ids, f)
        
    print("Precomputation finished successfully!")

if __name__ == '__main__':
    precompute()
