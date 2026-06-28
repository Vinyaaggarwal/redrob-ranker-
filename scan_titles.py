import json
from collections import Counter

def scan_titles():
    filepath = "candidates.jsonl"
    
    titles = []
    ai_ml_titles = []
    
    count = 0
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            cand = json.loads(line)
            profile = cand["profile"]
            title = profile.get("current_title", "")
            titles.append(title)
            
            # Check if it looks like a valid tech/engineering title
            title_lower = title.lower()
            if any(kw in title_lower for kw in ["ai", "ml", "machine learning", "data scientist", "nlp", "backend", "software", "developer", "data engineer"]):
                ai_ml_titles.append(title)
                count += 1
                if count <= 20:
                    print(f"Valid Tech Candidate {cand['candidate_id']}: Title: {title}, Exp: {profile.get('years_of_experience')} yrs, Skills: {[s['name'] for s in cand.get('skills', [])[:5]]}")

    print(f"\nTotal Candidates: {len(titles)}")
    print(f"Total with tech/AI/ML titles: {len(ai_ml_titles)} ({len(ai_ml_titles)/len(titles)*100:.2f}%)")
    
    print("\nMost common overall titles:")
    print(Counter(titles).most_common(20))

if __name__ == '__main__':
    scan_titles()
