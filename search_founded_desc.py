import json

def search_founded():
    filepath = r"C:\Users\vinya\Downloads\[PUB] India_runs_data_and_ai_challenge\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\candidates.jsonl"
    
    count = 0
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            cand = json.loads(line)
            history = cand.get("career_history", [])
            for job in history:
                desc = job.get("description", "")
                if "founded" in desc.lower() or "establishment" in desc.lower() or "est." in desc.lower():
                    print(f"CANDIDATE: {cand['candidate_id']}, Company: {job.get('company')}")
                    print(f"Desc: {desc}\n")
                    count += 1
                    if count >= 10:
                        return

if __name__ == '__main__':
    search_founded()
