import json
import re

def scan_founded():
    filepath = r"C:\Users\vinya\Downloads\[PUB] India_runs_data_and_ai_challenge\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\candidates.jsonl"
    
    count = 0
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            cand = json.loads(line)
            cid = cand["candidate_id"]
            history = cand.get("career_history", [])
            
            for job in history:
                desc = job.get("description", "")
                company = job.get("company", "")
                duration_months = job.get("duration_months", 0)
                start_date_str = job.get("start_date")
                
                # Check for "founded" in description
                if "founded" in desc.lower():
                    # Extract years if possible
                    years = re.findall(r'\b(19\d{2}|20\d{2})\b', desc)
                    if years:
                        founding_year = int(years[0])
                        if start_date_str:
                            try:
                                start_year = int(start_date_str.split("-")[0])
                                if start_year < founding_year:
                                    print(f"Honeypot Candidate {cid}: Worked at {company} starting {start_date_str} (start year {start_year}) but description says company founded in {founding_year}. Desc: {desc}")
                                    count += 1
                            except (ValueError, IndexError):
                                pass
                            
    print(f"Total found-mismatches: {count}")

if __name__ == '__main__':
    scan_founded()
