import json
from datetime import datetime

# Founding/release dates
TECH_RELEASE_YEARS = {
    "pytorch": 2016,
    "tensorflow": 2015,
    "fastapi": 2018,
    "langchain": 2022,
    "qdrant": 2021,
    "pinecone": 2019,
    "weaviate": 2019,
    "milvus": 2019,
    "kubernetes": 2014,
    "docker": 2013,
    "spark": 2014,
    "airflow": 2015,
    "snowflake": 2012,
    "databricks": 2013,
    "transformers": 2017,
    "huggingface": 2016,
    "llama": 2023,
    "openai": 2015
}

def is_honeypot(cand):
    profile = cand["profile"]
    history = cand.get("career_history", [])
    skills = cand.get("skills", [])
    
    # Rule 1: Expert/Advanced skill with 0 duration
    expert_zero = [s["name"] for s in skills if s.get("proficiency") in ["expert", "advanced"] and s.get("duration_months", 0) == 0]
    if expert_zero:
        return True, f"Expert/Advanced skills with 0 duration: {expert_zero}"
        
    # Rule 2: Used technology before release in career history description
    for job in history:
        desc = job.get("description", "").lower()
        start_date_str = job.get("start_date")
        if start_date_str:
            try:
                start_year = int(start_date_str.split("-")[0])
                for tech, yr in TECH_RELEASE_YEARS.items():
                    if tech in desc and start_year < yr:
                        return True, f"Claims using {tech} at {job.get('company')} in {start_year}, but released in {yr}"
            except (ValueError, IndexError):
                pass
                
    # Rule 3: Major profile experience vs history duration mismatch
    total_history_months = sum(job.get("duration_months", 0) for job in history)
    history_years = total_history_months / 12.0
    profile_years = profile.get("years_of_experience", 0)
    if abs(profile_years - history_years) > 5.0:
        return True, f"Profile exp ({profile_years} yrs) vs history duration ({history_years:.1f} yrs) mismatch"
        
    # Rule 4: Job duration mismatch (stated duration exceeds dates duration)
    for job in history:
        start_date_str = job.get("start_date")
        end_date_str = job.get("end_date")
        duration_months = job.get("duration_months", 0)
        
        if job.get("is_current") or end_date_str is None:
            end_date_str = "2026-06-26"
            
        if start_date_str and end_date_str:
            try:
                start_dt = datetime.strptime(start_date_str, "%Y-%m-%d")
                end_dt = datetime.strptime(end_date_str, "%Y-%m-%d")
                days = (end_dt - start_dt).days
                calculated_months = days / 30.4375
                if duration_months > calculated_months + 6:
                    return True, f"Job duration mismatch: {duration_months} months stated, dates imply {calculated_months:.1f} months"
            except Exception:
                pass
                
    return False, ""

def main():
    filepath = r"C:\Users\vinya\Downloads\[PUB] India_runs_data_and_ai_challenge\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\candidates.jsonl"
    
    honeypots = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            cand = json.loads(line)
            cid = cand["candidate_id"]
            
            flag, reason = is_honeypot(cand)
            if flag:
                honeypots.append((cid, reason))
                
    print(f"Total honeypots detected: {len(honeypots)}")
    print("\nFirst 10 honeypots:")
    for cid, reason in honeypots[:10]:
        print(f"- {cid}: {reason}")
        
    # Write them to a file for easy use/blacklisting
    with open("honeypots.json", "w") as f:
        json.dump([item[0] for item in honeypots], f)
    print("\nWrote honeypot IDs to honeypots.json")

if __name__ == '__main__':
    main()
