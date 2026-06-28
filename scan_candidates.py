import json
import gzip
import os
from datetime import datetime

# Founding dates of companies and release dates of technologies
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

COMPANY_FOUNDING_YEARS = {
    "openai": 2015,
    "pinecone": 2019,
    "qdrant": 2021,
    "weaviate": 2019,
    "milvus": 2019,
    "langchain": 2022,
    "cohere": 2019,
    "anthropic": 2021,
    "hugging face": 2016,
    "huggingface": 2016,
    "databricks": 2013,
    "snowflake": 2012,
}

def scan_dataset():
    filepath = r"C:\Users\vinya\Downloads\[PUB] India_runs_data_and_ai_challenge\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\candidates.jsonl"
    
    count = 0
    total_candidates = 0
    anomalies_detected = []
    
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            total_candidates += 1
            if not line.strip():
                continue
            cand = json.loads(line)
            cid = cand["candidate_id"]
            profile = cand["profile"]
            history = cand.get("career_history", [])
            skills = cand.get("skills", [])
            signals = cand.get("redrob_signals", {})
            
            anomalies = []
            
            # Rule 1: Expert/Advanced proficiency in skills but 0 months of use
            expert_zero_dur = [s["name"] for s in skills if s.get("proficiency") in ["expert", "advanced"] and s.get("duration_months", 0) == 0]
            if expert_zero_dur:
                anomalies.append(f"Expert/Advanced skills with 0 duration: {expert_zero_dur}")
                
            # Rule 2: Impossible technology usage in career history
            for job in history:
                company = job.get("company", "").lower()
                desc = job.get("description", "").lower()
                start_date_str = job.get("start_date")
                
                if start_date_str:
                    try:
                        start_year = int(start_date_str.split("-")[0])
                        
                        # Check company founding
                        for comp_name, found_year in COMPANY_FOUNDING_YEARS.items():
                            if comp_name in company and start_year < found_year:
                                anomalies.append(f"Started at {job.get('company')} in {start_year}, but it was founded in {found_year}")
                                
                        # Check tech release
                        for tech_name, release_year in TECH_RELEASE_YEARS.items():
                            # If they mention using it in the description, and the job started before its release
                            # We check for word boundaries or simple substring
                            if tech_name in desc and start_year < release_year:
                                # To avoid false positives, check if it's actually about using it
                                anomalies.append(f"Claims using {tech_name} at {job.get('company')} in {start_year}, but released in {release_year}")
                    except (ValueError, IndexError):
                        pass

            # Rule 3: Major experience duration mismatch
            total_history_months = sum(job.get("duration_months", 0) for job in history)
            history_years = total_history_months / 12.0
            profile_years = profile.get("years_of_experience", 0)
            if abs(profile_years - history_years) > 5.0:
                anomalies.append(f"Profile experience ({profile_years} yrs) vs history duration ({history_years:.1f} yrs) mismatch > 5 yrs")
                
            # Rule 4: Notice period too long or expected salary range is 0 or negative
            if signals.get("notice_period_days", 0) > 180:
                anomalies.append(f"Notice period {signals.get('notice_period_days')} > 180 days")
                
            expected_salary = signals.get("expected_salary_range_inr_lpa", {})
            if expected_salary:
                s_min = expected_salary.get("min", 0)
                s_max = expected_salary.get("max", 0)
                if s_min < 0 or s_max < 0 or s_min > s_max:
                    anomalies.append(f"Invalid salary range: {s_min} - {s_max}")

            if anomalies:
                anomalies_detected.append((cid, profile.get("anonymized_name"), anomalies))
                count += 1
                if count <= 15:
                    print(f"Candidate {cid} ({profile.get('anonymized_name')}):")
                    for a in anomalies:
                        print(f"  - {a}")
                        
    print(f"\nScan completed. Total Candidates analyzed: {total_candidates}")
    print(f"Anomalies detected: {len(anomalies_detected)} ({len(anomalies_detected)/total_candidates*100:.2f}%)")

if __name__ == '__main__':
    scan_dataset()
