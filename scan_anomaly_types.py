import json
from collections import Counter

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

def scan_types():
    filepath = r"C:\Users\vinya\Downloads\[PUB] India_runs_data_and_ai_challenge\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\candidates.jsonl"
    
    anomaly_counts = Counter()
    total_candidates = 0
    
    # Store some examples for each type
    examples = {}
    
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            total_candidates += 1
            cand = json.loads(line)
            cid = cand["candidate_id"]
            profile = cand["profile"]
            history = cand.get("career_history", [])
            skills = cand.get("skills", [])
            signals = cand.get("redrob_signals", {})
            
            # 1. Expert/advanced skill with 0 duration
            expert_zero = [s["name"] for s in skills if s.get("proficiency") in ["expert", "advanced"] and s.get("duration_months", 0) == 0]
            if expert_zero:
                anomaly_counts["expert_zero_duration"] += 1
                if "expert_zero_duration" not in examples:
                    examples["expert_zero_duration"] = (cid, f"Skills: {expert_zero}")
                    
            # 2. Worked at company before founding
            company_mismatch = []
            for job in history:
                company = job.get("company", "").lower()
                start_date_str = job.get("start_date")
                if start_date_str:
                    try:
                        start_year = int(start_date_str.split("-")[0])
                        for comp, yr in COMPANY_FOUNDING_YEARS.items():
                            if comp in company and start_year < yr:
                                company_mismatch.append((job.get("company"), start_year, yr))
                    except (ValueError, IndexError):
                        pass
            if company_mismatch:
                anomaly_counts["company_founding_mismatch"] += 1
                if "company_founding_mismatch" not in examples:
                    examples["company_founding_mismatch"] = (cid, f"Worked at: {company_mismatch}")
                    
            # 3. Used technology before release
            tech_mismatch = []
            for job in history:
                desc = job.get("description", "").lower()
                start_date_str = job.get("start_date")
                if start_date_str:
                    try:
                        start_year = int(start_date_str.split("-")[0])
                        for tech, yr in TECH_RELEASE_YEARS.items():
                            if tech in desc and start_year < yr:
                                tech_mismatch.append((tech, start_year, yr))
                    except (ValueError, IndexError):
                        pass
            if tech_mismatch:
                anomaly_counts["tech_release_mismatch"] += 1
                if "tech_release_mismatch" not in examples:
                    examples["tech_release_mismatch"] = (cid, f"Techs: {tech_mismatch}")
                    
            # 4. Profile experience years vs history total years mismatch
            total_history_months = sum(job.get("duration_months", 0) for job in history)
            history_years = total_history_months / 12.0
            profile_years = profile.get("years_of_experience", 0)
            if abs(profile_years - history_years) > 5.0:
                anomaly_counts["experience_mismatch"] += 1
                if "experience_mismatch" not in examples:
                    examples["experience_mismatch"] = (cid, f"Profile: {profile_years} yrs, History: {history_years:.1f} yrs")
                    
            # 5. Salary range min > max
            expected_salary = signals.get("expected_salary_range_inr_lpa", {})
            if expected_salary:
                s_min = expected_salary.get("min", 0)
                s_max = expected_salary.get("max", 0)
                if s_min > s_max:
                    anomaly_counts["salary_range_mismatch"] += 1
                    if "salary_range_mismatch" not in examples:
                        examples["salary_range_mismatch"] = (cid, f"Min: {s_min}, Max: {s_max}")
                        
            # 6. Notice period mismatch
            if signals.get("notice_period_days", 0) > 180:
                anomaly_counts["notice_period_mismatch"] += 1
                if "notice_period_mismatch" not in examples:
                    examples["notice_period_mismatch"] = (cid, f"Days: {signals.get('notice_period_days')}")

            # 7. Check for specific impossible timeline: e.g. working for 8 years at a company founded 3 years ago (a specific job duration > company age)
            # Company age today (2026) minus founding year is how long the company exists.
            # If job duration_months / 12 > (2026 - founding_year), it's impossible!
            company_age_mismatch = []
            for job in history:
                company = job.get("company", "").lower()
                duration_months = job.get("duration_months", 0)
                for comp, yr in COMPANY_FOUNDING_YEARS.items():
                    if comp in company:
                        max_possible_months = (2026 - yr) * 12
                        if duration_months > max_possible_months:
                            company_age_mismatch.append((job.get("company"), duration_months, max_possible_months))
            if company_age_mismatch:
                anomaly_counts["company_age_mismatch"] += 1
                if "company_age_mismatch" not in examples:
                    examples["company_age_mismatch"] = (cid, f"Worked at: {company_age_mismatch}")
                    
    print(f"Total candidates analyzed: {total_candidates}")
    print("\nAnomaly counts by type:")
    for k, v in anomaly_counts.items():
        print(f"- {k}: {v} ({v/total_candidates*100:.3f}%)")
        if k in examples:
            print(f"  Example: {examples[k][0]} -> {examples[k][1]}")

if __name__ == '__main__':
    scan_types()
