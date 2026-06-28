import json
from datetime import datetime

def analyze_candidates(path):
    with open(path, "r", encoding="utf-8") as f:
        candidates = json.load(f)
        
    print(f"Total candidates in sample: {len(candidates)}")
    for cand in candidates:
        cid = cand["candidate_id"]
        profile = cand["profile"]
        skills = cand.get("skills", [])
        history = cand.get("career_history", [])
        signals = cand.get("redrob_signals", {})
        
        anomalies = []
        
        # Check 1: Expert proficiency with 0 duration_months
        expert_zero_duration = [s["name"] for s in skills if s.get("proficiency") in ["expert", "advanced"] and s.get("duration_months", 0) == 0]
        if expert_zero_duration:
            anomalies.append(f"Expert/advanced skills with 0 duration: {expert_zero_duration}")
            
        # Check 2: Total number of expert/advanced skills is high but total duration is low
        total_expert_skills = sum(1 for s in skills if s.get("proficiency") in ["expert", "advanced"])
        if total_expert_skills >= 5 and sum(s.get("duration_months", 0) for s in skills) < 10:
            anomalies.append(f"High number of expert/advanced skills ({total_expert_skills}) with very low total duration.")
            
        # Check 3: Date overlaps or impossible timelines in career history
        # (e.g. 8 years of experience at a company founded 3 years ago; or working before birth / graduation? Or impossible overlaps)
        # Let's check company size and dates
        for job in history:
            company = job.get("company")
            title = job.get("title")
            duration_months = job.get("duration_months", 0)
            start_date_str = job.get("start_date")
            end_date_str = job.get("end_date")
            
            if start_date_str and end_date_str:
                try:
                    start_dt = datetime.strptime(start_date_str, "%Y-%m-%d")
                    end_dt = datetime.strptime(end_date_str, "%Y-%m-%d")
                    calculated_duration = (end_dt - start_dt).days / 30.4375
                    if abs(calculated_duration - duration_months) > 6: # large mismatch between dates and duration_months
                        anomalies.append(f"Job duration mismatch at {company}: listed {duration_months} months, dates imply {calculated_duration:.1f} months.")
                except ValueError:
                    pass
            
            # Look at description or other fields for clues
            # "8 years of experience at a company founded 3 years ago"
            # Wait, how would we know when a company was founded?
            # Let's inspect the text of the descriptions in history to see if they mention founding dates or if there are other date mismatches.
            
        # Check 4: Years of experience mismatch
        # Sum of durations in career history vs profile years of experience
        total_history_months = sum(job.get("duration_months", 0) for job in history)
        history_years = total_history_months / 12.0
        profile_years = profile.get("years_of_experience", 0)
        if abs(profile_years - history_years) > 3.0:
            anomalies.append(f"Experience mismatch: profile says {profile_years} years, but history duration is {history_years:.1f} years.")

        # Check 5: Notice period or other impossible combos
        # e.g., expected salary is extremely low for many years of experience, or signals show impossible states
        
        # Check 6: Check for "skills" overlap
        # Check if the candidate profile looks like a keyword stuffer: e.g. 30+ skills all with identical duration/endorsements
        skills_duration = [s.get("duration_months", 0) for s in skills]
        skills_endorsements = [s.get("endorsements", 0) for s in skills]
        if len(skills) > 15:
            # check if durations are identical or endorsements are identical
            if len(set(skills_duration)) <= 2:
                anomalies.append(f"Suspicious skills duration (almost all same): {set(skills_duration)}")
            if len(set(skills_endorsements)) <= 2:
                anomalies.append(f"Suspicious skill endorsements (almost all same): {set(skills_endorsements)}")

        if anomalies:
            print(f"Candidate {cid} ({profile.get('anonymized_name')}, {profile.get('current_title')}):")
            for a in anomalies:
                print(f"  - {a}")

if __name__ == '__main__':
    analyze_candidates(r"C:\Users\vinya\Downloads\[PUB] India_runs_data_and_ai_challenge\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\sample_candidates.json")
