import json
from datetime import datetime

def scan_durations():
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
                start_date_str = job.get("start_date")
                end_date_str = job.get("end_date")
                duration_months = job.get("duration_months", 0)
                company = job.get("company", "")
                
                # If current job, end date is June 2026 (dataset time)
                if job.get("is_current") or end_date_str is None:
                    end_date_str = "2026-06-26"
                    
                if start_date_str and end_date_str:
                    try:
                        start_dt = datetime.strptime(start_date_str, "%Y-%m-%d")
                        end_dt = datetime.strptime(end_date_str, "%Y-%m-%d")
                        days = (end_dt - start_dt).days
                        calculated_months = days / 30.4375
                        
                        # Check if duration_months is significantly greater than calculated_months
                        # (e.g., duration is 96 months (8 years) but dates only cover 36 months (3 years))
                        if duration_months > calculated_months + 6:
                            print(f"Mismatched Job Duration: Candidate {cid} at {company}")
                            print(f"  Start: {start_date_str}, End: {job.get('end_date')} (is_current: {job.get('is_current')})")
                            print(f"  Stated Duration: {duration_months} months, Calculated Duration: {calculated_months:.1f} months")
                            count += 1
                            if count >= 15:
                                return
                    except Exception as e:
                        pass

    print(f"Total duration mismatches: {count}")

if __name__ == '__main__':
    scan_durations()
