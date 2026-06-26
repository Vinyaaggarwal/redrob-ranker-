"""
Investigate why Mechanical/Civil engineers still rank #2-20 after the fix.
Check what their career_history titles look like and why is_non_tech_profile returns False.
"""
import json, re

NON_TECH_TITLES = [
    "civil engineer", "mechanical engineer", "electrical engineer", "structural engineer",
    "chemical engineer", "aerospace engineer", "automobile engineer", "industrial engineer",
    "hr manager", "human resources", "hr executive", "talent acquisition", "recruiter",
    "graphic designer", "ui designer", "fashion designer", "interior designer",
    "sales executive", "sales manager", "account manager", "business development",
    "operations manager", "operations executive", "supply chain", "logistics",
    "marketing manager", "marketing executive", "brand manager", "content writer",
    "project manager", "program manager",
    "finance manager", "accountant", "chartered accountant", "ca ",
    "customer support", "customer success", "customer service",
    "teacher", "professor", "lecturer",
]

GENUINE_TECH_KEYWORDS = [
    "software engineer", "ml engineer", "ai engineer", "machine learning",
    "data engineer", "data scientist", "nlp engineer", "backend engineer",
    "frontend engineer", "fullstack", "devops engineer", "sre", "platform engineer",
    "infrastructure engineer", "cloud engineer", "research engineer", "applied scientist",
    "search engineer", "recommendation", "computer vision", "deep learning engineer",
    "developer", "programmer", "software developer", "android engineer", "ios engineer",
    "security engineer", "systems engineer", "network engineer",
    "data analyst", "research scientist", "business intelligence",
    "product engineer", "solutions engineer",
]

def is_non_tech_profile(cand):
    profile = cand.get("profile", {})
    history = cand.get("career_history", [])
    current_title = profile.get("current_title", "").lower()
    if not history:
        return True
    tech_job_count = 0
    for job in history:
        title = job.get("title", "").lower()
        is_non_tech = any(nt in title for nt in NON_TECH_TITLES)
        if is_non_tech:
            continue
        if any(kw in title for kw in GENUINE_TECH_KEYWORDS):
            tech_job_count += 1
    total_jobs = len(history)
    is_current_non_tech = any(nt in current_title for nt in NON_TECH_TITLES)
    tech_ratio = tech_job_count / total_jobs
    if tech_job_count == 0:
        return True
    if is_current_non_tech and tech_ratio < 0.25:
        return True
    return False

# Check new top-ranked candidates (ranks 2-10)
check_ids = ['CAND_0097094', 'CAND_0064972', 'CAND_0009551', 'CAND_0080618', 'CAND_0029530', 'CAND_0035165']
found = {}
with open('candidates.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        m = re.search(r'"candidate_id":\s*"(CAND_\d{7})"', line)
        if m and m.group(1) in check_ids:
            cand = json.loads(line)
            cid = m.group(1)
            found[cid] = cand
            if len(found) == len(check_ids):
                break

for cid in check_ids:
    if cid not in found:
        continue
    cand = found[cid]
    profile = cand.get('profile', {})
    current_title = profile.get('current_title', '?')
    history = cand.get('career_history', [])
    hist_titles = [(j.get('title', '?'), j.get('company', '?')) for j in history]
    is_nt = is_non_tech_profile(cand)
    
    # Show which jobs passed/failed the tech check
    tech_jobs = []
    for job in history:
        title = job.get("title", "").lower()
        is_non_tech = any(nt in title for nt in NON_TECH_TITLES)
        has_tech = any(kw in title for kw in GENUINE_TECH_KEYWORDS)
        if not is_non_tech and has_tech:
            tech_jobs.append(title)
    
    print(f"{cid}: '{current_title}' -> non_tech={is_nt}")
    print(f"  History: {hist_titles}")
    print(f"  Tech jobs found: {tech_jobs}")
    print()
