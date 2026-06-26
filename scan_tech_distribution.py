"""
Scan the full candidates.jsonl to count:
1. How many candidates are legitimate AI/tech profiles
2. How many would pass is_non_tech_profile() check
3. Distribution of current_title values for tech vs non-tech
"""
import json, re
from collections import Counter

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

print("Scanning full candidates.jsonl...")
total = 0
tech_count = 0
non_tech_count = 0
tech_titles = Counter()
non_tech_titles = Counter()

with open('candidates.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        if not line.strip():
            continue
        cand = json.loads(line)
        total += 1
        profile = cand.get('profile', {})
        current_title = profile.get('current_title', 'Unknown')
        
        if is_non_tech_profile(cand):
            non_tech_count += 1
            non_tech_titles[current_title] += 1
        else:
            tech_count += 1
            tech_titles[current_title] += 1
        
        if total % 10000 == 0:
            print(f"  Processed {total}... tech={tech_count}, non_tech={non_tech_count}")

print(f"\nTotal: {total}")
print(f"Tech profiles: {tech_count} ({100*tech_count/total:.1f}%)")
print(f"Non-tech profiles: {non_tech_count} ({100*non_tech_count/total:.1f}%)")
print(f"\nTop 20 TECH titles:")
for title, count in tech_titles.most_common(20):
    print(f"  {count:5d}: {title}")
print(f"\nTop 20 NON-TECH titles:")
for title, count in non_tech_titles.most_common(20):
    print(f"  {count:5d}: {title}")
