"""
Quick test of is_non_tech_profile() against the 5 sample candidates we inspected.
Expected:
- CAND_0005649 (Senior Data Scientist at Sarvam AI) -> NOT non-tech (should pass)
- CAND_0055999 (Mechanical Engineer) -> IS non-tech (should fail)
- CAND_0027855 (Civil Engineer) -> IS non-tech (should fail)
- CAND_0037426 (Civil Engineer -> Infosys) -> IS non-tech (should fail)
- CAND_0095579 (Mechanical Engineer -> Dunder Mifflin) -> IS non-tech (should fail)
"""
import json, re, sys
sys.path.insert(0, '.')

# Import constants and function from rank.py
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

test_ids = ['CAND_0005649', 'CAND_0055999', 'CAND_0027855', 'CAND_0037426', 'CAND_0095579']
found = {}
with open('candidates.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        m = re.search(r'"candidate_id":\s*"(CAND_\d{7})"', line)
        if m and m.group(1) in test_ids:
            cand = json.loads(line)
            cid = m.group(1)
            found[cid] = cand
            if len(found) == len(test_ids):
                break

expected = {
    'CAND_0005649': False,  # Should NOT be non-tech (legitimate AI engineer)
    'CAND_0055999': True,   # Should be non-tech (Mechanical Engineer)
    'CAND_0027855': True,   # Should be non-tech (Civil Engineer)
    'CAND_0037426': True,   # Should be non-tech (Civil Engineer)
    'CAND_0095579': True,   # Should be non-tech (Mechanical Engineer)
}

print("=== Non-tech Profile Detection Test ===\n")
all_pass = True
for cid in test_ids:
    cand = found[cid]
    result = is_non_tech_profile(cand)
    exp = expected[cid]
    ok = result == exp
    if not ok:
        all_pass = False
    profile = cand.get('profile', {})
    title = profile.get('current_title', '?')
    jobs = [j.get('title', '?') for j in cand.get('career_history', [])]
    status = "[PASS]" if ok else "[FAIL]"
    print(f"{status} {cid}: '{title}' -> non_tech={result} (expected={exp})")
    print(f"       History titles: {jobs[:3]}")
    print()

print("=== OVERALL:", "ALL PASS" if all_pass else "SOME FAILURES ===")
