import json, re

# Look at top 5 ranked candidates' actual profiles
top_ids = ['CAND_0005649', 'CAND_0055999', 'CAND_0027855', 'CAND_0037426', 'CAND_0095579']
found = {}
with open('candidates.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        m = re.search(r'"candidate_id":\s*"(CAND_\d{7})"', line)
        if m and m.group(1) in top_ids:
            cand = json.loads(line)
            cid = m.group(1)
            profile = cand.get('profile', {})
            skills = [s.get('name') for s in cand.get('skills', [])]
            hist = [{'title': j.get('title'), 'company': j.get('company'), 'dur': j.get('duration_months')} for j in cand.get('career_history', [])]
            found[cid] = {
                'title': profile.get('current_title'), 
                'yoe': profile.get('years_of_experience'), 
                'skills': skills[:10], 
                'history': hist[:4],
                'summary': profile.get('summary', '')[:200],
                'headline': profile.get('headline', '')
            }
            if len(found) == len(top_ids):
                break

for cid, info in found.items():
    print(f'{cid}: {info["title"]} ({info["yoe"]} yrs)')
    print(f'  Headline: {info["headline"]}')
    print(f'  Summary: {info["summary"]}')
    print(f'  Skills: {info["skills"]}')
    print(f'  History: {info["history"]}')
    print()
