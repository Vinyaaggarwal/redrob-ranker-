# Redrob AI Candidate Ranking Engine 🤖

An intelligent, recruiter-centric candidate ranking system built for the Redrob Series A Founding Team AI Engineer role. 

This engine replaces primitive keyword-matching ATS filters with **dense semantic embeddings** (via local SentenceTransformers) combined with **structured multi-dimensional scoring rules** (representing hiring constraints, career progression, stability, and behavioral signals).

---

## 📋 System Blueprint & Approach

1. **Trap & Honeypot Filtering**: Implements a 4-rule deterministic filter that immediately blacklists the **76 honeypot candidates** in the dataset (assigning them a score of `-9999.0`).
2. **Two-Stage Coarse-to-Fine Matching**:
   - **Stage 1 (Retrieval)**: Fast dot-product cosine similarity search between the target Job Description (JD) and precomputed candidate embeddings (using `all-MiniLM-L6-v2` in float16) to pull the Top 2000 candidates.
   - **Stage 2 (Scoring)**: Evaluates detailed sub-scores (Skills Fit, Experience Fit, Project Relevance, Career Stability, and Behavioral Signals) for the Top 2000 candidates, executing in **under 3 seconds** on CPU.
3. **Deterministic Tie-Breaking**: Resolves matching scores by sorting candidates alphabetically by `candidate_id` ascending.

---

## 🚀 Setup & Execution Guide

### 1. Install Dependencies
Make sure you have Python installed, then run:
```bash
pip install -r requirements.txt
```
*(Dependencies include: `torch`, `sentence-transformers`, `numpy`, `streamlit`, `matplotlib`, `weasyprint`, and `pyyaml`)*

### 2. Precompute Candidate Embeddings (Offline)
This step encodes the 100,000 profiles in `candidates.jsonl` into the compressed dense representation `candidate_embeddings.npz` (float16, ~76MB) and saves their IDs to `candidate_ids.json`.
```bash
python precompute_embeddings.py
```
*Note: This runs locally on CPU and takes about 10–15 minutes depending on processor cores. You only need to run this once offline; the output files are checked into the repository so the server-side validator runs completely offline.*

### 3. Run the Ranking Pipeline (Reproduction Step)
Generate your Top 100 candidates ranked shortlist:
```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```
*Note: This completes in **under 3 seconds** on CPU with zero network dependencies, easily complying with the 5-minute hackathon constraint.*

### 4. Validate the Output CSV
Confirm that your generated CSV complies with all competition constraints (exactly 100 rows, proper columns, non-increasing scores, deterministic tie-breaks):
```bash
python validate_submission.py submission.csv
```

### 5. Launch the Recruiter Dashboard Sandbox
Run the interactive Streamlit application to explore matched profiles, adjust sliders, and generate interview questions:
```bash
streamlit run app.py
```

### 6. Generate the Slide Deck
Compile the project design presentation into `approach_deck.pdf`:
```bash
python generate_deck.py
```

---

## 📊 Dimension & Weights Breakdown

$$\text{Final Score} = 0.30 \times \text{Semantic} + 0.20 \times \text{Skills} + 0.25 \times \text{Experience} + 0.10 \times \text{Projects} + 0.10 \times \text{Velocity} + 0.05 \times \text{Behavioral}$$

* **Semantic Match (30%)**: Direct semantic match from JD vs candidate profile embeddings.
* **Skill Overlap (20%)**: Core & preferred skills checked with proficiencies and durations. Penalizes keyword-stuffers. Hard reduces score for non-tech profiles.
* **Experience Fit (25%)**: Evaluates years of experience target (5–9 yrs) and applied AI/ML years in career history. **Hard disqualifies Tier-5 profiles** (Civil/Mechanical/HR/non-tech career histories stuffed with AI keywords).
* **Project Relevance (10%)**: Durations of jobs focused on building search/retrieval systems.
* **Career Velocity & Stability (10%)**: Penalizes job-hopping (average tenure < 18 months) and disqualifies service-firm-only careers.
* **Behavioral Signals (5%)**: Multiplier of availability, response rate, and login frequency.

### Tier-5 Non-Tech Profile Disqualifier

The dataset contains "Tier-5 disguised" candidates: Civil Engineers, Mechanical Engineers, HR Managers, Graphic Designers, Sales Executives etc. who keyword-stuff their skills list with AI terms (FAISS, LangChain, Pinecone, etc.) to fool semantic similarity models.

Our `is_non_tech_profile()` function hard-disqualifies these:
- Returns `True` if the candidate has **zero tech roles** (by job title) in their career history
- Returns `True` if ≥80% of their career history is in non-tech roles AND their current title is non-tech
- Assigns near-zero experience score (`0.02`) and multiplies skills score by `0.1x` for confirmed non-tech profiles
