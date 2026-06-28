"""
parse_jd.py - Generic Requirement Extractor (Scenario B)
=========================================================
Reads job_description.txt and extracts structured hiring requirements
using ONLY local, offline methods:
  1. Regex -- extracts experience ranges, location, notice period
  2. Tech Taxonomy Dictionary -- detects which skills appear in the JD text
  3. Semantic Section Splitter -- separates must-have vs nice-to-have sections

Output: parsed_jd.json -- loaded by rank.py at runtime (zero network calls).

Run this script whenever the JD changes:
    python parse_jd.py
    python parse_jd.py --jd path/to/custom_jd.txt
"""

import os
import re
import json
import argparse

# ==============================================================================
# TECH TAXONOMY DICTIONARY
# A large, general-purpose dictionary mapping skill categories to their synonyms.
# When the JD changes, the parser automatically detects which entries are relevant.
# ==============================================================================

TECH_TAXONOMY = {
    # AI / ML Core
    "embeddings": [
        "embeddings", "embedding", "dense retrieval", "sentence-transformers",
        "sentence transformers", "bge", "e5", "openai embeddings", "text embeddings"
    ],
    "vector_databases": [
        "vector database", "vector db", "faiss", "pinecone", "milvus", "qdrant",
        "weaviate", "chroma", "chromadb", "opensearch", "elasticsearch", "vespa"
    ],
    "rag": [
        "rag", "retrieval augmented generation", "retrieval-augmented",
        "information retrieval", "ir system", "search engine"
    ],
    "hybrid_search": [
        "hybrid search", "bm25", "sparse search", "keyword search", "lexical search",
        "dense sparse", "hybrid retrieval"
    ],
    "llm_finetuning": [
        "fine-tuning", "fine tuning", "finetuning", "lora", "qlora", "peft",
        "sft", "rlhf", "pre-training", "pretraining", "instruction tuning",
        "deepspeed", "fsdp", "transformers"
    ],
    "llm_frameworks": [
        "llm", "large language model", "gpt", "chatgpt", "claude", "gemini",
        "llama", "mistral", "falcon", "huggingface", "hugging face", "openai",
        "anthropic", "langchain", "llamaindex"
    ],
    "learning_to_rank": [
        "learning to rank", "ltr", "neural ranking", "pointwise", "pairwise",
        "listwise", "lambdamart", "ranknet"
    ],
    "ml_frameworks": [
        "pytorch", "tensorflow", "keras", "jax", "sklearn", "scikit-learn",
        "xgboost", "lightgbm", "catboost"
    ],
    "nlp": [
        "nlp", "natural language processing", "bert", "roberta", "t5", "bart",
        "named entity recognition", "ner", "text classification", "sentiment analysis",
        "question answering", "summarization", "tokenization"
    ],
    "computer_vision": [
        "computer vision", "image recognition", "object detection",
        "convolutional neural network", "cnn", "yolo", "resnet", "vit"
    ],
    "evaluation_frameworks": [
        "ndcg", "mrr", "recall@", "precision@", "f1", "a/b test",
        "a/b testing", "ab testing", "offline evaluation", "online evaluation",
        "evaluation framework", "benchmark"
    ],
    "python": ["python", "django", "flask", "fastapi", "pydantic"],
    "java": ["java", "spring", "spring boot", "maven", "gradle"],
    "javascript": ["javascript", "typescript", "node.js", "nodejs", "react",
                   "next.js", "vue", "angular"],
    "go": ["golang"],
    "cpp": ["c++", "cpp"],
    "rust": ["rust lang", "rust programming"],
    "scala": ["scala", "akka"],
    "kotlin": ["kotlin"],
    "swift": ["swift ios"],
    "rest_api": ["rest api", "restful", "api design", "graphql", "grpc",
                 "microservices", "backend"],
    "sql": ["sql", "postgresql", "postgres", "mysql", "sqlite", "relational database"],
    "nosql": ["nosql", "mongodb", "cassandra", "dynamodb", "redis", "couchdb"],
    "aws": ["aws", "amazon web services", "ec2", "s3", "sagemaker", "lambda",
            "eks", "ecs", "rds", "cloudformation"],
    "gcp": ["gcp", "google cloud", "vertex ai", "bigquery", "gke", "cloud run"],
    "azure": ["azure", "microsoft azure", "azure ml", "aks"],
    "docker": ["docker", "dockerfile", "containerization"],
    "kubernetes": ["kubernetes", "k8s", "helm", "kubectl"],
    "distributed_systems": [
        "distributed", "apache spark", "ray", "dask", "flink",
        "kafka", "messaging queue", "pub/sub", "event-driven"
    ],
    "mlops": [
        "mlops", "ml pipeline", "model serving", "model deployment",
        "feature store", "experiment tracking", "mlflow", "kubeflow",
        "airflow", "model monitoring", "drift detection"
    ],
    "devops": ["devops", "sre", "site reliability", "terraform", "ansible",
               "github actions", "jenkins"],
    "open_source": [
        "open-source", "open source", "github contributions",
        "pull requests", "open source contributions"
    ],
    "hr_tech": [
        "hr-tech", "hr tech", "recruiting tech", "recruitment",
        "talent acquisition", "marketplace", "talent intelligence",
        "applicant tracking", "ats"
    ],
}

# Patterns that signal a "must-have" section in the JD
REQUIRED_SIGNALS = [
    r"\b(must|required|absolutely|mandatory|non-negotiable)\b",
    r"things you absolutely",
    r"what you.{0,20}need",
    r"requirements?:",
    r"you must",
    r"you need",
]

# Patterns that signal a "nice-to-have" section in the JD
PREFERRED_SIGNALS = [
    r"\b(preferred|nice to have|bonus|ideally|ideal|helpful|desirable)\b",
    r"things we.{0,20}like",
    r"good to have",
    r"would be a plus",
    r"like you to have",
]

# Patterns that signal a disqualifier section in the JD
DISQUALIFIER_SIGNALS = [
    r"\b(do not want|not a fit|disqualif|explicitly not|won.t move forward|will not move forward)\b",
    r"things we.{0,20}(not|don.t|do not) want",
]


def extract_experience_range(jd_text):
    """Extract years-of-experience range from JD using regex."""
    # "5-9 years" / "5 to 9 years" / "5-9 yrs"
    range_match = re.search(
        r'(\d+)\s*(?:to|-|to)\s*(\d+)\s*(?:years?|yrs?)',
        jd_text, re.IGNORECASE
    )
    if range_match:
        return int(range_match.group(1)), int(range_match.group(2))

    # "3+ years"
    plus_match = re.search(
        r'(\d+)\+\s*(?:years?|yrs?)',
        jd_text, re.IGNORECASE
    )
    if plus_match:
        min_years = int(plus_match.group(1))
        return min_years, min_years + 6

    return 3, 10  # fallback


def extract_location(jd_text):
    """Extract location from JD text."""
    loc_match = re.search(
        r'[Ll]ocation[:\s]+([A-Za-z\s,/]+?)(?:\n|\.|\()',
        jd_text
    )
    if loc_match:
        return loc_match.group(1).strip()
    return ""


def extract_notice_period(jd_text):
    """Extract preferred notice period from JD text."""
    # "sub-30-day notice" or "30-day notice"
    sub_match = re.search(r'sub[- ]?(\d+)[- ]?day', jd_text, re.IGNORECASE)
    if sub_match:
        return int(sub_match.group(1))
    notice_match = re.search(
        r'(\d+)[- ]?day\s*notice',
        jd_text, re.IGNORECASE
    )
    if notice_match:
        return int(notice_match.group(1))
    return 90  # default


def split_jd_into_sections(jd_text):
    """Split JD text into requirement sections by scanning each sentence."""
    sections = {"required": [], "preferred": [], "disqualifier": [], "context": []}
    sentences = re.split(r'(?<=[.!?\n])\s+', jd_text)
    current_section = "context"

    for sentence in sentences:
        s_lower = sentence.lower()
        if any(re.search(p, s_lower) for p in REQUIRED_SIGNALS):
            current_section = "required"
        elif any(re.search(p, s_lower) for p in PREFERRED_SIGNALS):
            current_section = "preferred"
        elif any(re.search(p, s_lower) for p in DISQUALIFIER_SIGNALS):
            current_section = "disqualifier"
        sections[current_section].append(sentence.strip())

    return sections


def extract_skills_from_text(text, taxonomy):
    """Find all taxonomy skill categories mentioned in the text."""
    text_lower = text.lower()
    found = set()
    for category, synonyms in taxonomy.items():
        for syn in synonyms:
            pattern = re.escape(syn)
            if re.search(pattern, text_lower):
                found.add(category)
                break
    return found


def extract_domain(jd_text):
    """Infer the primary role domain from JD text."""
    text_lower = jd_text.lower()
    domain_signals = {
        "AI/ML":             ["machine learning", "deep learning", "nlp", "embeddings",
                              "retrieval", "ranking", "llm", "ai engineer", "ml engineer"],
        "Data Engineering":  ["data pipeline", "data engineer", "etl", "spark", "warehouse"],
        "DevOps":            ["devops", "sre", "infrastructure", "terraform", "ci/cd"],
        "Mobile":            ["android", "ios", "mobile", "flutter", "react native"],
        "Frontend":          ["frontend", "react", "vue", "angular", "ui developer"],
        "Backend":           ["backend", "api", "microservice", "database", "server-side"],
    }
    scores = {d: sum(1 for s in sigs if s in text_lower) for d, sigs in domain_signals.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "General"


def parse_jd(jd_path="job_description.txt"):
    """
    Main parser: reads JD text file, extracts structured requirements offline.
    Returns a dict ready to be JSON-serialized.
    """
    if not os.path.exists(jd_path):
        raise FileNotFoundError(f"JD file not found: {jd_path}")

    print(f"[parse_jd] Reading: {jd_path}")
    with open(jd_path, "r", encoding="utf-8") as f:
        jd_text = f.read()

    print("[parse_jd] Splitting JD into requirement sections...")
    sections = split_jd_into_sections(jd_text)

    required_text  = " ".join(sections["required"])
    preferred_text = " ".join(sections["preferred"])
    disq_text      = " ".join(sections["disqualifier"])

    # If the required section is empty, fall back to full JD text
    required_skills  = extract_skills_from_text(required_text or jd_text, TECH_TAXONOMY)
    preferred_skills = extract_skills_from_text(preferred_text, TECH_TAXONOMY)
    disq_skills      = extract_skills_from_text(disq_text, TECH_TAXONOMY)
    preferred_skills -= required_skills  # deduplicate

    exp_min, exp_max  = extract_experience_range(jd_text)
    location          = extract_location(jd_text)
    notice_period     = extract_notice_period(jd_text)
    domain            = extract_domain(jd_text)

    # Build mappings (category -> synonym list) for rank.py to use
    core_skills_mapping = {
        skill: TECH_TAXONOMY.get(skill, [skill])
        for skill in sorted(required_skills)
    }
    preferred_skills_mapping = {
        skill: TECH_TAXONOMY.get(skill, [skill])
        for skill in sorted(preferred_skills)
    }

    result = {
        "experience_min":               exp_min,
        "experience_max":               exp_max,
        "domain":                       domain,
        "location":                     location,
        "notice_period_days":           notice_period,
        "required_skill_categories":    sorted(required_skills),
        "preferred_skill_categories":   sorted(preferred_skills),
        "disqualifier_skill_categories":sorted(disq_skills),
        "core_skills_mapping":          core_skills_mapping,
        "preferred_skills_mapping":     preferred_skills_mapping,
        "debug_sections": {
            "required_sentences":     sections["required"][:5],
            "preferred_sentences":    sections["preferred"][:5],
            "disqualifier_sentences": sections["disqualifier"][:5],
        }
    }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Offline JD parser: extracts requirements from any job_description.txt"
    )
    parser.add_argument("--jd",  default="job_description.txt",
                        help="Path to the job description text file")
    parser.add_argument("--out", default="parsed_jd.json",
                        help="Output path for the structured JSON")
    args = parser.parse_args()

    requirements = parse_jd(args.jd)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(requirements, f, indent=2)

    print(f"\n[parse_jd] Done! Saved to: {args.out}")
    print(f"  Domain           : {requirements['domain']}")
    print(f"  Experience Range : {requirements['experience_min']}-{requirements['experience_max']} years")
    print(f"  Location         : {requirements['location']}")
    print(f"  Notice Period    : up to {requirements['notice_period_days']} days")
    print(f"  Required Skills  : {', '.join(requirements['required_skill_categories'])}")
    print(f"  Preferred Skills : {', '.join(requirements['preferred_skill_categories'])}")
