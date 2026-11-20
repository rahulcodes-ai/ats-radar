import math
import re
from collections import Counter
import streamlit as st
from pypdf import PdfReader

# Page Config
st.set_page_config(page_title="ATS-Radar | Resume vs JD Engine", page_icon="🎯", layout="wide")

# Custom CSS for clean keyword pills
st.markdown(
    """
    <style>
    .pill-match {
        display: inline-block; padding: 4px 10px; margin: 3px;
        border-radius: 14px; background-color: #123c26; color: #7ef2b1;
        font-size: 0.85rem; font-weight: 600; border: 1px solid #1f6f46;
    }
    .pill-miss {
        display: inline-block; padding: 4px 10px; margin: 3px;
        border-radius: 14px; background-color: #42181d; color: #ff8e99;
        font-size: 0.85rem; font-weight: 600; border: 1px solid #8a2c37;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Stopwords & Curated High-Signal Dictionaries
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
    "by", "from", "as", "is", "was", "are", "were", "be", "been", "being", "have",
    "has", "had", "do", "does", "did", "will", "would", "shall", "should", "can",
    "could", "may", "might", "must", "that", "this", "these", "those", "it", "its",
    "we", "you", "your", "our", "they", "their", "them", "he", "she", "his", "her",
    "who", "which", "what", "when", "where", "why", "how", "all", "any", "both",
    "each", "more", "most", "other", "some", "such", "no", "nor", "not", "only",
    "own", "same", "so", "than", "too", "very", "just", "about", "into", "through",
    "during", "before", "after", "above", "below", "between", "under", "again",
    "further", "then", "once", "here", "there", "also", "using", "used", "work",
    "working", "experience", "years", "year", "role", "team", "skills", "strong",
    "good", "ability", "knowledge", "looking", "seeking", "join", "company", "job",
    "requirements", "responsibilities", "including", "across", "within", "etc",
}

TECH_SKILLS = {
    "python", "java", "c++", "c#", "javascript", "typescript", "go", "rust", "sql",
    "nosql", "postgresql", "mysql", "mongodb", "redis", "docker", "kubernetes", "aws",
    "gcp", "azure", "git", "github", "linux", "bash", "react", "next.js", "node.js",
    "express", "django", "flask", "fastapi", "streamlit", "html", "css", "tailwind",
    "machine learning", "deep learning", "nlp", "computer vision", "data science",
    "data analysis", "pandas", "numpy", "scikit-learn", "pytorch", "tensorflow",
    "opencv", "power bi", "tableau", "excel", "rest api", "graphql", "microservices",
    "ci/cd", "agile", "scrum", "system design", "data structures", "algorithms",
    "oop", "unit testing", "pytest", "selenium", "playwright", "spark", "hadoop",
    "kafka", "airflow", "llm", "rag", "prompt engineering", "cybersecurity",
    "networking", "cloud computing", "devops", "terraform", "jenkins", "figma",
}

ACTION_VERBS = {
    "architected", "automated", "built", "created", "delivered", "deployed",
    "designed", "developed", "engineered", "enhanced", "executed", "implemented",
    "improved", "increased", "launched", "led", "migrated", "optimized", "reduced",
    "refactored", "resolved", "scaled", "shipped", "spearheaded", "streamlined",
}

WEAK_BUZZWORDS = {
    "hardworking", "team player", "go-getter", "synergy", "detail-oriented",
    "self-motivated", "think outside the box", "passionate", "dynamic", "rockstar",
}

SAMPLE_RESUME = """
Rahul Kumar | Software Engineer (Python & Backend)
Email: rahul.dev@example.com | github.com/rahulcodes-ai | linkedin.com/in/rahul-dev

SUMMARY
Computer Science graduate skilled in Python, SQL, and building automated web utilities.

PROJECTS
DocAudit AI — Smart PDF Auditor
- Built a Python and Streamlit web app to parse multi-page PDFs with 99% extraction accuracy.
- Automated report generation, reducing manual document review time by 65%.
- Deployed live on cloud with clean modular GitHub workflows.

Data Pipeline Script
- Engineered a NumPy and Pandas script processing 50,000+ rows in under 3 seconds.
- Optimized SQL queries in PostgreSQL, cutting query latency by 40%.

SKILLS
Python, SQL, PostgreSQL, Git, GitHub, Linux, Pandas, NumPy, Streamlit, REST API, OOP, Data Structures, Algorithms
"""

SAMPLE_JD = """
Software Engineer — Python Backend
We are looking for a Python Developer to design, build, and scale backend APIs and data tools.
Requirements:
- Strong proficiency in Python, SQL, PostgreSQL, and REST API development.
- Hands-on experience with FastAPI or Django, Docker, and AWS cloud services.
- Solid foundation in Data Structures, Algorithms, Git, and Linux.
- Experience with Redis, CI/CD pipelines, and pytest unit testing is a strong plus.
- Ability to optimize system performance and ship reliable production code.
"""


def extract_pdf_text(uploaded_file):
    reader = PdfReader(uploaded_file)
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def tokenize(text):
    words = re.findall(r"[a-zA-Z0-9+#\.]+", text.lower())
    return [w for w in words if len(w) > 1 and w not in STOPWORDS]


def compute_tfidf_cosine(text1, text2):
    """Pure-Python TF-IDF Cosine Similarity (zero external ML libraries)."""
    t1, t2 = tokenize(text1), tokenize(text2)
    if not t1 or not t2:
        return 0.0
    c1, c2 = Counter(t1), Counter(t2)
    vocab = set(c1.keys()) | set(c2.keys())

    # Smooth IDF across the 2-document corpus + TF weighting
    vec1, vec2 = {}, {}
    for word in vocab:
        df = (1 if word in c1 else 0) + (1 if word in c2 else 0)
        idf = math.log((2 + 1) / (df + 1)) + 1.0
        vec1[word] = (c1[word] / len(t1)) * idf
        vec2[word] = (c2[word] / len(t2)) * idf

    dot = sum(vec1[w] * vec2[w] for w in vocab)
    norm1 = math.sqrt(sum(v * v for v in vec1.values()))
    norm2 = math.sqrt(sum(v * v for v in vec2.values()))
    return (dot / (norm1 * norm2)) if (norm1 and norm2) else 0.0


def extract_target_keywords(jd_text):
    """Extracts known tech skills + top frequent domain terms from JD."""
    jd_lower = jd_text.lower()
    found_skills = set()
    for skill in TECH_SKILLS:
        pattern = r"(?<!\w)" + re.escape(skill) + r"(?!\w)"
        if re.search(pattern, jd_lower):
            found_skills.add(skill)

    # Also add top 12 meaningful words from JD if not already covered
    tokens = tokenize(jd_text)
    for word, _ in Counter(tokens).most_common(12):
        if len(word) >= 3 and not word.isdigit():
            found_skills.add(word)
    return sorted(found_skills)


def audit_resume(resume_text, jd_text):
    res_lower = resume_text.lower()
    target_kw = extract_target_keywords(jd_text)

    matched, missing = [], []
    for kw in target_kw:
        pattern = r"(?<!\w)" + re.escape(kw) + r"(?!\w)"
        if re.search(pattern, res_lower):
            matched.append(kw)
        else:
            missing.append(kw)

    kw_score = (len(matched) / len(target_kw) * 100) if target_kw else 0.0
    tfidf_score = compute_tfidf_cosine(resume_text, jd_text) * 100
    overall_score = round(kw_score * 0.65 + tfidf_score * 0.35, 1)

    # Structural & Impact Signals
    words = resume_text.split()
    word_count = len(words)
    metrics_found = re.findall(r"\b\d+(?:\.\d+)?\s*(?:%|x|k|m|\+|ms|sec|seconds|hours|users|rows)\b", res_lower)
    verbs_found = sorted({v for v in ACTION_VERBS if re.search(rf"\b{v}\b", res_lower)})
    buzzwords_found = sorted({b for b in WEAK_BUZZWORDS if b in res_lower})
    has_email = bool(re.search(r"[\w\.-]+@[\w\.-]+\.\w+", resume_text))
    has_links = bool(re.search(r"(github\.com|linkedin\.com|https?://)", res_lower))

    return {
        "overall": overall_score,
        "kw_score": round(kw_score, 1),
        "tfidf_score": round(tfidf_score, 1),
        "matched": matched,
        "missing": missing,
        "word_count": word_count,
        "metrics": metrics_found,
        "verbs": verbs_found,
        "buzzwords": buzzwords_found,
        "has_email": has_email,
        "has_links": has_links,
    }


# --- UI HEADER ---
st.title("🎯 ATS-Radar — Zero-API Resume vs. JD Engine")
st.caption("Deterministic ATS simulator built with pure Python math (TF-IDF Cosine Similarity + N-Gram Skill Gap + Impact Audit). Zero API keys. 100% private.")

# One-click Sample Demo Toggle
if "use_sample" not in st.session_state:
    st.session_state.use_sample = False

col_demo, _ = st.columns([1, 3])
with col_demo:
    if st.button("⚡ Load Sample Resume & JD (1-Click Demo)"):
        st.session_state.use_sample = True

# Input Layout
left, right = st.columns(2)
resume_text = ""

with left:
    st.subheader("1️⃣ Your Resume")
    pdf_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
    if pdf_file:
        resume_text = extract_pdf_text(pdf_file)
        st.success(f"Extracted {len(resume_text.split())} words from {pdf_file.name}")
    else:
        default_res = SAMPLE_RESUME.strip() if st.session_state.use_sample else ""
        resume_text = st.text_area("Or paste Resume text directly:", value=default_res, height=260)

with right:
    st.subheader("2️⃣ Target Job Description (JD)")
    default_jd = SAMPLE_JD.strip() if st.session_state.use_sample else ""
    jd_text = st.text_area("Paste the Job Description here:", value=default_jd, height=345)

# Run Audit
if st.button("🚀 Run ATS Scan", type="primary", use_container_width=True):
    if not resume_text.strip() or not jd_text.strip():
        st.warning("Please provide both a Resume and a Job Description (or click 'Load Sample Demo' above).")
    else:
        res = audit_resume(resume_text, jd_text)

        st.markdown("---")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Overall ATS Score", f"{res['overall']}%")
        m2.metric("Skill / Keyword Match", f"{res['kw_score']}%")
        m3.metric("TF-IDF Similarity", f"{res['tfidf_score']}%")
        m4.metric("Quantified Metrics", len(res["metrics"]))

        st.progress(min(int(res["overall"]), 100))

        # Keyword Gap Breakdown
        k1, k2 = st.columns(2)
        with k1:
            st.subheader(f"✅ Matched Keywords ({len(res['matched'])})")
            if res["matched"]:
                st.markdown("".join(f'<span class="pill-match">{k}</span>' for k in res["matched"]), unsafe_allow_html=True)
            else:
                st.info("No exact keyword matches found.")

        with k2:
            st.subheader(f"❌ Missing Critical Keywords ({len(res['missing'])})")
            if res["missing"]:
                st.markdown("".join(f'<span class="pill-miss">{k}</span>' for k in res["missing"]), unsafe_allow_html=True)
            else:
                st.success("Zero missing keywords — strong match!")

        # Deterministic Health Checks
        st.markdown("---")
        st.subheader("🩺 Deterministic Resume Health Audit")
        c1, c2 = st.columns(2)
        with c1:
            wc = res["word_count"]
            if 350 <= wc <= 800:
                st.success(f"**Word Count:** {wc} words (Ideal ATS range: 350–800)")
            else:
                st.warning(f"**Word Count:** {wc} words (Aim for 350–800 words for a 1-page resume)")

            if res["has_email"] and res["has_links"]:
                st.success("**Contact & Portfolio Links:** Email + GitHub/LinkedIn detected")
            else:
                st.warning("**Contact & Links:** Make sure both Email and GitHub/LinkedIn URL are clearly visible")

        with c2:
            if len(res["verbs"]) >= 5:
                st.success(f"**Strong Action Verbs ({len(res['verbs'])}):** {', '.join(res['verbs'][:8])}")
            else:
                st.warning(f"**Strong Action Verbs ({len(res['verbs'])}):** Add stronger verbs like *Engineered, Optimized, Deployed, Scaled*")

            if res["buzzwords"]:
                st.error(f"**Weak Cliché Buzzwords Found:** {', '.join(res['buzzwords'])} — replace with proof & numbers.")
            else:
                st.success("**Cliché Check:** Zero weak fluff buzzwords detected.")

        # Downloadable Action Plan
        report_md = f"""# ATS-Radar Audit Report
- **Overall ATS Score:** {res['overall']}%
- **Keyword Match Rate:** {res['kw_score']}%
- **TF-IDF Similarity:** {res['tfidf_score']}%
- **Word Count:** {res['word_count']}

## Matched Keywords
{', '.join(res['matched']) or 'None'}

## Missing Keywords (Add these honestly if you have the skill)
{', '.join(res['missing']) or 'None'}

## Actionable Fixes
- Quantified metrics detected: {len(res['metrics'])} ({', '.join(res['metrics']) or 'None — add %, $, or time saved'})
- Weak buzzwords to remove: {', '.join(res['buzzwords']) or 'None'}
"""
        st.download_button("⬇️ Download ATS Action Plan (.md)", report_md, file_name="ats_radar_report.md")
