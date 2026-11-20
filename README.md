# 🎯 ATS-Radar — Zero-API Resume vs. JD Engine

A deterministic, zero-API Applicant Tracking System (ATS) simulator built in **pure Python and Streamlit**. Upload any PDF resume and paste a Job Description to compute **TF-IDF Cosine Similarity**, uncover missing critical skills, and audit resume impact metrics—100% privately with zero external AI APIs.

### 🚀 Live Demo
👉 **[Try ATS-Radar Live](PASTE_YOUR_LIVE_URL_HERE)** *(Includes a 1-Click Sample Demo!)*

### ✨ Key Features
- **⚡ Zero API / Zero Login:** Runs entirely on pure Python math—instant results, zero cost, 100% privacy.
- **📊 TF-IDF Cosine Similarity:** Custom-built vector similarity engine without heavy ML dependencies.
- **🔍 Skill & Keyword Gap Radar:** Extracts tech skills and domain terms from the JD and highlights matched vs. missing keywords.
- **🩺 Deterministic Health Audit:** Checks word count, quantified metrics (`%`, `x`, `k`, time saved), strong action verbs, contact links, and cliché buzzwords.
- **⬇️ Action Plan Export:** Download a clean Markdown audit report in one click.

### 🛠️ Tech Stack
- **Language:** Python (Pure Math + Regex)
- **Frontend:** Streamlit
- **PDF Parsing:** `pypdf`

### 💻 Run Locally
```bash
git clone [https://github.com/rahulcodes-ai/ats-radar.git](https://github.com/rahulcodes-ai/ats-radar.git)
cd ats-radar
pip install -r requirements.txt
streamlit run app.py
