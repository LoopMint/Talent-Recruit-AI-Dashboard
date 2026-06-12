import streamlit as st
import pandas as pd
import re
import pdfplumber
from PyPDF2 import PdfReader
import docx

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------
# APP SETUP
# ----------------------------
st.set_page_config(page_title="AI Recruiter ATS", layout="wide")
st.title("🧠 AI Recruiter ATS — Semantic Matching + Next Phase Selection")

# ----------------------------
# MODEL
# ----------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# ----------------------------
# SKILLS
# ----------------------------
SKILLS = [
    "python", "java", "javascript", "react", "node", "sql",
    "aws", "docker", "kubernetes",
    "machine learning", "nlp", "deep learning",
    "fastapi", "django", "flask"
]

# ----------------------------
# CLEAN TEXT
# ----------------------------
def clean(text):
    return re.sub(r'\s+', ' ', text or "").strip()

# ----------------------------
# FILE READERS
# ----------------------------
def read_pdf(file):
    text = ""
    try:
        with pdfplumber.open(file) as pdf:
            for p in pdf.pages:
                t = p.extract_text()
                if t:
                    text += t + " "
    except:
        pass

    if not text:
        try:
            file.seek(0)
            reader = PdfReader(file)
            for p in reader.pages:
                text += p.extract_text() or ""
        except:
            pass

    return clean(text)

def read_docx(file):
    doc = docx.Document(file)
    return clean(" ".join([p.text for p in doc.paragraphs]))

def read_txt(file):
    return clean(file.read().decode("utf-8", errors="ignore"))

# ----------------------------
# EXPERIENCE
# ----------------------------
def get_exp(text):
    matches = re.findall(r"(\d+)\+?\s*(years|yrs)", text.lower())
    vals = [float(m[0]) for m in matches]
    return max(vals) if vals else 0

# ----------------------------
# SKILLS
# ----------------------------
def get_skills(text):
    t = text.lower()
    return list({s for s in SKILLS if s in t})

# ----------------------------
# PROFILE
# ----------------------------
def parse_cv(text, name):
    email = re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    phone = re.findall(r"\+?\d[\d\s-]{8,}\d", text)

    return {
        "name": name,
        "email": email[0] if email else "Not found",
        "experience": get_exp(text),
        "skills": get_skills(text),
        "text": text
    }

# ----------------------------
# SEMANTIC SCORE
# ----------------------------
def semantic(jd, cv):
    return float(cosine_similarity(
        model.encode([jd]),
        model.encode([cv])
    )[0][0])

# ----------------------------
# STAR RATING
# ----------------------------
def star(score):
    if score >= 0.85: return "★★★★★"
    if score >= 0.70: return "★★★★☆"
    if score >= 0.55: return "★★★☆☆"
    if score >= 0.40: return "★★☆☆☆"
    return "★☆☆☆☆"

# ----------------------------
# LOAD FILES
# ----------------------------
def load(files):
    profiles = []

    for f in files:
        if f.name.endswith(".pdf"):
            text = read_pdf(f)
        elif f.name.endswith(".docx"):
            text = read_docx(f)
        else:
            text = read_txt(f)

        profiles.append(parse_cv(text, f.name.split(".")[0]))

        with st.expander(f"📄 {f.name} preview"):
            st.write(text[:2000])

    return profiles

# ----------------------------
# MATCH ENGINE
# ----------------------------
def match(jd, profiles):

    jd_skills = set(get_skills(jd))
    results = []

    for p in profiles:
        sem = semantic(jd, p["text"])

        skill_match = len(set(p["skills"]) & jd_skills)
        skill_score = skill_match / max(len(jd_skills), 1)

        exp_score = min(p["experience"] / 10, 1)

        final = (0.65 * sem) + (0.20 * skill_score) + (0.15 * exp_score)

        results.append({
            "name": p["name"],
            "email": p["email"],
            "experience": p["experience"],
            "skills": ", ".join(p["skills"]),
            "score": round(final, 3),
            "stars": star(final),

            # HR CONTROL FIELDS
            "shortlisted": final >= 0.65,
            "remarks": ""
        })

    return pd.DataFrame(results)

# ----------------------------
# UI INPUTS
# ----------------------------
col1, col2 = st.columns(2)

with col1:
    jd = st.text_area("📌 Job Description", height=250)

with col2:
    files = st.file_uploader(
        "📄 Upload CVs",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )

# ----------------------------
# RUN
# ----------------------------
if st.button("🚀 Run AI Matching"):

    if not jd or not files:
        st.error("Upload CVs + Job Description")
        st.stop()

    profiles = load(files)
    df = match(jd, profiles)

    st.session_state.df = df

# ----------------------------
# RESULTS
# ----------------------------
if "df" in st.session_state:

    df = st.session_state.df.sort_values("score", ascending=False)

    st.subheader("🏆 Candidate Ranking (AI Semantic Match)")

    st.dataframe(df, use_container_width=True)

    # ----------------------------
    # NEXT PHASE CANDIDATES (EDITABLE)
    # ----------------------------

    st.subheader("🚀 Next Phase Candidates")

    edited_df = st.data_editor(
        df,
        use_container_width=True,
        num_rows="dynamic",
        column_config={
            "shortlisted": st.column_config.CheckboxColumn("Shortlisted")
        }
    )

    st.session_state.df = edited_df

    # ----------------------------
    # DOWNLOAD
    # ----------------------------
    csv = edited_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "📥 Download Final Selection",
        csv,
        "next_phase_candidates.csv",
        "text/csv"
    )

# ----------------------------
# FOOTER
# ----------------------------
st.markdown("---")
st.caption("AI ATS | Semantic Matching Engine | Next Phase Candidate Workflow")
