# ============================
#  HR ASSISTANT PORTAL (NO LLM)
# ============================

import os
import re
import textwrap
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="HR Assistant Portal",
    layout="wide",
    page_icon="💼"
)

# ---------------------------------------------------------
# GLOBAL THEME (PASTEL UI)
# ---------------------------------------------------------
def inject_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

        .stApp {
            background-color:#F8FAFC !important;
            font-family:'Inter', sans-serif;
            color:#1E293B;
        }

        section[data-testid="stSidebar"] {
            background-color:#1E293B !important;
            color:white;
        }
        section[data-testid="stSidebar"] * {
            color:white !important;
        }

        .pro-card {
            background:white;
            border-radius:12px;
            padding:20px;
            box-shadow:0 2px 6px rgba(0,0,0,0.1);
            border:1px solid #E2E8F0;
        }

        .chat-bubble-user {
            background:#2563EB;
            padding:10px 18px;
            border-radius:14px 14px 4px 14px;
            color:white;
            margin-bottom:6px;
            max-width:80%;
            margin-left:auto;
        }
        .chat-bubble-bot {
            background:#EEF2FF;
            padding:10px 18px;
            border-radius:14px 14px 14px 4px;
            color:#1E293B;
            border:1px solid #CBD5E1;
            max-width:80%;
            margin-bottom:6px;
        }

        .page-pill {
            background:#DBEAFE;
            padding:4px 10px;
            border-radius:12px;
            font-size:12px;
            color:#1E40AF;
            margin-right:6px;
            border:1px solid #BFDBFE;
        }
    </style>
    """, unsafe_allow_html=True)


inject_css()

# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------
DEFAULT_SESSION_KEYS = [
    "employees", "logged_in", "current_user",
    "chat_history", "pdf_chunks", "tfidf_matrix",
    "vectorizer", "pdf_name", "summary_sentences"
]

for key in DEFAULT_SESSION_KEYS:
    if key not in st.session_state:
        st.session_state[key] = None

if st.session_state.summary_sentences is None:
    st.session_state.summary_sentences = 5

# ---------------------------------------------------------
# EMPLOYEE DATABASE (STATIC)
# ---------------------------------------------------------
def create_employee_db():
    names = [
        "Nandini R","Rakesh Kumar","Aditi Sharma","Praveen Rao","Megha Joshi",
        "Rahul Verma","Sneha Iyer","Ankit Patel","Divya Nair","Kiran Singh",
        "Ayesha Khan","Vikram Desai","Pooja Gupta","Sanjay Kumar","Neha Jain",
        "Harish B","Swati Kulkarni","Aman Shah","Kavya R","Rohit S",
        "Anusha M","Karthik R","Shruti P","Abhishek T","Shreya G",
        "Manoj K","Ishita S","Lokesh V","Priya D","Arjun R"
    ]
    roles = ["Data Science Intern","ML Engineer","Backend Developer","Frontend Developer",
             "Data Analyst","DevOps Engineer","AI Research Intern","Full Stack Engineer"]
    curr_projects = ["AI HR Assistant","Customer 360","Sales Forecasting","Recommender",
                     "Fraud Detection","Chatbot System","Attrition Model","Doc Summarizer"]

    pend_proj = ["Onboarding System","Invoice Pipeline","Lead Scoring","Ticket Classifier",
                 "Supply Chain","HR Portal"]

    prev_proj = ["CRM Migration","ETL Modernization","Churn POC","AB Testing","Mix Model","Data Audit"]

    employees = {}
    for i in range(30):
        username = f"user{i+1:02d}"
        password = f"pass{i+1:02d}"

        employees[username] = {
            "username": username,
            "password": password,
            "name": names[i],
            "role": roles[i % len(roles)],
            "current_project": {
                "name": curr_projects[i % len(curr_projects)],
                "status": "In Progress",
                "client": "Internal" if i % 3 == 0 else "External"
            },
            "pending_projects": [
                pend_proj[(i+j) % len(pend_proj)] for j in range(2)
            ],
            "previous_projects": [
                prev_proj[(i+j) % len(prev_proj)] for j in range(3)
            ],
            "salary": {
                "basic": 50000 + i*1200,
                "hra": int((50000 + i*1200) * 0.4),
                "allowance": 8000 + i*250,
                "tax": int((50000 + i*1200)*1.4*0.18),
                "net": (50000 + i*1200)*1.4 + (8000+i*250) - int((50000+i*1200)*1.4*0.18)
            },
            "leaves": {
                "total": 18,
                "used": (i*2) % 18,
            }
        }
        employees[username]["leaves"]["pending"] = \
            employees[username]["leaves"]["total"] - employees[username]["leaves"]["used"]

        employees[username]["weekly_hours"] = {
            "Mon": 8+(i%3),"Tue": 7+(i%4),"Wed": 8.5,
            "Thu": 9,"Fri": 7.5+(i%2),"Sat": 0,"Sun": 0
        }

    return employees


if st.session_state.employees is None:
    st.session_state.employees = create_employee_db()

# ---------------------------------------------------------
# COMPANY FAQ (STATIC)
# ---------------------------------------------------------
COMPANY_FAQ = [
    {"keywords":["working hours","office hours"],"answer":"Our work hours are 9:30AM–6:30PM Mon–Fri."},
    {"keywords":["leave","vacation"],"answer":"Employees get 18 paid leaves annually."},
    {"keywords":["probation"],"answer":"Probation is 3 months."},
    {"keywords":["remote","wfh"],"answer":"WFH allowed 2 days/week."},
    {"keywords":["company","about"],"answer":"We are NovaMind AI, building enterprise AI systems."}
]

def answer_faq(q: str):
    q_low = q.lower()
    for item in COMPANY_FAQ:
        if any(k in q_low for k in item["keywords"]):
            return item["answer"]
    return None

# ---------------------------------------------------------
# PDF READING + CHUNKING
# ---------------------------------------------------------
def read_pdf(path):
    reader = PdfReader(path)
    pages = []
    for i,page in enumerate(reader.pages):
        txt = page.extract_text() or ""
        pages.append({"page_num":i+1,"text":txt})
    return pages

def chunk_pages(pages, chunk_size=900, overlap=200):
    chunks=[]
    for p in pages:
        clean = re.sub(r"\s+"," ",p["text"]).strip()
        if not clean:
            continue
        start=0
        while start < len(clean):
            chunk = clean[start:start+chunk_size]
            chunks.append({"page_num":p["page_num"],"text":chunk})
            start += chunk_size - overlap
    return chunks

def process_uploaded_pdf(uploaded):
    with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded.getvalue())
        tmp_path = tmp.name

    pages = read_pdf(tmp_path)
    chunks = chunk_pages(pages)

    texts=[c["text"] for c in chunks]
    vectorizer=TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform(texts)

    st.session_state.pdf_chunks = chunks
    st.session_state.vectorizer = vectorizer
    st.session_state.tfidf_matrix = tfidf
    st.session_state.pdf_name = uploaded.name

# ---------------------------------------------------------
# RAG: TF-IDF SEARCH
# ---------------------------------------------------------
def search_pdf_chunks(query, k=6):
    if st.session_state.vectorizer is None:
        return [], []

    vec = st.session_state.vectorizer.transform([query])
    sim = cosine_similarity(vec, st.session_state.tfidf_matrix)[0]
    idx = np.argsort(sim)[::-1][:k]

    chunks = [st.session_state.pdf_chunks[i] for i in idx]
    pages = sorted({c["page_num"] for c in chunks})
    return chunks, pages

def build_answer_from_pdf(query):
    """
    Pure extractive answer (no LLM).
    """
    chunks,pages = search_pdf_chunks(query, k=6)
    if not chunks:
        return "No relevant content found in this document.", []

    extracted=[]
    for c in chunks:
        text = re.sub(r"\s+"," ",c["text"])
        sentences = re.split(r'(?<=[.!?]) +', text)
        filtered=[s for s in sentences if len(s.split())>6]
        extracted.extend(filtered)

    extracted = extracted[:6]

    final = "Here are the most relevant points:\n\n"
    for s in extracted:
        final += f"- {s.strip()}\n"

    return final, pages

# ---------------------------------------------------------
# EXTRACTIVE PDF SUMMARY
# ---------------------------------------------------------
def summarize_pdf(n_sent=5):
    if st.session_state.pdf_chunks is None:
        return "Upload a PDF to generate summary.", []

    text = " ".join([c["text"] for c in st.session_state.pdf_chunks])
    text = re.sub(r"\s+"," ",text)
    sentences = re.split(r'(?<=[.!?]) +', text)
    sentences = [s for s in sentences if len(s.split())>6][:n_sent]

    summary = "Document Summary:\n\n" + "\n".join([f"- {s}" for s in sentences])
    return summary, list(range(1, min(6, len(st.session_state.pdf_chunks))+1))
# ---------------------------------------------------------
# EMPLOYEE-SPECIFIC QA (Offline)
# ---------------------------------------------------------
def answer_employee_specific_query(query, user):
    q = query.lower()

    # ➤ Leaves
    if "leave" in q or "leaves" in q or "vacation" in q:
        used = user["leaves"]["used"]
        pending = user["leaves"]["pending"]
        total = user["leaves"]["total"]
        return (
            f"You have **{pending} pending leaves**, and you have used **{used}** "
            f"out of **{total}** total leaves."
        )

    # ➤ Salary
    if "salary" in q or "ctc" in q or "pay" in q:
        s = user["salary"]
        gross = s["basic"] + s["hra"] + s["allowance"]
        return (
            f"Your **Net Salary: ₹{s['net']:,}**\n\n"
            f"Breakdown:\n"
            f"- Basic: ₹{s['basic']:,}\n"
            f"- HRA: ₹{s['hra']:,}\n"
            f"- Allowance: ₹{s['allowance']:,}\n"
            f"- Monthly Gross: ₹{gross:,}\n"
            f"- Tax: ₹{s['tax']:,}"
        )

    # ➤ Role
    if "my role" in q or "designation" in q or "position" in q:
        return f"Your role is **{user['role']}**."

    # ➤ Current project
    if "current project" in q or "working on" in q:
        cp = user["current_project"]
        return (
            f"You're currently working on **{cp['name']}**\n"
            f"- Status: {cp['status']}\n"
            f"- Client: {cp['client']}"
        )

    # ➤ Pending projects
    if "pending project" in q or "upcoming project" in q:
        return (
            "Here are your pending projects:\n" +
            "\n".join([f"- {p}" for p in user["pending_projects"]])
        )

    # ➤ Previous projects
    if "previous project" in q or "completed project" in q:
        return (
            "Your previous projects include:\n" +
            "\n".join([f"- {p}" for p in user["previous_projects"]])
        )

    return None



# ---------------------------------------------------------
# LOGIN PAGE
# ---------------------------------------------------------
def login_page():
    st.markdown("## 🔐 HR Assistant Login")

    st.markdown("""
        <div class='pro-card' style='background:#EFF6FF; border:1px solid #BFDBFE;'>
            <b>Demo Credentials</b><br>
            Username: <code>user01</code> to <code>user30</code><br>
            Password: <code>pass01</code> to <code>pass30</code>
        </div>
    """, unsafe_allow_html=True)

    col, _ = st.columns([1.3, 1])

    with col:
        st.markdown("<div class='pro-card'>", unsafe_allow_html=True)
        st.markdown("### Login to Continue")

        with st.form("login_form"):
            username = st.text_input("Username", placeholder="user01")
            password = st.text_input("Password", placeholder="pass01", type="password")
            submit = st.form_submit_button("Login", type="primary")

            if submit:
                emp = st.session_state.employees

                if username in emp and emp[username]["password"] == password:
                    st.session_state.logged_in = True
                    st.session_state.current_user = username

                    # Reset session per user
                    st.session_state.chat_history = []
                    st.session_state.pdf_chunks = None
                    st.session_state.vectorizer = None
                    st.session_state.tfidf_matrix = None
                    st.session_state.pdf_name = None

                    st.success(f"Welcome {emp[username]['name']}!")
                    st.rerun()
                else:
                    st.error("Invalid username or password.")

        st.markdown("</div>", unsafe_allow_html=True)



# ---------------------------------------------------------
# DASHBOARD PAGE
# ---------------------------------------------------------
def dashboard_page(user):
    st.markdown("## 🏠 Employee Dashboard")

    col1, col2, col3 = st.columns(3)

    # ---------------- Profile Card ----------------
    with col1:
        st.markdown("<div class='pro-card'>", unsafe_allow_html=True)
        st.markdown("**Employee Profile**")
        st.markdown(f"<h3 style='color:#1D4ED8'>{user['name']}</h3>", unsafe_allow_html=True)
        st.caption(user["role"])
        st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- Salary ----------------
    s = user["salary"]
    gross = s["basic"] + s["hra"] + s["allowance"]

    with col2:
        st.markdown("<div class='pro-card'>", unsafe_allow_html=True)
        st.markdown("**Net Salary**")
        st.markdown(f"<h2>₹{s['net']:,}</h2>", unsafe_allow_html=True)
        st.caption(f"Gross: ₹{gross:,}")
        st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- Leaves ----------------
    l = user["leaves"]
    with col3:
        st.markdown("<div class='pro-card'>", unsafe_allow_html=True)
        st.markdown("**Leave Balance**")
        st.markdown(f"<h2>{l['pending']} Days</h2>", unsafe_allow_html=True)
        st.caption(f"Used {l['used']} / {l['total']}")
        st.markdown("</div>", unsafe_allow_html=True)

    # =======================================================
    # PROJECTS SECTION
    # =======================================================
    colA, colB = st.columns([1.7, 1])

    # ---------------- Current & Past Projects ----------------
    with colA:
        st.markdown("<div class='pro-card'>", unsafe_allow_html=True)
        st.markdown("### 📌 Projects Overview")

        cp = user["current_project"]
        st.info(f"**Current Project:** {cp['name']}  \nStatus: {cp['status']} | Client: {cp['client']}")

        st.markdown("#### Pending Projects:")
        for p in user["pending_projects"]:
            st.markdown(f"- ⏳ {p}")

        st.markdown("#### Completed Projects:")
        for p in user["previous_projects"]:
            st.markdown(f"- ✅ {p}")

        st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- Activity Chart ----------------
    with colB:
        st.markdown("<div class='pro-card'>", unsafe_allow_html=True)
        st.markdown("### ⏱ Weekly Hours")

        df = pd.DataFrame({
            "Day": list(user["weekly_hours"].keys()),
            "Hours": list(user["weekly_hours"].values())
        })

        chart = (
            alt.Chart(df)
            .mark_bar(color="#3B82F6")
            .encode(
                x=alt.X("Day:N", sort=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]),
                y="Hours:Q"
            )
        )
        st.altair_chart(chart, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
# ---------------------------------------------------------
# QA PAGE (Offline RAG + Extractive Summaries)
# ---------------------------------------------------------

def qa_page():
    user = st.session_state.employees[st.session_state.current_user]

    st.markdown("## 🤖 HR AI Assistant")

    # ======================================================
    # TOP CONTROLS — PDF Upload + Summary settings
    # ======================================================
    col1, col2 = st.columns([1.4, 1])

    # ---------------- PDF Upload ----------------
    with col1:
        st.markdown("<div class='pro-card'>", unsafe_allow_html=True)
        st.markdown("### 📄 Upload Policy Document")

        uploaded = st.file_uploader("Upload PDF", type=["pdf"], key="pdf_uploader")

        if uploaded:
            process_uploaded_pdf(uploaded)
            st.success(f"✅ PDF **{uploaded.name}** processed successfully!")

        if st.session_state.pdf_name:
            st.caption(f"Currently using: **{st.session_state.pdf_name}**")
        else:
            st.caption("No PDF uploaded yet. I will answer using employee details and company FAQs.")

        st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- Summary Settings ----------------
    with col2:
        st.markdown("<div class='pro-card'>", unsafe_allow_html=True)
        st.markdown("### ⚙️ Summary Settings")

        st.session_state.summary_sentences = st.slider(
            "Sentences in summary:",
            2, 15,
            value=st.session_state.summary_sentences,
            help="Controls length of PDF summary."
        )

        if st.button("📌 Summarize PDF", type="primary"):
            if st.session_state.pdf_chunks:
                with st.spinner("🔍 Extracting key information..."):
                    summary, pages = summarize_pdf(st.session_state.summary_sentences)

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": summary,
                    "pages": pages
                })
                st.success("Summary added to chat.")
            else:
                st.warning("Upload a PDF first.")

        st.markdown("---")

        if st.button("🧹 Clear Chat"):
            st.session_state.chat_history = []
            st.success("Chat history cleared.")

        st.markdown("</div>", unsafe_allow_html=True)

    # ======================================================
    # CHAT INTERFACE
    # ======================================================
    st.markdown("### 💬 Conversation")

    chat_box = st.container()

    with chat_box:
        if not st.session_state.chat_history:
            st.markdown(
                "<p style='text-align:center;color:#9ca3af'>Ask about salary, "
                "leaves, projects, company policies, or your uploaded PDF.</p>",
                unsafe_allow_html=True
            )

        # Display chat
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(
                    f"<div class='chat-bubble-user'>{msg['content']}</div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<div class='chat-bubble-bot'>{msg['content']}</div>",
                    unsafe_allow_html=True
                )
                if msg.get("pages"):
                    pills = "".join(
                        f"<span class='page-pill'>Page {p}</span>"
                        for p in msg["pages"]
                    )
                    st.markdown(pills, unsafe_allow_html=True)

    # Placeholder for "AI is thinking..."
    thinking = st.empty()

    # ======================================================
    # USER INPUT (BOTTOM CHAT BOX)
    # ======================================================
    user_msg = st.chat_input("Ask your question here…")

    if user_msg:
        # Save user message
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_msg,
            "pages": []
        })

        # Thinking bubble
        thinking.markdown(
            "<div class='chat-bubble-bot'>🤖 Processing your request...<br><b>...</b></div>",
            unsafe_allow_html=True
        )

        reply = None
        pages = []

        # PRIORITY 1 — Employee answers
        emp_ans = answer_employee_specific_query(user_msg, user)
        if emp_ans:
            reply = emp_ans

        # PRIORITY 2 — FAQ
        elif answer_faq(user_msg):
            reply = answer_faq(user_msg)

        # PRIORITY 3 — PDF RAG
        elif st.session_state.pdf_chunks:
            if "summary" in user_msg.lower():
                reply, pages = summarize_pdf(st.session_state.summary_sentences)
            else:
                reply, pages = build_answer_from_pdf(user_msg)

        # PRIORITY 4 — Offline fallback
        else:
            reply = (
                "I don't have information for that.\n"
                "Try asking about **salary, leaves, projects, company FAQs**, "
                "or upload a PDF for document-based answers."
            )

        # Remove thinking bubble
        thinking.empty()

        # Save reply
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": reply,
            "pages": pages
        })

        st.rerun()
# ---------------------------------------------------------
# ANALYTICS PAGE
# ---------------------------------------------------------
def analytics_page(user):
    st.markdown("## 📊 Performance Analytics")

    salary = user["salary"]
    leaves = user["leaves"]
    hours = user["weekly_hours"]

    c1, c2, c3, c4 = st.columns(4)

    # Salary
    with c1:
        st.markdown("<div class='pro-card'>", unsafe_allow_html=True)
        st.markdown("**Net Salary**")
        st.markdown(f"<h2>₹{salary['net']:,}</h2>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Leaves
    with c2:
        st.markdown("<div class='pro-card'>", unsafe_allow_html=True)
        st.markdown("**Leaves Pending**")
        st.markdown(f"<h2>{leaves['pending']}</h2>", unsafe_allow_html=True)
        st.caption(f"Used {leaves['used']} / {leaves['total']}")
        st.markdown("</div>", unsafe_allow_html=True)

    # Weekly Hours
    with c3:
        st.markdown("<div class='pro-card'>", unsafe_allow_html=True)
        st.markdown("**Weekly Hours**")
        st.markdown(f"<h2>{sum(hours.values()):.1f}</h2>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Project Count
    with c4:
        st.markdown("<div class='pro-card'>", unsafe_allow_html=True)
        st.markdown("**Total Projects**")
        total = 1 + len(user["pending_projects"]) + len(user["previous_projects"])
        st.markdown(f"<h2>{total}</h2>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### 📊 Visual Insights")

    colA, colB = st.columns(2)

    # ---------------- Hours Bar Chart ----------------
    with colA:
        st.markdown("<div class='pro-card'>", unsafe_allow_html=True)
        st.markdown("#### Weekly Hours Breakdown")

        df = pd.DataFrame({
            "Day": list(hours.keys()),
            "Hours": list(hours.values())
        })

        chart = alt.Chart(df).mark_bar(color="#2563EB").encode(
            x=alt.X("Day:N", sort=list(hours.keys())),
            y="Hours:Q"
        )
        st.altair_chart(chart, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- Leave Pie Chart ----------------
    with colB:
        st.markdown("<div class='pro-card'>", unsafe_allow_html=True)
        st.markdown("#### Leave Utilization")

        df = pd.DataFrame({
            "Status": ["Used", "Pending"],
            "Days": [leaves["used"], leaves["pending"]]
        })

        pie = alt.Chart(df).mark_arc().encode(
            theta="Days:Q",
            color=alt.Color("Status:N", scale=alt.Scale(range=["#EF4444", "#3B82F6"]))
        )
        st.altair_chart(pie, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)



# ---------------------------------------------------------
# LOGOUT PAGE
# ---------------------------------------------------------
def logout_page():
    st.markdown("## 🚪 Logout")
    st.markdown("<div class='pro-card'>", unsafe_allow_html=True)
    st.write("Are you sure you want to log out?")

    if st.button("Logout", type="primary"):
        st.session_state.logged_in = False
        st.session_state.current_user = None
        st.session_state.chat_history = []
        st.session_state.pdf_chunks = None
        st.session_state.vectorizer = None
        st.session_state.tfidf_matrix = None
        st.session_state.pdf_name = None
        st.success("Logged out successfully.")
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)



# ---------------------------------------------------------
# MAIN APPLICATION
# ---------------------------------------------------------
def main():
    if not st.session_state.logged_in:
        login_page()
        return

    current_user = st.session_state.employees[st.session_state.current_user]

    with st.sidebar:
        st.markdown(f"### 👋 {current_user['name']}")
        st.caption(current_user["role"])
        st.markdown("---")

        page = st.radio("Navigation", ["Dashboard", "QA", "Analytics", "Logout"])
        st.markdown("---")
        st.caption(f"Logged in as `{current_user['username']}`")

    if page == "Dashboard":
        dashboard_page(current_user)
    elif page == "QA":
        qa_page()
    elif page == "Analytics":
        analytics_page(current_user)
    elif page == "Logout":
        logout_page()


if __name__ == "__main__":
    main()
