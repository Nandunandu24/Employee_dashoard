import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import re
import textwrap
from tempfile import NamedTemporaryFile

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pypdf import PdfReader

# ---------------------------------------------------------
# PAGE CONFIG & GLOBAL STYLE
# ---------------------------------------------------------

st.set_page_config(page_title="HR Assistant Portal", layout="wide", page_icon="💼")


def inject_css():
    """Professional Corporate Theme + Custom Components."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

        /* --------------------------------------------------------
           GLOBAL RESET & TYPOGRAPHY
        --------------------------------------------------------- */
        .stApp {
            background-color: #F3F4F6 !important; /* Light Gray Background */
            color: #1F2937 !important; /* Dark Gray Text */
            font-family: 'Inter', sans-serif;
        }

        .block-container {
            padding-top: 2rem;
        }

        h1, h2, h3 {
            color: #111827 !important;
            font-weight: 700 !important;
        }

        /* --------------------------------------------------------
           SIDEBAR (Dark Corporate Blue)
        --------------------------------------------------------- */
        section[data-testid="stSidebar"] {
            background-color: #1E293B !important; /* Dark Slate Blue */
            border-right: 1px solid #334155 !important;
        }

        section[data-testid="stSidebar"] h1, 
        section[data-testid="stSidebar"] h2, 
        section[data-testid="stSidebar"] h3, 
        section[data-testid="stSidebar"] span, 
        section[data-testid="stSidebar"] div,
        section[data-testid="stSidebar"] label {
            color: #F8FAFC !important; /* White Text */
        }
        
        section[data-testid="stSidebar"] hr {
            border-color: #475569 !important;
        }

        /* --------------------------------------------------------
           PROFESSIONAL CARDS (Solid White)
        --------------------------------------------------------- */
        .pro-card {
            background-color: #FFFFFF !important;
            border-radius: 8px !important;
            padding: 1.5rem !important;
            border: 1px solid #E5E7EB !important;
            box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
            margin-bottom: 1rem;
        }

        /* --------------------------------------------------------
           METRICS
        --------------------------------------------------------- */
        .metric-label {
            font-size: 0.85rem;
            font-weight: 600;
            letter-spacing: 0.05em;
            color: #6B7280; /* Gray-500 */
            text-transform: uppercase;
            margin-bottom: 0.25rem;
        }

        .metric-value {
            font-size: 1.75rem;
            font-weight: 700;
            color: #111827; /* Gray-900 */
        }

        .metric-sub {
            font-size: 0.875rem;
            color: #4B5563; /* Gray-600 */
            margin-top: 0.25rem;
        }

        /* --------------------------------------------------------
           INPUT FIELDS (High Contrast)
        --------------------------------------------------------- */
        /* Force inputs to be white with black text */
        input, textarea, select, .stTextInput > div > div > input {
            background-color: #FFFFFF !important;
            color: #111827 !important; /* Almost Black */
            border: 1px solid #D1D5DB !important; /* Visible Border */
            border-radius: 6px !important;
            padding: 0.6rem !important;
        }

        /* Input Label Color */
        .stTextInput > label, .stSelectbox > label, .stFileUploader > label {
            color: #374151 !important;
            font-weight: 600;
        }

        input:focus {
            outline: none !important;
            border: 1px solid #2563EB !important; /* Corporate Blue Focus */
            box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.2);
        }

        /* --------------------------------------------------------
           BUTTONS (Solid Corporate Blue)
        --------------------------------------------------------- */
        button[kind="primary"] {
            background-color: #2563EB !important; /* Royal Blue */
            color: #FFFFFF !important;
            border-radius: 6px !important;
            border: none !important;
            font-weight: 600 !important;
            transition: background-color 0.2s;
        }

        button[kind="primary"]:hover {
            background-color: #1D4ED8 !important; /* Darker Blue */
        }
        
        button[kind="secondary"] {
            background-color: #FFFFFF !important;
            color: #374151 !important;
            border: 1px solid #D1D5DB !important;
        }

        /* --------------------------------------------------------
           CHAT AREA
        --------------------------------------------------------- */
        .chat-container {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }

        /* User Chat Bubble (Blue) */
        .chat-bubble-user {
            background-color: #2563EB !important;
            color: #FFFFFF !important;
            padding: 10px 16px;
            border-radius: 12px 12px 2px 12px;
            align-self: flex-end;
            max-width: 80%;
            font-size: 0.95rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 8px;
            margin-left: auto;
        }

        /* Bot Chat Bubble (White/Gray) */
        .chat-bubble-bot {
            background-color: #F3F4F6 !important;
            color: #1F2937 !important;
            padding: 10px 16px;
            border-radius: 12px 12px 12px 2px;
            align-self: flex-start;
            max-width: 80%;
            border: 1px solid #E5E7EB;
            font-size: 0.95rem;
            margin-bottom: 8px;
        }

        /* --------------------------------------------------------
           PAGE NUMBER PILLS
        --------------------------------------------------------- */
        .page-pill {
            background: #E0F2FE !important;
            color: #0369A1 !important;
            border: 1px solid #BAE6FD !important;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.75rem;
            font-weight: 600;
            margin-right: 5px;
            display: inline-block;
            margin-top: 4px;
        }

        /* --------------------------------------------------------
           CREDENTIALS BOX (Login Page)
        --------------------------------------------------------- */
        .credentials-box {
            background-color: #EFF6FF; /* Light Blue */
            border: 1px solid #BFDBFE;
            color: #1E3A8A;
            padding: 1rem;
            border-radius: 6px;
            font-size: 0.9rem;
            margin-bottom: 1.5rem;
        }
        .credentials-box code {
            background-color: #FFFFFF;
            padding: 2px 5px;
            border-radius: 4px;
            border: 1px solid #BFDBFE;
            color: #D97706; /* Amber for highlight */
            font-weight: bold;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )


inject_css()

# ---------------------------------------------------------
# SESSION STATE INIT
# ---------------------------------------------------------

if "employees" not in st.session_state:
    st.session_state.employees = {}

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "current_user" not in st.session_state:
    st.session_state.current_user = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "pdf_chunks" not in st.session_state:
    st.session_state.pdf_chunks = None

if "tfidf_matrix" not in st.session_state:
    st.session_state.tfidf_matrix = None

if "vectorizer" not in st.session_state:
    st.session_state.vectorizer = None

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

# ---------------------------------------------------------
# EMPLOYEE "DATABASE" (30 employees)
# ---------------------------------------------------------


def create_employee_db():
    names = [
        "Nandini R", "Rakesh Kumar", "Aditi Sharma", "Praveen Rao", "Megha Joshi",
        "Rahul Verma", "Sneha Iyer", "Ankit Patel", "Divya Nair", "Kiran Singh",
        "Ayesha Khan", "Vikram Desai", "Pooja Gupta", "Sanjay Kumar", "Neha Jain",
        "Harish B", "Swati Kulkarni", "Aman Shah", "Kavya R", "Rohit S",
        "Anusha M", "Karthik R", "Shruti P", "Abhishek T", "Shreya G",
        "Manoj K", "Ishita S", "Lokesh V", "Priya D", "Arjun R",
    ]
    roles = [
        "Data Science Intern", "ML Engineer", "Backend Developer", "Frontend Developer",
        "Data Analyst", "DevOps Engineer", "AI Research Intern", "Full Stack Engineer",
    ]
    current_projects = [
        "AI HR Assistant", "Customer 360 Dashboard", "Sales Forecasting Engine",
        "Recommendation System", "Fraud Detection Model", "Chatbot Support System",
        "Employee Attrition Prediction", "Document Summarization Tool",
    ]
    pending_projects = [
        "Onboarding Automation", "Invoice Processing Pipeline", "Lead Scoring Model",
        "NLP Ticket Classifier", "Supply Chain Optimizer", "HR Analytics Portal",
    ]
    previous_projects = [
        "CRM Migration", "Legacy ETL Modernization", "Churn Prediction POC",
        "AB Testing Framework", "Marketing Mix Model", "Data Quality Audit",
    ]

    employees = {}
    for i in range(30):
        username = f"user{i+1:02d}"
        password = f"pass{i+1:02d}"

        name = names[i]
        role = roles[i % len(roles)]
        curr_proj = {
            "name": current_projects[i % len(current_projects)],
            "status": "In Progress",
            "client": "Internal" if i % 3 == 0 else "External",
        }
        pend = [
            pending_projects[(i + j) % len(pending_projects)]
            for j in range(2)
        ]
        prev = [
            previous_projects[(i + j) % len(previous_projects)]
            for j in range(3)
        ]

        base = 50000 + (i * 1200)
        hra = int(base * 0.4)
        allowance = 8000 + (i * 250)
        gross = base + hra + allowance
        tax = int(gross * 0.18)
        net = gross - tax

        leaves_total = 18
        used = (i * 2) % 18
        employees[username] = {
            "username": username,
            "password": password,
            "name": name,
            "role": role,
            "current_project": curr_proj,
            "pending_projects": pend,
            "previous_projects": prev,
            "salary": {
                "basic": base,
                "hra": hra,
                "allowance": allowance,
                "tax": tax,
                "net": net,
            },
            "leaves": {
                "total": leaves_total,
                "used": used,
                "pending": leaves_total - used,
            },
            "weekly_hours": {
                "Mon": 8 + (i % 3),
                "Tue": 7 + (i % 4),
                "Wed": 8.5,
                "Thu": 9,
                "Fri": 7.5 + (i % 2),
                "Sat": 0,
                "Sun": 0,
            },
        }
    return employees


if not st.session_state.employees:
    st.session_state.employees = create_employee_db()

# ---------------------------------------------------------
# SIMPLE COMPANY FAQ (when no PDF)
# ---------------------------------------------------------

COMPANY_FAQ = [
    {
        "keywords": ["working hours", "office hours", "timing"],
        "answer": "Standard working hours are 9:30 AM to 6:30 PM, Monday to Friday."
    },
    {
        "keywords": ["leave", "leaves", "vacation", "holiday"],
        "answer": "Employees receive 18 days of paid leave per year, including casual and sick leaves."
    },
    {
        "keywords": ["probation", "confirmation"],
        "answer": "Probation period is 3 months from the date of joining, after which performance is reviewed for confirmation."
    },
    {
        "keywords": ["remote", "work from home", "wfh"],
        "answer": "Employees can work remotely up to 2 days per week, subject to project and manager approval."
    },
    {
        "keywords": ["company", "what do you do", "about"],
        "answer": (
            "We are **NovaMind AI Solutions**, an AI & Data Science focused technology company.\n"
            "- HQ: Bangalore\n- 1200+ employees\n"
            "- Products: HR AI Assistants, predictive analytics, NLP automation, and enterprise GenAI platforms."
        ),
    },
]


def answer_faq(query: str):
    q = query.lower()
    for item in COMPANY_FAQ:
        if any(kw in q for kw in item["keywords"]):
            return item["answer"]
    return None

# ---------------------------------------------------------
# PDF PROCESSING & TF-IDF RAG
# ---------------------------------------------------------


def read_pdf(path: str):
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append({"page_num": i + 1, "text": text})
    return pages


def chunk_pages(pages, chunk_size=900, overlap=200):
    chunks = []
    for p in pages:
        txt = re.sub(r"\s+", " ", p["text"]).strip()
        if not txt:
            continue
        start = 0
        while start < len(txt):
            chunk = txt[start:start + chunk_size]
            if chunk.strip():
                chunks.append({"page_num": p["page_num"], "text": chunk})
            start += chunk_size - overlap
    return chunks


def process_uploaded_pdf(uploaded_file):
    with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    pages = read_pdf(tmp_path)
    chunks = chunk_pages(pages)
    if not chunks:
        st.error("No readable text found in the PDF.")
        return

    texts = [c["text"] for c in chunks]
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(texts)

    st.session_state.pdf_chunks = chunks
    st.session_state.vectorizer = vectorizer
    st.session_state.tfidf_matrix = tfidf_matrix
    st.session_state.pdf_name = uploaded_file.name


def search_pdf_chunks(query, top_k=4):
    if st.session_state.vectorizer is None or st.session_state.tfidf_matrix is None:
        return [], []

    vec = st.session_state.vectorizer.transform([query])
    sim = cosine_similarity(vec, st.session_state.tfidf_matrix)[0]
    idx = np.argsort(sim)[::-1][:top_k]
    chunks = [st.session_state.pdf_chunks[i] for i in idx]
    pages = sorted({c["page_num"] for c in chunks})
    return chunks, pages


def build_answer_from_pdf(query: str):
    chunks, pages = search_pdf_chunks(query, top_k=4)
    if not chunks:
        return "I couldn't find relevant information in the uploaded PDF.", []

    lines = []
    for c in chunks:
        snippet = textwrap.shorten(c["text"], width=260, placeholder="...")
        lines.append(f"- (Page {c['page_num']}) {snippet}")

    answer = "Here are the most relevant points from the PDF:\n\n" + "\n".join(lines)
    return answer, pages


def summarize_pdf():
    if st.session_state.vectorizer is None:
        return "Please upload a PDF first to generate a summary.", []
    chunks, pages = search_pdf_chunks("overall summary of key points and policies", top_k=5)
    if not chunks:
        return "I couldn't generate a summary for this PDF.", []

    lines = []
    for c in chunks:
        snippet = textwrap.shorten(c["text"], width=260, placeholder="...")
        lines.append(f"- (Page {c['page_num']}) {snippet}")
    summary = "Summary based on important sections:\n\n" + "\n".join(lines)
    return summary, pages

# ---------------------------------------------------------
# EMPLOYEE-SPECIFIC QA
# ---------------------------------------------------------


def answer_employee_specific_query(query, user):
    """Answer questions about this specific logged-in employee."""
    q = query.lower()

    # Leaves
    if "leave" in q or "leaves" in q:
        used = user["leaves"]["used"]
        pending = user["leaves"]["pending"]
        total = user["leaves"]["total"]
        return f"You have **{pending} pending leaves** and have used **{used}** out of **{total}**."

    # Salary
    if "salary" in q or "ctc" in q or "pay" in q:
        sal = user["salary"]
        return (
            f"Your net salary is **₹{sal['net']:,}**.\n"
            f"- Basic: ₹{sal['basic']:,}\n"
            f"- HRA: ₹{sal['hra']:,}\n"
            f"- Allowance: ₹{sal['allowance']:,}\n"
            f"- Tax: ₹{sal['tax']:,}"
        )

    # Role / designation
    if "my role" in q or "designation" in q or "position" in q:
        return f"Your role is **{user['role']}**."

    # Current project
    if ("project" in q and "current" in q) or "working on" in q:
        cp = user["current_project"]
        return (
            f"Your current project is **{cp['name']}**.\n"
            f"Status: `{cp['status']}`\n"
            f"Client: `{cp['client']}`"
        )

    # Previous projects
    if "previous projects" in q or "past projects" in q or "worked on before" in q:
        return "Here are some of your completed projects:\n" + "\n".join(
            f"- {p}" for p in user["previous_projects"]
        )

    # Pending projects
    if "pending projects" in q or "upcoming projects" in q:
        return "These are your pending/assigned projects:\n" + "\n".join(
            f"- {p}" for p in user["pending_projects"]
        )

    # Company high-level questions (fallback)
    if "company" in q or "about" in q or "what do we do" in q:
        return (
            "We are **NovaMind AI Solutions**, a global AI & Data Science company.\n"
            "- 1200+ employees, HQ in Bangalore.\n"
            "- We build AI assistants, predictive analytics solutions, and GenAI platforms "
            "for HR, finance, customer analytics, and operations."
        )

    return None

# ---------------------------------------------------------
# LOGIN PAGE
# ---------------------------------------------------------


def login_page():
    st.markdown("## 🔐 HR Assistant Portal")
    
    # DISTINCT CREDENTIALS BOX
    st.markdown(
        """
        <div class="credentials-box">
            <strong>ℹ️ Demo Credentials:</strong><br>
            Username: <code>user01</code>, <code>user02</code> ... <code>user30</code><br>
            Password: <code>pass01</code>, <code>pass02</code> ... <code>pass30</code>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, _ = st.columns([1.5, 1])
    with col1:
        st.markdown('<div class="pro-card">', unsafe_allow_html=True)
        st.markdown("### Sign In")
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            
            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button("Secure Login", type="primary")

            if submitted:
                employees = st.session_state.employees
                if username in employees and employees[username]["password"] == password:
                    # set login state
                    st.session_state.logged_in = True
                    st.session_state.current_user = username

                    # reset per-user session data
                    st.session_state.chat_history = []
                    st.session_state.pdf_chunks = None
                    st.session_state.vectorizer = None
                    st.session_state.tfidf_matrix = None
                    st.session_state.pdf_name = None

                    st.success(f"Welcome, {employees[username]['name']}!")
                    st.rerun()
                else:
                    st.error("Invalid username or password.")
        st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# DASHBOARD PAGE
# ---------------------------------------------------------


def dashboard_page(user):
    st.markdown("## 🏠 Executive Dashboard")

    col1, col2, col3 = st.columns(3)
    salary = user["salary"]
    leaves = user["leaves"]

    with col1:
        st.markdown('<div class="pro-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Employee Profile</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{user["name"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-sub" style="color:#2563EB;">{user["role"]}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="pro-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Net Compensation</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">₹ {salary["net"]:,}</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="metric-sub">Monthly Gross: ₹{salary["basic"] + salary["hra"] + salary["allowance"]:,}</div>',
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="pro-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Leave Balance</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="metric-value">{leaves["pending"]} Days</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="metric-sub">Utilization: {leaves["used"]}/{leaves["total"]} days</div>',
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns([1.5, 1])
    with col_a:
        st.markdown('<div class="pro-card">', unsafe_allow_html=True)
        st.markdown("### 📌 Projects Overview")
        cp = user["current_project"]
        st.info(f"**Current Active Project:** {cp['name']} \n\n *Status:* {cp['status']} | *Client:* {cp['client']}")

        st.markdown("**Pending Projects:**")
        for p in user["pending_projects"]:
            st.markdown(f"- ⏳ {p}")

        st.markdown("**Completed Projects:**")
        for p in user["previous_projects"]:
            st.markdown(f"- ✅ {p}")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="pro-card">', unsafe_allow_html=True)
        st.markdown("### ⏱ Activity Snapshot")
        hours = user["weekly_hours"]
        total_hours = sum(hours.values())
        st.metric("Total Hours (This Week)", f"{total_hours:.1f} hrs")
        st.markdown(" ")
        df_hours = pd.DataFrame(
            {"Day": list(hours.keys()), "Hours": list(hours.values())}
        )
        # UPDATED CHART COLOR: Corporate Blue
        chart = alt.Chart(df_hours).mark_bar(color="#2563EB").encode(
            x=alt.X("Day:N", sort=list(hours.keys())),
            y="Hours:Q",
            tooltip=["Day", "Hours"]
        )
        st.altair_chart(chart, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# QA PAGE (PDF summarising + QA + chat)
# ---------------------------------------------------------


def qa_page():
    user = st.session_state.employees[st.session_state.current_user]
    st.markdown("## 🤖 AI HR Chat Assistant")

    # --- SHOW PDF STATUS ---
    if st.session_state.pdf_name:
        st.success(f"Using PDF: **{st.session_state.pdf_name}**")
    else:
        st.info("No PDF uploaded — answering using employee data + company policies.")

    # --- PDF UPLOADER CARD ---
    with st.expander("📄 Upload PDF for Document-Based Answers", expanded=True):
        uploaded = st.file_uploader("Upload Policy PDF", type=["pdf"])
        if uploaded:
            process_uploaded_pdf(uploaded)
            st.success(f"PDF **{uploaded.name}** processed successfully!")

    st.markdown("### 💬 Chat Window")

    # --- CHAT WINDOW (SCROLLABLE AREA) ---
    chat_container = st.container()

    with chat_container:
        st.markdown(
            """
            <div style="
                height: 450px;
                overflow-y: auto;
                padding: 10px;
                background-color: #FFFFFF;
                border-radius: 10px;
                border: 1px solid #E5E7EB;
            ">
            """,
            unsafe_allow_html=True,
        )

        if not st.session_state.chat_history:
            st.markdown(
                "<p style='text-align:center; color:#9CA3AF;'>Start chatting with your HR Assistant...</p>",
                unsafe_allow_html=True,
            )

        # SHOW CHAT MESSAGES
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(
                    f"<div class='chat-bubble-user'>{msg['content']}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div class='chat-bubble-bot'>{msg['content']}</div>",
                    unsafe_allow_html=True,
                )

                if msg.get("pages"):
                    pills = "".join(
                        f'<span class="page-pill">Page {p}</span>' for p in msg["pages"]
                    )
                    st.markdown(pills, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # --- CHAT INPUT (BOTTOM, OUTSIDE ALL CONTAINERS) ---
    user_msg = st.chat_input("Ask about your salary, leaves, company, or the uploaded PDF...")

    # --- PROCESS MESSAGE ---
    if user_msg:
        st.session_state.chat_history.append(
            {"role": "user", "content": user_msg, "pages": []}
        )

        with st.spinner("🤖 Processing your request..."):
            # 1️⃣ EMPLOYEE-SPECIFIC Q&A
            emp_reply = answer_employee_specific_query(user_msg, user)
            if emp_reply:
                reply, pages = emp_reply, []

            # 2️⃣ COMPANY FAQ
            elif answer_faq(user_msg):
                reply, pages = answer_faq(user_msg), []

            # 3️⃣ PDF-BASED ANSWERS
            elif st.session_state.pdf_chunks:
                if "summary" in user_msg.lower():
                    reply, pages = summarize_pdf()
                else:
                    reply, pages = build_answer_from_pdf(user_msg)

            # 4️⃣ DEFAULT
            else:
                reply, pages = (
                    "Please upload a PDF or ask about your role, salary, leaves, or company policies.",
                    [],
                )

        st.session_state.chat_history.append(
            {"role": "assistant", "content": reply, "pages": pages}
        )

        st.rerun()

# ---------------------------------------------------------
# ANALYTICS PAGE
# ---------------------------------------------------------


def analytics_page(user):
    st.markdown("## 📊 Performance Analytics")

    salary = user["salary"]
    leaves = user["leaves"]
    hours = user["weekly_hours"]

    # ---- Top metrics row ----
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="pro-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Net Salary</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">₹ {salary["net"]:,}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-sub">Post-Tax</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="pro-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Leaves Pending</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{leaves["pending"]}</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="metric-sub">Used {leaves["used"]} of {leaves["total"]}</div>',
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with c3:
        st.markdown('<div class="pro-card">', unsafe_allow_html=True)
        total_hours = sum(hours.values())
        st.markdown('<div class="metric-label">Weekly Hours</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{total_hours:.1f} hrs</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-sub">Mon–Sun</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c4:
        st.markdown('<div class="pro-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Total Projects</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="metric-value">{1 + len(user["pending_projects"]) + len(user["previous_projects"])}</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="metric-sub">Active + Queue + History</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ---- Row 1: Working hours + Leaves pie ----
    r1c1, r1c2 = st.columns(2)

    with r1c1:
        st.markdown('<div class="pro-card">', unsafe_allow_html=True)
        st.markdown("### ⏱ Weekly Working Hours")
        df_hours = pd.DataFrame({"Day": list(hours.keys()), "Hours": list(hours.values())})
        
        # UPDATED COLORS: Professional Blue
        chart_hours = (
            alt.Chart(df_hours)
            .mark_bar(color="#3B82F6") # Corporate Blue
            .encode(
                x=alt.X("Day:N", sort=list(hours.keys())),
                y="Hours:Q",
                tooltip=["Day", "Hours"],
            )
            .properties(height=280)
        )
        st.altair_chart(chart_hours, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with r1c2:
        st.markdown('<div class="pro-card">', unsafe_allow_html=True)
        st.markdown("### 🧾 Leave Utilization")
        df_leaves = pd.DataFrame(
            {
                "Status": ["Used", "Pending"],
                "Days": [leaves["used"], leaves["pending"]],
            }
        )
        # UPDATED COLORS: Red (Used) vs Blue (Pending)
        pie = (
            alt.Chart(df_leaves)
            .mark_arc(innerRadius=50)
            .encode(
                theta="Days:Q",
                color=alt.Color(
                    "Status:N",
                    scale=alt.Scale(
                        domain=["Used", "Pending"],
                        range=["#EF4444", "#3B82F6"], # Red / Blue
                    ),
                ),
                tooltip=["Status", "Days"],
            )
            .properties(height=280)
        )
        st.altair_chart(pie, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ---- Row 2: Salary components + Projects mix ----
    r2c1, r2c2 = st.columns(2)

    with r2c1:
        st.markdown('<div class="pro-card">', unsafe_allow_html=True)
        st.markdown("### 💰 Salary Components")
        df_salary = pd.DataFrame(
            {
                "Component": ["Basic", "HRA", "Allowance", "Tax", "Net"],
                "Amount": [
                    salary["basic"],
                    salary["hra"],
                    salary["allowance"],
                    salary["tax"],
                    salary["net"],
                ],
            }
        )
        # UPDATED COLORS: Professional Data Palette
        chart_sal = (
            alt.Chart(df_salary)
            .mark_bar()
            .encode(
                x="Component:N",
                y="Amount:Q",
                color=alt.Color(
                    "Component:N",
                    scale=alt.Scale(
                        range=["#6366F1", "#8B5CF6", "#F59E0B", "#EF4444", "#10B981"]
                        # Indigo, Purple, Amber, Red, Emerald
                    ),
                ),
                tooltip=["Component", "Amount"],
            )
            .properties(height=280)
        )
        st.altair_chart(chart_sal, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with r2c2:
        st.markdown('<div class="pro-card">', unsafe_allow_html=True)
        st.markdown("### 📁 Project Portfolio Mix")
        df_proj = pd.DataFrame(
            {
                "Type": ["Current", "Pending", "Previous"],
                "Count": [
                    1,
                    len(user["pending_projects"]),
                    len(user["previous_projects"]),
                ],
            }
        )
        # UPDATED COLORS: Professional Status Colors
        proj_chart = (
            alt.Chart(df_proj)
            .mark_bar()
            .encode(
                x="Type:N",
                y="Count:Q",
                color=alt.Color(
                    "Type:N",
                    scale=alt.Scale(
                        range=["#10B981", "#F59E0B", "#64748B"] 
                        # Green (Current), Amber (Pending), Slate (Previous)
                    ),
                ),
                tooltip=["Type", "Count"],
            )
            .properties(height=280)
        )
        st.altair_chart(proj_chart, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# LOGOUT
# ---------------------------------------------------------


def logout_page():
    st.markdown("## 🚪 Secure Logout")
    st.markdown('<div class="pro-card">', unsafe_allow_html=True)
    st.write("Are you sure you want to end your session?")
    if st.button("Logout", type="primary"):
        st.session_state.logged_in = False
        st.session_state.current_user = None
        st.session_state.chat_history = []
        st.session_state.pdf_chunks = None
        st.session_state.vectorizer = None
        st.session_state.tfidf_matrix = None
        st.session_state.pdf_name = None
        st.success("You have been logged out.")
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# MAIN APP
# ---------------------------------------------------------


def main():
    if not st.session_state.logged_in:
        login_page()
        return

    user = st.session_state.employees[st.session_state.current_user]

    with st.sidebar:
        st.markdown(f"### 👋 {user['name']}")
        st.caption(user["role"])
        st.markdown("---")
        page = st.radio("Navigate", ["Dashboard", "QA", "Analytics", "Logout"])
        st.markdown("---")
        
    if page == "Dashboard":
        dashboard_page(user)
    elif page == "QA":
        qa_page()
    elif page == "Analytics":
        analytics_page(user)
    elif page == "Logout":
        logout_page()


if __name__ == "__main__":
    main()

