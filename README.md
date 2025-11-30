# 💼 HR Assistant AI Portal

A complete HR Assistant Web Application built with **Streamlit**, offering:
- Employee Dashboard  
- PDF Summarization  
- Document-Based Q&A (RAG using TF-IDF)  
- Company Policy & Employee-Specific QA  
- Analytics with Visualizations  
- Secure Username/Password Login  
- 30 Preloaded Employees  

This project is ideal for **HR automation**, **AI portals**, **Gen AI assistants**, and **enterprise chatbots**.

---
URL: https://employeedashoard-4rckrqx7y4mn5aappzqytk2.streamlit.app/

## 🚀 Features

### 🔐 Authentication
- 30 employees with unique username/password  
- Session-based login  
- Role-specific dashboard content  

### 🏠 Employee Dashboard
- Personal information  
- Current project  
- Pending projects  
- Previous work  
- Salary details  
- Leave status  
- Weekly working hours  

### 🧠 AI-Powered Q&A (RAG System)
- Upload PDF and ask questions  
- Extractive summarization  
- Page number citation  
- Chat-style conversation  
- Company policy Q&A  
- Employee-specific Q&A:  
  - Salary  
  - Leaves  
  - Past projects  
  - Current projects  

### 📊 Analytics Dashboard
- Working hours bar chart  
- Leave usage pie chart  
- Salary component breakdown  
- Project distribution chart  
- All visualizations powered by **Altair**  

### 💬 Chatbot Interface
- Scrollable chat window  
- Chat bubbles (user & bot)  
- Input box fixed at the bottom  
- “Processing request…” spinner  

---

## 🔧 Tech Stack

| Component | Technology |
|----------|------------|
| Frontend | Streamlit + Custom CSS (Glassmorphism UI) |
| AI Model | TF-IDF Vectorizer (Lightweight RAG) |
| Parsing | pypdf PDF Reader |
| Charts | Altair |
| Auth | Custom session-based login |
| Styling | HTML + CSS |

---

## 📂 Project Structure

📁 hr-assistant-ai
├── app.py # Main Streamlit application
├── employees.json # (Optional) Employees stored in code
├── requirements.txt # Dependencies list
└── README.md # Project documentation
---

## ⚙️ Installation & Setup (Local)

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/YOUR-USERNAME/hr-assistant-portal.git
cd hr-assistant-portal
<img width="1024" height="559" alt="image" src="https://github.com/user-attachments/assets/ecb71a0d-c42b-4088-8606-b4b16450cace" />
