#  ML Instructor — Agentic AI  
**An interactive Streamlit-based Machine Learning teaching assistant powered by LangChain + Groq LLM**

---

##  Project Overview

The **ML Instructor Agent** is an **AI-powered tutor** designed to help students learn **Machine Learning (ML)** and **Deep Learning (DL)** interactively.  
It explains ML concepts, generates runnable code examples, quizzes students, and tracks progress persistently.  

Built using **Streamlit**, **LangChain**, and **Groq’s LLaMA-3.1 model**, this system ensures responses stay within ML topics and rejects non-relevant questions (like cooking, history, etc.).

---

##  System Architecture
┌────────────────────────┐
│ Streamlit UI │
│ (application.py) │
│ │
│ ┌────────────────────┐ │
│ │ teacher_agent.py │ │
│ │ ├ explain_concept │ │
│ │ ├ explain_code │ │
│ │ ├ generate_quiz │ │
│ │ └ ask_teacher │ │
│ └────────────────────┘ │
│ │
│ Persistent Storage │
│ (student_progress.json)│
└────────────┬───────────┘
│
▼
LangChain + Groq API
(LLaMA-3.1-8b-instant)

---

## ⚙️ Features

-  **Concept Explanation** — Explains ML/DL concepts in structured format (definition, analogy, and use).  
-  **Code Generation** — Generates Python code examples using frameworks like NumPy, scikit-learn, TensorFlow, and PyTorch.  
-  **Quiz Generation** — Creates 5-question multiple-choice quizzes for any ML topic.  
-  **Progress Tracking** — Stores student interactions and topics in `student_progress.json`.  
-  **Strict ML Topic Validation** — Filters out non-ML topics automatically.  

---

##  Key Files

| File | Description |
|------|--------------|
| `application.py` | Streamlit frontend that handles user interface and interactions |
| `teacher_agent.py` | Backend logic — LangChain agent, prompt templates, topic validation, and LLM calls |
| `student_progress.json` | Persistent data store for student progress (topics introduced, quizzes generated) |
| `.env` | Environment file containing your Groq API key |
| `requirements.txt` | Dependencies required to run the app |

---

##  Installation and Setup

### 1️ Clone the repository
```bash
git clone https://github.com/yourusername/ml-instructor-agent.git
cd ml-instructor-agent
### 2  Create and activate virtual environment
python -m venv venv
source venv/bin/activate    # on Linux/Mac
venv\Scripts\activate       # on Windows

### 3 Install dependencies
pip install -r requirements.txt

### 4 Set up environment variables

Create a .env file in the project root with the following:

GROQ_API_KEY=your_groq_api_key_here

 ### 5 Run the Streamlit app
streamlit run application.py

### Requirements File
streamlit
langchain
langchain-groq
python-dotenv


