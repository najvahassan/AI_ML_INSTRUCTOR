from langchain_groq import ChatGroq
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os, json
from typing import Dict, Any

# ------------------------------------------------------------
# ✅ Load environment variables
# ------------------------------------------------------------
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY not found. Please set it in your .env file.")

# ------------------------------------------------------------
# ✅ Initialize the LLM
# ------------------------------------------------------------
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    streaming=False
)

# ------------------------------------------------------------
# ✅ Persistent progress file
# ------------------------------------------------------------
PROGRESS_FILE = "student_progress.json"

def load_progress() -> Dict[str, Any]:
    """Load student progress from disk."""
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_progress(progress: Dict[str, Any]) -> None:
    """Save student progress to disk."""
    try:
        with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
            json.dump(progress, f, indent=2)
    except Exception as e:
        print(f"⚠️ Could not save progress: {e}")

progress = load_progress()

# ------------------------------------------------------------
# 🧠 Tool Implementations
# ------------------------------------------------------------
def explain_concept(topic: str) -> str:
    """Return a structured explanation of ML topic only if it's related to machine learning."""
    topic = topic.strip()
    
    # First, check if the topic is related to machine learning
    validation_prompt = f"""You are a strict machine learning curriculum validator.

Analyze this EXACT topic: "{topic}"

Is this ENTIRE topic specifically about machine learning, data science, artificial intelligence, deep learning, or directly related ML/AI concepts?

IMPORTANT RULES:
- If the topic mentions cooking, recipes, food, history, literature, sports, music, or other non-ML subjects → Answer NO
- Only answer YES if the MAIN topic is clearly ML/AI/Data Science related
- Don't be fooled by ML keywords mixed with unrelated topics (like "recipe of chicken engineering")
- "Feature engineering" = YES, but "recipe of chicken engineering" = NO

Answer with ONLY one word: YES or NO

Answer:"""
    
    try:
        validation_resp = llm.invoke(validation_prompt)
        validation_text = validation_resp.content if hasattr(validation_resp, "content") else str(validation_resp)
        validation_text = validation_text.strip().upper()
        
        # More strict checking - look for NO or if YES is not clearly present
        is_ml_topic = "YES" in validation_text and validation_text.startswith("YES")
        
        if not is_ml_topic:
            return f"""❌ **Not Included in Syllabus**

The topic **'{topic}'** is not part of the machine learning curriculum.

📚 **I can only help with ML topics like:**
- Machine Learning algorithms (regression, classification, clustering)
- Deep Learning & Neural Networks
- Feature engineering & data preprocessing
- Model training, evaluation & optimization
- AI concepts (NLP, Computer Vision, etc.)
- Data Science fundamentals

💡 **Please ask about a machine learning-related topic!**"""
        
        # If YES, proceed with explanation
        explanation_prompt = f"""You are an expert ML teacher.
Explain the machine learning concept '{topic}' in a clear, structured way:

1️⃣ **Definition** (2-3 sentences)
2️⃣ **Simple Analogy** (easy-to-understand comparison)
3️⃣ **Use in ML** (where/why it's important)

Keep it concise and educational."""
        
        resp = llm.invoke(explanation_prompt)
        text = resp.content if hasattr(resp, "content") else str(resp)
        
        # Save progress only for valid ML topics
        progress.setdefault("topics", {})
        progress["topics"].setdefault(topic.lower(), {})["introduced"] = True
        save_progress(progress)
        
        return text
        
    except Exception as e:
        return f"⚠️ Error generating explanation: {e}"


def explain_code(code: str) -> str:
    """Explain ML-related code line by line."""
    code = code.strip()
    
    if not code:
        return "❌ No code provided. Please paste some code to explain."
    
    # Check if code seems to be ML-related
    ml_libraries = ['sklearn', 'tensorflow', 'torch', 'keras', 'numpy', 'pandas', 'scipy']
    ml_keywords = ['model', 'train', 'predict', 'fit', 'neural', 'layer', 'feature', 'dataset']
    
    code_lower = code.lower()
    has_ml_content = any(lib in code_lower for lib in ml_libraries) or any(kw in code_lower for kw in ml_keywords)
    
    if not has_ml_content:
        return """❌ **Not ML-Related Code**

This code doesn't appear to be related to machine learning.

📚 **I can explain code that uses:**
- scikit-learn (sklearn)
- TensorFlow / Keras
- PyTorch
- NumPy/Pandas for ML
- ML algorithms and models

💡 **Please provide ML-related code!**"""
    
    prompt = f"""You are an expert ML code instructor. Explain this machine learning code in detail:

```python
{code}
```

Provide:
1️⃣ **Overview**: What does this code do? (2-3 sentences)
2️⃣ **Line-by-Line Explanation**: Explain key lines and their purpose
3️⃣ **Key Concepts**: What ML concepts are demonstrated?
4️⃣ **Output/Result**: What would this code produce?

Be clear, educational, and focus on the ML aspects."""
    
    try:
        resp = llm.invoke(prompt)
        explanation = resp.content if hasattr(resp, "content") else str(resp)
        return explanation
    except Exception as e:
        return f"⚠️ Error explaining code: {e}"


def generate_code_example(input_str: str) -> str:
    """Generate runnable minimal example for ML topics only."""
    topic = input_str.strip()
    framework = "numpy"
    
    # First validate if it's ML-related with stricter prompt
    validation_prompt = f"""You are a strict machine learning curriculum validator.

Is this EXACT topic: "{topic}" specifically about machine learning, data science, AI, or directly related ML programming?

RULES:
- If it mentions non-ML subjects (cooking, games, general programming, etc.) → Answer NO
- Only answer YES if it's clearly ML/AI/Data Science related
- Don't be fooled by ML keywords mixed with unrelated topics

Answer with only: YES or NO"""
    
    try:
        validation_resp = llm.invoke(validation_prompt)
        validation_text = validation_resp.content if hasattr(validation_resp, "content") else str(validation_resp)
        validation_text = validation_text.strip().upper()
        
        # Strict checking
        is_ml_topic = "YES" in validation_text and validation_text.startswith("YES")
        
        if not is_ml_topic:
            return f"""# ❌ Not Included in Syllabus
# 
# The topic '{topic}' is not part of the machine learning curriculum.
#
# 📚 I can generate code for ML topics like:
# - Linear/Logistic Regression
# - Decision Trees, Random Forests
# - Neural Networks
# - K-means Clustering
# - Data preprocessing & feature engineering
# - Model evaluation techniques
#
# 💡 Please request code for an ML-related topic!"""
    except:
        pass  # Continue if validation fails
    
    # Parse input - handle various formats
    input_lower = input_str.lower()
    
    # Check for framework mentions
    if "sklearn" in input_lower or "scikit" in input_lower:
        framework = "sklearn"
    elif "tensorflow" in input_lower or "tf" in input_lower:
        framework = "tensorflow"
    elif "pytorch" in input_lower or "torch" in input_lower:
        framework = "pytorch"
    elif "pandas" in input_lower:
        framework = "pandas"
    
    # If format is "topic with framework" or "topic using framework"
    for fw in ["sklearn", "numpy", "tensorflow", "pytorch", "pandas"]:
        if fw in input_lower:
            framework = fw
            topic = input_str.replace(fw, "").replace("using", "").replace("with", "").strip()
    
    prompt = f"""Write a complete, runnable Python code example for: {topic}

Requirements:
- Use {framework} library
- Include all necessary imports
- Add brief comments explaining key steps
- Keep it under 30 lines
- Make it a working, executable example
- Do NOT include any explanatory text before or after the code
- Output ONLY the Python code

Start writing the code now:"""
    
    try:
        resp = llm.invoke(prompt)
        code = resp.content if hasattr(resp, "content") else str(resp)
        
        # Clean up the response - remove markdown code blocks if present
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0].strip()
        elif "```" in code:
            code = code.split("```")[1].split("```")[0].strip()
            
    except Exception as e:
        code = f"# Error generating code: {e}\n# Please try again with a more specific topic."
    
    return code


def generate_quiz(topic: str) -> str:
    """Generate 5 well-formatted MCQs for valid ML topics only."""
    topic = topic.strip()

    # --- STEP 1: Validate ML relevance ---
    validation_prompt = f"""You are a strict ML curriculum validator.
Is the topic "{topic}" specifically related to machine learning, data science, AI, or deep learning?

RULES:
- Say "YES" only if it is clearly about ML, AI, data science, or statistics concepts.
- Say "NO" if it is unrelated (e.g., cooking, sports, general computer science topics not tied to ML).
- Reply with only: YES or NO
"""
    try:
        validation_resp = llm.invoke(validation_prompt)
        validation_text = (
            validation_resp.content.strip().upper()
            if hasattr(validation_resp, "content")
            else str(validation_resp).strip().upper()
        )
        is_ml_topic = validation_text.startswith("YES")
        if not is_ml_topic:
            return f"""❌ **Not Included in Syllabus**

The topic **'{topic}'** is not part of the machine learning curriculum.

📘 **I can generate quizzes for topics like:**
- Supervised Learning (Regression, Classification)
- Unsupervised Learning (Clustering, Dimensionality Reduction)
- Deep Learning & Neural Networks
- Feature Engineering & Data Preprocessing
- Model Evaluation & Optimization
- Specific algorithms (Decision Trees, SVM, K-Means, etc.)

💡 Please enter an ML-related topic to continue."""
    except Exception as e:
        return f"⚠️ Validation failed due to: {e}"

    # --- STEP 2: Generate Quiz ---
    prompt = f"""
You are an expert ML instructor.

Generate a quiz on the topic **'{topic}'** in machine learning.

**Requirements:**
1. Create exactly **5 multiple-choice questions (MCQs)**.
2. Each question must have exactly **4 options (A, B, C, D)**.
3. Indicate the **correct answer clearly** after each question.
4. Add a **1-line explanation** for each correct answer.
5. Cover various aspects of the topic: definition, working principle, advantages, limitations, and real-world application.
6. Make the questions **moderately challenging but clear**.
7. Format output cleanly in Markdown like this:

---

**Question 1:** What does SVM stand for?
A) Simple Vector Machine  
B) Support Vector Machine  
C) Supervised Variance Model  
D) Statistical Variance Matrix  
**Correct Answer:** B  
**Explanation:** SVM stands for Support Vector Machine, a supervised learning algorithm for classification and regression tasks.

---

Now generate the quiz for **{topic}**:
"""
    try:
        resp = llm.invoke(prompt)
        quiz_text = resp.content if hasattr(resp, "content") else str(resp)
    except Exception as e:
        quiz_text = f"⚠️ Error generating quiz: {e}"

    # --- STEP 3: Update progress ---
    progress.setdefault("topics", {})
    progress["topics"].setdefault(topic.lower(), {})["quiz_generated"] = True
    progress["last_quiz"] = {"topic": topic, "quiz": quiz_text}
    save_progress(progress)

    return quiz_text



# ------------------------------------------------------------
# 🔧 Wrap tools for the agent
# ------------------------------------------------------------
tools = [
    Tool(
        name="ExplainConcept", 
        func=explain_concept, 
        description="Use this to explain MACHINE LEARNING concepts only. Input should be just the ML topic name as a string, e.g., 'gradient descent', 'neural networks', 'random forest'. Will reject non-ML topics."
    ),
    Tool(
        name="GenerateCodeExample", 
        func=generate_code_example, 
        description="Use this to generate Python code examples for MACHINE LEARNING topics only. Input should be the ML topic name and optionally mention the framework. Examples: 'linear regression', 'decision tree using sklearn', 'neural network with tensorflow'. Will reject non-ML topics."
    ),
    Tool(
        name="ExplainCode",
        func=explain_code,
        description="Use this to explain ML-related Python code. Input should be the complete code as a string. Will provide line-by-line explanation of what the code does, focusing on ML concepts."
    ),
    Tool(
        name="GenerateQuiz", 
        func=generate_quiz, 
        description="Use this to generate a quiz with 5 MCQ questions on MACHINE LEARNING topics only. Input should be just the ML topic name as a string, e.g., 'activation functions', 'decision trees', 'K-means clustering'. Will reject non-ML topics."
    ),
]

# ------------------------------------------------------------
# 🧠 Memory + Agent Initialization
# ------------------------------------------------------------
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# System message to guide the agent
SYSTEM_MESSAGE = """You are an expert Machine Learning teacher. You can help with:
- Machine Learning (supervised, unsupervised, reinforcement learning)
- Deep Learning and Neural Networks
- Data Science and Statistics for ML
- AI concepts related to ML
- Model training, evaluation, and optimization
- Feature engineering and data preprocessing
- Explaining ML-related Python code

If a user asks about topics OUTSIDE machine learning (like cooking, history, general programming, etc.), 
immediately respond: "❌ This topic is not included in the machine learning syllabus. Please ask about ML-related topics."

When explaining code, focus on ML concepts and provide line-by-line analysis.
Be direct and don't waste iterations trying to handle non-ML topics. Always be helpful, clear, and educational for ML topics."""

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent_type=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    memory=memory,
    max_iterations=5,          # 🔧 Reduced from 15 to 5
    max_execution_time=60,     # 🔧 Reduced from 180 to 60 seconds
    handle_parsing_errors=True,
    early_stopping_method="generate",  # 🔧 Added early stopping
    agent_kwargs={
        "system_message": SYSTEM_MESSAGE
    }
)

# ------------------------------------------------------------
# 🎓 Main Entry Function with Better Error Handling
# ------------------------------------------------------------
def ask_teacher(query: str) -> str:
    """Ask the ML teacher agent a question with improved error handling."""
    try:
        # Quick pre-check for obviously non-ML queries
        non_ml_keywords = [
            'cooking', 'recipe', 'food', 'curry', 'chicken', 'pizza', 
            'history', 'geography', 'sports', 'football', 'music', 
            'art', 'painting', 'literature', 'novel', 'movie','health',
        ]
        
        query_lower = query.lower()
        
        # Reject non-ML queries
        if any(keyword in query_lower for keyword in non_ml_keywords):
            ml_keywords = ['learning', 'model', 'algorithm', 'neural', 'data', 'training']
            if not any(ml_kw in query_lower for ml_kw in ml_keywords):
                return """❌ **Not Included in Syllabus**

This topic is not part of the machine learning curriculum.

📚 **I can only help with ML topics like:**
- Machine Learning algorithms
- Deep Learning & Neural Networks
- Data Science fundamentals
- Model training & evaluation
- Feature engineering

💡 **Please ask about a machine learning-related topic!**"""

        # 🧠 Detect quiz generation commands
        if "generate a quiz" in query_lower or "get quiz" in query_lower or "quiz on" in query_lower:
              # Replace with your actual LLM import
            
            # Clean topic extraction
            topic = query.replace("Generate a quiz on", "").replace("quiz on", "").strip()
            prompt = f"""
You are an expert AI teacher.

Generate a **5-question multiple-choice quiz** on the machine learning topic: **{topic}**.

🧠 Requirements:
- Each question must have exactly 4 options (A, B, C, D)
- Clearly indicate the correct answer after each question
- Questions should be conceptual and moderately challenging
- Focus strictly on ML concepts related to {topic}

📘 Format:
Question 1: [Question text]
A) [Option]
B) [Option]
C) [Option]
D) [Option]
Correct Answer: [Letter]

Now generate the quiz:
"""
            response = llm.invoke(prompt)
            return response.content if hasattr(response, "content") else str(response)
        
        # Default: use agent for general queries
        result = agent.run(query)
        return result.strip()
    
    except Exception as e:
        error_msg = str(e)
        if "iteration limit" in error_msg.lower() or "time limit" in error_msg.lower():
            return """⚠️ **Request Timeout**

The request took too long to process. Try:
- A more specific ML question
- Using Quick Actions buttons
- Staying within ML topics"""
        elif "parsing" in error_msg.lower():
            return """⚠️ **Processing Error**

I had trouble understanding your request.
Please rephrase your question or use Quick Actions."""
        else:
            return f"⚠️ **Error:** {error_msg}"
