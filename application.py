# app.py
import streamlit as st
from teacher_agent import ask_teacher, progress, save_progress

st.set_page_config(page_title="ML Teacher Agent", layout="wide")
st.title(" AI Instructor — Agentic AI")

st.sidebar.header("Student Info & Progress")
student_name = st.sidebar.text_input("Your name", value=progress.get("name", "Student"))
if st.sidebar.button("Save name"):
    progress["name"] = student_name
    save_progress(progress)
    st.sidebar.success("Saved.")

#st.sidebar.markdown("### Progress")
#if progress.get("topics"):
    #for t, meta in progress["topics"].items():
       # quiz_status = "✅ Quiz taken" if meta.get('quiz_generated', False) else ""
       # st.sidebar.markdown(f"- **{t}**: introduced={meta.get('introduced', False)} {quiz_status}")
#else:
    st.sidebar.markdown("_No topics yet_")

st.header("Talk to your ML Instructor")
#st.write("Ask questions like: 'Explain gradient descent', 'Give me a code example for logistic regression', 'Quiz me on activation functions'")

# Main chat interface
query = st.text_area("💬 Your question or command", height=120)

if st.button("Ask the Teacher"):
    if not query.strip():
        st.warning("⚠️ Please enter a question.")
    else:
        with st.spinner("🤔 Thinking..."):
            # Clear and ML-focused instruction
            ml_prompt = (
                f"Answer this question **only if it is related to Machine Learning (ML), Deep Learning (DL), or AI.**\n"
                f"If the question is outside ML (e.g., medical, history, food, sports, etc.), "
                f"respond strictly with: \n\n"
                f"❌ **Not Included in Syllabus**\n\n"
                f"This topic is not part of the machine learning curriculum.\n\n"
                f"📚 **I can only help with ML topics like:**\n"
                f"- Machine Learning algorithms\n"
                f"- Deep Learning & Neural Networks\n"
                f"- Data Science fundamentals\n"
                f"- Model training & evaluation\n"
                f"- Feature engineering\n\n"
                f"Question:\n{query.strip()}"
            )

            answer = ask_teacher(ml_prompt)

        st.markdown("### 👨‍🏫 Teacher:")
        st.write(answer)

st.markdown("---")

st.header("Quick Actions")
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📚 Explain Topic")
    topic = st.text_input("Topic to explain", value="", key="explain_topic")
    if st.button("Explain", key="explain_btn"):
        if topic.strip():
            with st.spinner("Generating detailed explanation..."):
                from teacher_agent import explain_concept
                ans = explain_concept(topic.strip())
            st.markdown("### Explanation:")
            st.markdown(ans)
        else:
            st.warning("Please enter a topic.")


with col2:
    st.subheader("💻 Code Example")
    code_topic = st.text_input("Topic for code", value="", key="code_topic")
    code_framework = st.selectbox("Framework", ["numpy", "sklearn", "tensorflow", "pytorch"], index=0)
    if st.button("Generate Code", key="code_btn"):
        if code_topic.strip():
            with st.spinner("Generating code..."):
                # Natural language instruction for the agent
                query = f"Generate a code example for {code_topic.strip()} using {code_framework}."
                ans = ask_teacher(query)
            st.markdown("### Generated Code:")
            # Display as code block
            st.code(ans, language="python")
            
            # Optional: Add a copy button hint
            st.caption("💡 Click the copy icon in the top-right of the code block to copy")
        else:
            st.warning("Please enter a topic.")
# ---------------------- ✨ NEW COLLAPSIBLE SECTION ✨ ----------------------
with st.sidebar.expander("💡 Explain Code (line by line)"):
    code_input = st.text_area("Paste your ML code here", height=150, key="code_explain_input")

    if st.button("🔍 Explain Code", key="explain_code_btn"):
        if code_input.strip():
            with st.spinner("Explaining your code line by line..."):
                from teacher_agent import explain_code
                explanation = explain_code(code_input.strip())
            st.markdown("### 🧠 Explanation:")
            st.write(explanation)
        else:
            st.warning("Please paste some ML-related code to explain.")
# -------------------------------------------------------------------------

with col3:
    st.subheader("📝 Quiz")
    quiz_topic = st.text_input("Quiz topic", value="", key="quiz_topic")
    if st.button("Get Quiz", key="quiz_btn"):
        if quiz_topic.strip():
            with st.spinner("Generating quiz..."):
                prompt = f"""
Generate a detailed 5-question multiple-choice quiz on the topic: **{quiz_topic.strip()}**.

Requirements:
- Each question must have exactly 4 options (A, B, C, D)
- Mark the correct answer clearly at the end of each question
- Focus only on machine learning concepts if relevant
- Questions should be conceptually accurate and moderately challenging

Format each question exactly like this:

Question 1: [question text]
A) [option]
B) [option]
C) [option]
D) [option]
Correct Answer: [Letter]

Now generate the quiz:
"""
                ans = ask_teacher(prompt)

            st.success(f"The quiz on **{quiz_topic.strip()}** has been generated successfully! 🎯")
            st.markdown("### Quiz:")
            st.markdown(ans)

            # Optionally store and allow answering
            #if st.button("Show Answers", key="show_answers"):
                #st.info("Answers are marked with 'Correct Answer:' in the quiz above.")
        #else:
           # st.warning("Please enter a topic.")

st.caption("⚠️ Code execution is sandboxed with basic checks. Avoid running untrusted code.")