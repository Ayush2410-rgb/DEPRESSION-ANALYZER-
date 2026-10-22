import streamlit as st
import pandas as pd
import numpy  as np
import joblib
import json
from pathlib import Path

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Mental Health NLP Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.hero {
    padding: 2rem;
    border-radius: 18px;
    background: linear-gradient(
        135deg,
        #1f2937,
        #374151
    );
    color: white;
    margin-bottom: 1.5rem;
}

.hero h1 {
    font-size: 2.5rem;
    margin-bottom: 0.5rem;
}

.hero p {
    font-size: 1.05rem;
    opacity: 0.9;
}

.metric-card {
    padding: 1rem;
    border-radius: 12px;
    border: 1px solid #d1d5db;
    text-align: center;
}

.warning-box {
    padding: 1rem;
    border-radius: 10px;
    background-color: #fff7ed;
    border: 1px solid #fed7aa;
    margin: 1rem 0;
}

.result-box {
    padding: 1.2rem;
    border-radius: 12px;
    border: 1px solid #d1d5db;
    margin-top: 1rem;
}

.footer {
    text-align: center;
    padding: 2rem 0 1rem 0;
    color: #6b7280;
    font-size: 0.9rem;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODEL
# =========================================================

MODEL_PATH = "model.pkl"
METRICS_PATH = "metrics.json"

if not Path(MODEL_PATH).exists():
    st.error("Model not found. Please run `python train.py` first.")
    st.stop()

if not Path(METRICS_PATH).exists():
    st.error("metrics.json not found. Please run `python train.py` first.")
    st.stop()

model = joblib.load(MODEL_PATH)

with open(METRICS_PATH, "r") as file:
    metrics = json.load(file)

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("🧠 NLP Classifier")

    st.markdown("---")

    st.subheader("Project Pipeline")

    st.markdown("""
    **Text Input**
    
    ↓
    
    **TF-IDF**
    
    ↓
    
    **Logistic Regression**
    
    ↓
    
    **Research Category**
    """)

    st.markdown("---")

    st.subheader("Technologies")

    st.markdown("""
    - Python
    - Pandas
    - Scikit-learn
    - TF-IDF
    - Logistic Regression
    - Streamlit
    """)

    st.markdown("---")

    st.caption(
        "Educational machine learning project"
    )

# =========================================================
# HERO SECTION
# =========================================================

st.markdown("""
<div class="hero">

<h1>🧠 Mental Health Text Classification</h1>

<p>
An NLP-based machine learning application that classifies
text into categories represented in a research dataset.
</p>

</div>
""", unsafe_allow_html=True)

# =========================================================
# DISCLAIMER
# =========================================================

st.markdown("""
<div class="warning-box">

<b>⚠️ Responsible Use Notice</b>

<br><br>

This application is an educational and research demonstration.
It is <b>not a medical diagnostic system</b> and should not be used
to diagnose depression, anxiety, suicidal thoughts, or any other
mental health condition.

</div>
""", unsafe_allow_html=True)

# =========================================================
# PERFORMANCE
# =========================================================

st.subheader("📊 Model Performance")

report = metrics["classification_report"]

accuracy = metrics["accuracy"]
precision = report["macro avg"]["precision"]
recall = report["macro avg"]["recall"]
f1 = report["macro avg"]["f1-score"]

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Accuracy",
    f"{accuracy * 100:.2f}%"
)

c2.metric(
    "Macro Precision",
    f"{precision * 100:.2f}%"
)

c3.metric(
    "Macro Recall",
    f"{recall * 100:.2f}%"
)

c4.metric(
    "Macro F1",
    f"{f1 * 100:.2f}%"
)

# =========================================================
# CLASSIFICATION
# =========================================================

st.markdown("---")

st.subheader("🔍 Text Classification")

st.write(
    "Enter a sample of text below and let the trained model "
    "classify it into one of the categories represented in "
    "the dataset."
)

user_text = st.text_area(
    "Text input",
    placeholder=(
        "Example: I have been feeling stressed about "
        "my studies lately..."
    ),
    height=160
)

classify = st.button(
    "🔎 Classify Text",
    type="primary",
    use_container_width=True
)

if classify:

    if not user_text.strip():

        st.warning("Please enter some text first.")

    else:

        prediction = model.predict([user_text])[0]

        probabilities = model.predict_proba([user_text])[0]

        st.markdown(
            f"""
            <div class="result-box">
            <h3>Model Output</h3>

            <p>
            Predicted research category:
            <b>{prediction}</b>
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )

        probability_df = pd.DataFrame({
            "Category": model.classes_,
            "Model Score (%)": probabilities * 100
        })

        probability_df["Model Score (%)"] = (
            probability_df["Model Score (%)"].round(2)
        )

        st.write("### Model Scores")

        st.bar_chart(
            probability_df.set_index("Category")
        )

        st.dataframe(
            probability_df,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "Model scores represent the classifier's output for "
            "the provided text. They do not represent clinical "
            "probabilities or medical assessments."
        )

# =========================================================
# VISUAL ANALYSIS
# =========================================================

st.markdown("---")

st.subheader("📈 Dataset & Model Analysis")

graph1, graph2 = st.columns(2)

with graph1:

    st.write("#### Class Distribution")

    if Path("class_distribution.png").exists():

        st.image(
            "class_distribution.png",
            use_container_width=True
        )

    else:

        st.info(
            "Run train.py to generate this graph."
        )

with graph2:

    st.write("#### Confusion Matrix")

    if Path("confusion_matrix.png").exists():

        st.image(
            "confusion_matrix.png",
            use_container_width=True
        )

    else:

        st.info(
            "Run train.py to generate this graph."
        )

# =========================================================
# MODEL DETAILS
# =========================================================

st.markdown("---")

st.subheader("⚙️ Model Details")

details1, details2, details3 = st.columns(3)

details1.info(
    "**Algorithm**\n\n"
    "Logistic Regression"
)

details2.info(
    "**Feature Extraction**\n\n"
    "TF-IDF"
)

details3.info(
    "**Categories**\n\n"
    + ", ".join(model.classes_)
)

# =========================================================
# PROJECT DESCRIPTION
# =========================================================

st.markdown("---")

st.subheader("📚 About the Project")

st.write("""
This project demonstrates a complete Natural Language Processing
workflow. Text data is converted into numerical TF-IDF features,
which are then used by a Logistic Regression classifier.

The project focuses on machine learning methodology rather than
clinical diagnosis.
""")

# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<div class="footer">

Built with Python • Pandas • Scikit-learn • Streamlit

<br>

Mental Health NLP Classification Project

</div>
""", unsafe_allow_html=True)
