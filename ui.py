import streamlit as st
import pandas as pd
import joblib

# ----- Page Config -----
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="💬",
    layout="wide"
)

# ----- Load Model -----
@st.cache_resource
def load_model():
    try:
        return joblib.load("sentiment_model.pkl")
    except:
        return None

model = load_model()

# ----- Title -----
st.title("💬 Sentiment Analysis Dashboard")
st.markdown("Analyze text sentiment with ease — **Positive**, **Negative**, or **Neutral**.")

# ----- Input -----
user_input = st.text_area("Enter your text:", placeholder="Type something here...")

# ----- Predict -----
if st.button("Analyze Sentiment"):
    if model:
        prediction = model.predict([user_input])[0]
        st.success(f"Sentiment: **{prediction}**")
    else:
        st.error("⚠ Model not found! Please train and save your model as 'sentiment_model.pkl'.")

# ----- Example Data -----
st.subheader("📊 Example Sentiment Data")
data = {
    "Text": ["I love this product!", "This is the worst experience ever.", "It’s okay, not great."],
    "Sentiment": ["Positive", "Negative", "Neutral"]
}
df = pd.DataFrame(data)
st.table(df)