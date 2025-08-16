import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis")
nlp=load_model()
st.set_page_config(page_title="Sentiment Analytics Dashboard", layout="wide")
st.markdown("""
<style>
    .main{
        padding: 2rem;
    }
    h1,h2,h3 {
        font-family: 'Arial',sans-serif;
    }
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)
st.markdown("<h1 style='color:#2E86C1;'>Sentiment Analytics Dashboard</h1>",unsafe_allow_html=True)
st.write("Analyze text sentiment in real-time with clean analytics")
st.markdown("---")
user_input=st.text_area("Enter text to analyze sentiment:",height=150)
if "results" not in st.session_state:
    st.session_state.results=[]
if st.button("Analyze Sentiment"):
    if user_input.strip():
        result=nlp(user_input)[0]
        label=result['label']
        score=result['score']
        st.session_state.results.append({"Text":user_input, "Sentiment":label, "Score":round(score,2)})
        if label=="POSITIVE":
            color="#27AE60"
        elif label=="NEGATIVE":
            color="#C0392B"
        else:
            color="#7F8C8D"
        st.markdown(
            f"<div style='padding:15px;background-color:{color};color:white;border-radius:10px;'>"
            f"<h3>Sentiment: {label} ({score:.2f})</h3></div>", 
            unsafe_allow_html=True
        )
    else:
        st.warning("Please enter some text!")
if st.session_state.results:
    df=pd.DataFrame(st.session_state.results)
    col1, col2=st.columns(2)
    with col1:
        st.subheader("Sentiment History")
        st.dataframe(df)
    with col2:
        st.subheader("Sentiment Distribution")
        fig, ax=plt.subplots()
        df["Sentiment"].value_counts().plot.pie(
            autopct="%1.1f%%", ax=ax, colors=["#27AE60", "#C0392B", "#7F8C8D"]
        )
        ax.set_ylabel("")
        st.pyplot(fig)
st.markdown("---")
st.markdown("<p style='text-align:center;color:gray;'>© 2025 Sentiment Analytics Dashboard</p>", unsafe_allow_html=True)
st.set_page_config(page_title="Sentiment Dashboard", page_icon="💬", layout="wide")



