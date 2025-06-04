"""Simple Streamlit app to visualize training logs."""
import pandas as pd
import streamlit as st

LOG_PATH = "training.log"

@st.cache_data
def load_logs(path: str = LOG_PATH):
    data = pd.read_csv(path, names=["episode", "reward", "pred_reward"])
    return data

data = load_logs()
st.title("Training Progress")
st.line_chart(data["reward"], height=200)
st.line_chart(data["pred_reward"], height=200)
