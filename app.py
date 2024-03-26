import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load("forest.pkl")


st.set_page_config("wide")


with st.sidebar:
    st.header(":blue[About Me] :man:")
    st.write("I am an AI and Data Science Student. Passionate about Data science and ML")
    github_emoji = "\U0001F680"
    github_link = f"[Github Profile {github_emoji}](https://github.com/BHEESETTIANAND)"
    st.markdown(github_link, unsafe_allow_html=True)
    st.write("To see my work, please visit the link to my portfolio below.")
    portfolio_link = "https://anandbheesetti.wixsite.com/portfolio"
    st.markdown(portfolio_link, unsafe_allow_html=True)
    gmail_emoji = "\U0001F4E7"
    st.markdown(f"email me at {gmail_emoji}")
    st.write("anandbheesetti@gmail.com")

st.title('Sentiment Analyzer')
user_input = st.text_input("enter you comment")

positive_emoji = "😊"  
negative_emoji = "😞" 



if st.button("Predict"):
    pred=model.predict([user_input])
    if pred==1:
        st.write("Positive sentiment:", positive_emoji)
    else:
        st.write("Negative sentiment:", negative_emoji)
    
        