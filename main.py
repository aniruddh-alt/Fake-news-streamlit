from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import streamlit as st
import joblib,os
from sklearn import *
import pandas as pd
fake=pd.read_csv("Fake.csv")
true=pd.read_csv("True.csv")
fake["class"]=0
true["class"]=1
df_marge = pd.concat([fake, true], axis =0 )
df = df_marge.drop(["title", "subject","date"], axis = 1)

tfvect = TfidfVectorizer(stop_words='english', max_df=0.7)
loaded_model = pickle.load(open('DT_MODEL', 'rb'))
dataframe = df
x =  dataframe["text"]
y = dataframe['class']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

def fake_news_det(news):
    tfid_x_train = tfvect.fit_transform(x_train)
    tfid_x_test = tfvect.transform(x_test)
    loaded_model.fit(tfid_x_train, y_train)
    input_data = [news]
    vectorized_input_data = tfvect.transform(input_data)
    prediction = loaded_model.predict(vectorized_input_data)
    return prediction


st.title("Fake News Classifier")
# st.subheader("ML App with Streamlit")
html_temp = """
<div style="background-color:blue;padding:10px">
<h1 style="color:white;text-align:center;">Streamlit ML App </h1>
</div>
"""
st.markdown(html_temp,unsafe_allow_html=True)
news_text=st.text_input("Enter Text")

if st.button("classify"):
    p=fake_news_det(news_text)

    if p==0:
        st.title("""FAKE NEWS""")
    else:
        st.title("""TRUE NEWS""")


