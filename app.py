import streamlit as st
import pandas as pd
from pycaret.classification import load_model, predict_model
from streamlit_lottie import st_lottie
from PIL import Image
import requests
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Insurance technical assessment ", page_icon=":tada:", layout="wide")


def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


# ---- LOAD ASSETS ----
lottie_coding = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_fcfjwiyb.json")

# ---- HEADER SECTION ----

with st.container():
    left_column, right_column = st.columns(2)
    with left_column:
        st.subheader("Hi, I am Ignacio :wave:")
        st.write("##")
        st.title("A newbie Data Analyst From Barcelona")
        st.write("I specialize in data analysis, data visualization, machine learning, and possess in-depth knowledge of cloud technologies and SQL. My passion lies in transforming data into actionable insights to drive strategic decision-making and organizational growth.")
        st.write( 'Welcome to my first project in Streamlit, an app to predict the price with AIRBNB service in Roma')
    with right_column:
        st_lottie(lottie_coding, height=300, key="coding")


# Model
model = load_model("trained_model")
st.header("Insurance Prediction System")
st.write('Check if you are eligible to request our services')



genero = st.slider('Enter your gender: 0 for Female ,1 for Male', min_value=0, max_value=1, value=0)
income = st.slider('Enter monthly income: ', min_value=0, max_value=100000, value=0)
profesion = st.slider('Enter your profession: Working:4, Commercial:0,Pensioner:1, servant:2, Student:3', min_value=0, max_value=4, value=0)
educacion = st.slider('Enter the highest type of education you have achieved: Secondary / secondary special :4, Higher education: 1, Incomplete higher: 2, Lower secondary: 3, Academic degree: 0', min_value=0, max_value=4, value=0)
estado_civil = st.slider('Enter marital status: Married: 1, Single:3,Civil marriage:0, Separated:2, Widow:4', min_value=0, max_value=4, value=0)
casa = st.slider('Enter the type of housing: House / apartment:1, Rented apartment:4, With parents:5, Municipal apartment:2, Co-op apartment:0, Office apartment:3',  min_value=0, max_value=5, value=0)
empleo = st.slider('Are you currently working?: 0 for No, 1 for Yes', min_value=0, max_value=1, value=0)
edad = st.slider('Enter your age: ', min_value=0, max_value=100, value=0)




input_data= pd.DataFrame([[genero,income,profesion,educacion,estado_civil,casa,empleo,edad]],columns=['CODE_GENDER', 'AMT_INCOME_TOTAL', 'NAME_INCOME_TYPE',
       'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE',
       'Empleo', 'Edad'])


if st.button('Get matched with us'):
    prediction = model.predict(input_data)
    if prediction == 0: 
        st.write('Sorry, you do not fit the requirements, go to our website and discover our other services')
    else:   
        st.write('Congratulations, you are the perfect client, contact us.')



