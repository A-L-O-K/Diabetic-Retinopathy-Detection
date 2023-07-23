import streamlit as st
import pandas as pd
from PIL import Image
import checker
import predictor as pred

st.title("Diabetics detection")
st.write("This is a simple image classification web app to predict whether a person is diabetic or not")
st.write("")
st.write("")


tab1, tab2, tab3 = st.tabs(["Diagnose", "Ask", "About"])

with tab1:
   # st.subheader("Diabetics level predictor")
   
   uploaded_file = st.file_uploader(" ", type=("jpg", "png", "jpeg"))
   
   
   col1, col2 = st.columns(2)
   
   with col1:
      if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', width=300)

        with col2:
            col11, col12 = st.columns(2)

            # with col11:
            #     diagnoseButton = st.button("Diagnose")

            
            
            # if diagnoseButton:
            prediction, percent = pred.predict(image)


            with col11:
               st.write("Result")

               st.write("")
               if prediction == 0:
                  st.success("Not Diabetic")  
               else:
                  st.error("Diabetic")
         

            with col12:
               if prediction == 0:
                  percent = ((percent - 0.5)/0.5) * 100
               else:
                  percent = ((0.5 - percent)/0.5) * 100

               st.bar_chart({"Reliability (%)": [percent]})  
               
               
         



   

with tab2:
   st.header("Ask Doubts")

   # age = st.slider("Slide to your age", 10, 120)

   question = st.text_input("", placeholder = "Question...")

   if question:
      answer = checker.check(question)
      st.write(answer)



with tab3:
   st.header("About")

   st.write("This is a tool to check whether a person has diabetics or not using his retina images")
   # st.write("")
   
   st.write("The reliability percentage of each prediction is shown in the bar chart")
   # st.write("")

   st.write("Users can also ask doubts about diabetics")
   st.write("The unknown questions will be stored in a file and will be answered by the admin")
   # st.write("")

   st.write("The model used for prediction of diabetic is made using CNN with the following dataset from kaggle")
   st.write("Diabetic Retinopathy (resized) - https://www.kaggle.com/datasets/tanlikesmath/diabetic-retinopathy-resized")
   # st.write("")

   st.write("The model used for question answering is a pretrained BERT model")
   

   



















css = '''
<style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    font-size:1rem;
    }
</style>
'''

st.markdown(css, unsafe_allow_html=True)
