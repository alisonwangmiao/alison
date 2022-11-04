# Author: Alison Mengnan Wang
# Aug 2021

import streamlit as st
# other libs
import numpy as np
import pandas as pd
import pickle
# import pyautogui # for reset button: pip install pyautogui


# load the logistic.pkl
# path = r'D:\work\courses\SI-Solution Implementation\SI\code\streamlit\app3\logistic.pkl'
with open('logistic.pkl', "rb") as f:
	logistic = pickle.load(f)

# Streamlit provides a caching mechanism that allows your app to stay performant 
# even when loading data from the web, manipulating large datasets, 
# or performing expensive computations. This is done with the @st.cache decorator.
@st.cache()


#dataset = pd.read_csv("Social_Network_Ads.csv")

def prediction( Age, EstimatedSalary):
	# Making predictions
	prediction = logistic.predict([[ Age, EstimatedSalary]])
	if prediction == 0:
		pred = 'Not Buy'
	else:
		pred = 'Buy'
	return pred


# putting the app related codes in main()
def main():
	# -- Set page config
	apptitle = 'DSSI'
	st.set_page_config(page_title=apptitle, page_icon='random', 
		layout= 'wide', initial_sidebar_state="expanded")
	# random icons in the browser tab

	# give a title to your app
	st.title('Solution Implementation')
	#front end elements of the web page 
	# pick colors from: https://www.w3schools.com/tags/ref_colornames.asp
	html_temp = """ <div style ="background-color:AntiqueWhite;padding:15px"> 
       <h1 style ="color:black;text-align:center;">A SUV Social Network Ads application assessment app</h1> 
       </div> <br/>"""

    #display the front end aspect
	st.markdown(html_temp, unsafe_allow_html = True)
	# let us make infrastructure to provide inputs
	# we will add the inputs to side bar
	st.sidebar.info('Provide input using the panel')
	st.info('Click Assess button below')


	Age = st.sidebar.slider('Age', 0, 60, 5)
	st.write('input Age', Age)
	EstimatedSalary = st.sidebar.slider('EstimatedSalary yearly', 10000, 200000, 50000)
	st.write('input EstimatedSalary', EstimatedSalary*1000)
	
	result =""
	# assessment button
	if st.button("Predict"):
		assessment = prediction( Age, EstimatedSalary)
		st.success('**System assessment says:** {}'.format(assessment))

	# if st.button("Reset"):
	# 	pyautogui.hotkey("ctrl","F5")

	# st.balloons()
	st.success("App is working!!") # other tags include st.error, st.warning, st.help etc.

if __name__ == '__main__':
	main()

# save model in a pickle format .pkl


import pickle
pickle.dump(logistic, open('logistic.pkl', 'wb'))