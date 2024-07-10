#WEB APP - https://appml-fye4y2scjmgholkh2mkqnr.streamlit.app/

import pandas as pd
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import pickle
import scipy.fftpack
import pywt

from datetime import datetime
from datetime import date
import holidays
import pandas as pd
from datetime import timedelta
from holidays import country_holidays


import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import streamlit as st
import io

from model_prediction import *

def model_serve(usage,acct):    
    if usage.shape[0]!=0 and acct.shape[0]!=0:
        iv = pd.read_csv('./iv/iv.csv')
        cutoffs = pd.read_csv("./ks/KS Table train_cutoffs.csv")
        master_data_pred = model_prediction_main(usage,acct,iv,cutoffs)
        print(master_data_pred.head())
        #st.session_state.clicked = True

        st.session_state.pred = master_data_pred

        return master_data_pred

st.title('Account Conversion Prediction Model API')

uploaded_files = st.file_uploader("Upload Account Level Identifier and Usage Files",accept_multiple_files=True,type=['xlsx'])

if 'acct' not in st.session_state:
    st.session_state.acct = pd.DataFrame()

if 'usage' not in st.session_state:
    st.session_state.usage = pd.DataFrame()

col1, col2 = st.columns([1,1])

with col1:
    if 'acct' in st.session_state and 'usage' in st.session_state:
        st.button('Predict Conversion Likelihood', on_click=model_serve,args=(st.session_state.usage,st.session_state.acct))


if 'pred' in st.session_state:
    @st.cache_data
    def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
        return df.to_csv().encode("utf-8")
    
    predicted = st.session_state.pred
    csv = convert_df(predicted)

    with col2:
        st.download_button(
            label="Download predicted file as CSV",
            data=csv,
            file_name="predicted.csv",
            mime="text/csv",
        )



for i in uploaded_files:
    if i is not None:
        # Can be used wherever a "file-like" object is accepted:
        dataframe = pd.read_excel(i)
        if 'Acct type' in dataframe.columns:
            st.session_state['acct']=dataframe
            st.session_state['name_file']=i.name
        else:
            st.session_state['usage']=dataframe
            st.session_state['name_file']=i.name            

if 'clicked' not in st.session_state:
    st.session_state.clicked = False



if st.session_state.acct.shape[0]==0 or st.session_state.usage.shape[0]==0:
    st.text("Files not uploaded")
else:
    buffer = io.StringIO()
    st.session_state['acct'].info(buf=buffer)
    s1 = buffer.getvalue()

    buffer = io.StringIO()
    st.session_state['usage'].info(buf=buffer)
    s2 = buffer.getvalue()

    st.text(f"Processed data Account {s1}")
    st.text(f"Processed data Usage {s2}")
