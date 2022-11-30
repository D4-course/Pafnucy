# import the streamlit library
import streamlit as st
from pathlib import Path
import json
import requests
import pandas as pd
import zipfile
import os


# Base URL for API Calls
BASE_URL="http://localhost:8000/api/v1/"

# give a title to our app
st.title('Welcome to Pafnucy Tool')
st.write('Pafnucy [paphnusy] is a 3D convolutional neural network that predicts binding affinity for protein-ligand complexes')
# Upload Ligand File
ligand_file = st.file_uploader("Upload Input Ligand file", type=["mol2"])
# Upload pockets File
pocket_file = st.file_uploader("Upload Input Pocket file", type=["mol2"])


predict = st.button('Predict')

# check if the button is pressed or not
if(predict):
    if(ligand_file and pocket_file):
        st.success("Request Submitted")
        # Upload two input files to backend
        files = {'file1': ligand_file, 'file2': pocket_file}
        # files = {"file": protien_file.getvalue()}

        with st.spinner("Please Wait.."):
            # Raise the post request
            res = requests.post(f"{BASE_URL}score/", files=files)
            # display response
            df = pd.read_csv('predictions.csv')
            st.write(df)
            st.info('Note: "name" is the name of the input ligand, "prediction" is the binding affinity score i.e -log(Kd) predicted by the model')


    else:
        st.error("Please Input all the files")



