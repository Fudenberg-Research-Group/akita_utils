import akita_utils.format_io
import glob
import bioframe
import pandas as pd
import numpy as np
import streamlit as st
import streamlit_pandas as sp
import plotly.express as px 


h5_dirs = "/home1/kamulege/akita_utils/bin/insert_promoter_experiment/data/promoter_scores_varying_gene_bp/*/*.h5" 

@st.cache(allow_output_mutation=True)
def get_data():
    dfs = []
    for h5_file in glob.glob(h5_dirs):
        dfs.append(akita_utils.format_io.h5_to_df(h5_file, drop_duplicates_key=None))    
    return pd.concat(dfs)
dfs = get_data()


# create_data = {"Name": "text",
#                 "Sex": "multiselect",
#                 "Embarked": "multiselect",
#                 "Ticket": "text",
#                 "Pclass": "multiselect"}

all_widgets = sp.create_widgets(dfs) #, create_data, ignore_columns=["PassengerId"])
res = sp.filter_df(dfs, all_widgets)
st.title("Insert Promoters Experiment")
st.header("Original DataFrame")
st.write(dfs)

st.header("Result DataFrame")
st.write(res)