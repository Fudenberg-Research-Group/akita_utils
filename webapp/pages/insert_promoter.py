import glob

import bioframe
import numpy as np
import pandas as pd
import pandas_profiling
import plotly.express as px
import streamlit as st
import streamlit_pandas as sp
from streamlit_pandas_profiling import st_profile_report

import akita_utils.format_io

h5_dirs_1 = "/Users/phad/akita_utils/bin/insert_promoter_experiment/data/promoter_scores_no_swap/*/*.h5"
h5_dirs_2 = "/Users/phad/akita_utils/bin/insert_promoter_experiment/data/promoter_scores_with_swap/*/*.h5"

@st.cache(allow_output_mutation=True)
def get_data():
    dfs = []
    for directory in [h5_dirs_1,h5_dirs_2]:
        for h5_file in glob.glob(directory):
            dfs.append(akita_utils.format_io.h5_to_df(h5_file, drop_duplicates_key=None)) 
    dfs = pd.concat(dfs)
    
    dfs["mean_SCD_score"] = (dfs["SCD_h1_m1_t0"]+dfs["SCD_h1_m1_t1"]+dfs["SCD_h1_m1_t2"]+dfs["SCD_h1_m1_t3"]+dfs["SCD_h1_m1_t4"]+dfs["SCD_h1_m1_t5"])/6
    
    dfs = dfs.drop(columns=['background_seqs']) # droping because it is constant
    return dfs
dfs = get_data()


create_data = {
                "gene_id": "multiselect",
                "locus_orientation": "multiselect",
                "insert_strand": "multiselect"}

all_widgets = sp.create_widgets(dfs, create_data, ignore_columns=["gene_locus_specification","insert_loci","out_folder","ctcf_strand","gene_strand","ctcf_locus_specification"])

res = sp.filter_df(dfs, all_widgets)
st.title("Insert Promoters Experiment")
st.header(f"Original DataFrame n={dfs.shape[0]}")
st.dataframe(dfs)

st.header(f"Result DataFrame n={res.shape[0]}")
st.write(res)


st.markdown("""---""")

operation = st.radio(
    "ANALYSIS CHOICE",
    ('visualize important plots','profile selected dataset' ))

if operation == 'profile selected dataset':
    pr = res.profile_report()
    st_profile_report(pr)
else:
    # FLANKS BOX PLOT 
    fig_flanks = px.box(
        res,
        x="flank_bp",
        y="mean_SCD_score",
        title="<b>Effect of flanks to SCD score</b>",
        color_discrete_sequence=["#0083B8"] * res.shape[0],
        template="plotly_white",
    )
    fig_flanks.update_layout(
        # xaxis=dict(tickmode="linear"),
        plot_bgcolor="rgba(0,0,0,0)",
        # yaxis=(dict(showgrid=False)),
    )

    st.plotly_chart(fig_flanks)


    # SPACERS BOX PLOT 
    fig_spacer = px.box(
        res,
        x="spacer_bp",
        y="mean_SCD_score",
        title="<b>Effect of spacing to SCD score</b>",
        color_discrete_sequence=["#0083B8"] * res.shape[0],
        template="plotly_white",
        # points="all"
    )
    fig_spacer.update_layout(
        # xaxis=dict(tickmode="linear"),
        # plot_bgcolor="rgba(0,0,0,0)",
        # yaxis=(dict(showgrid=False)),
    )

    st.plotly_chart(fig_spacer)


    fig = px.scatter_3d(res,
        x="spacer_bp",
        y="flank_bp",
        z="mean_SCD_score",
        color = "mean_SCD_score" )
    st.plotly_chart(fig)


    # ---- HIDE STREAMLIT STYLE ----
hide_st_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)