import glob
from pathlib import Path

import bioframe
import numpy as np
import pandas as pd
import pandas_profiling
import plotly.express as px
import streamlit as st
import streamlit_pandas as sp
from streamlit_pandas_profiling import st_profile_report

import akita_utils.format_io

st.set_page_config(layout="wide")

current_file_path = Path(__file__)
h5_dirs_1 = current_file_path.parents[2] / "bin/insert_promoter_experiment/data/promoter_scores_no_swap/*/*.h5"
h5_dirs_2 = current_file_path.parents[2] / "bin/insert_promoter_experiment/data/promoter_scores_with_swap/*/*.h5"

h5_dirs_1 = f"{h5_dirs_1}"
h5_dirs_2 = f"{h5_dirs_2}"

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
                "insert_strand": "multiselect",
                "swap_flanks": "multiselect",
                "ctcf_genomic_score": "multiselect"}

all_widgets = sp.create_widgets(dfs, create_data, ignore_columns=["gene_locus_specification","insert_loci","out_folder","ctcf_strand","gene_strand","ctcf_locus_specification","insert_flank_bp"])

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

    col_1, col_2 = st.columns((2,2))   
    col_3, col_4 = st.columns((2,2)) 
    # CTCF FLANKS BOX PLOT 
    fig_ctcf_flanks = px.box(
        res,
        x="ctcf_flank_bp",
        y="mean_SCD_score",
        title="<b>Effect of ctcf flanks to SCD score</b>",
        color_discrete_sequence=["#0083B8"] * res.shape[0],
        template="plotly_white",
    )
    fig_ctcf_flanks.update_layout(
        # xaxis=dict(tickmode="linear"),
        plot_bgcolor="rgba(0,0,0,0)",
        # yaxis=(dict(showgrid=False)),
    )

    col_1.plotly_chart(fig_ctcf_flanks, use_container_width=True)


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


    col_2.plotly_chart(fig_spacer, use_container_width=True) 

    

    # GENE FLANKS BOX PLOT 
    fig_gene_flanks = px.box(
        res,
        x="gene_flank_bp",
        y="mean_SCD_score",
        title="<b>Effect of gene flanks to SCD score</b>",
        color_discrete_sequence=["#0083B8"] * res.shape[0],
        template="plotly_white",
    )
    fig_gene_flanks.update_layout(
        # xaxis=dict(tickmode="linear"),
        plot_bgcolor="rgba(0,0,0,0)",
        # yaxis=(dict(showgrid=False)),
    )

    col_3.plotly_chart(fig_gene_flanks,  use_container_width=True)


    #  ORIENTATION BOX PLOT 
    fig_orientation = px.box(
        res,
        x="locus_orientation",
        y="mean_SCD_score",
        title="<b>Effect of locus orienation to SCD score</b>",
        color_discrete_sequence=["#0083B8"] * res.shape[0],
        template="plotly_white",
    )
    fig_orientation.update_layout(
        # xaxis=dict(tickmode="linear"),
        plot_bgcolor="rgba(0,0,0,0)",
        # yaxis=(dict(showgrid=False)),
    )

    col_4.plotly_chart(fig_orientation,  use_container_width=True)


    fig = px.scatter_3d(res,
        x="spacer_bp",
        y="ctcf_flank_bp",
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