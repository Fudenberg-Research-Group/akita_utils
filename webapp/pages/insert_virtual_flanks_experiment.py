import akita_utils.format_io
import bioframe
import pandas as pd
import numpy as np
import streamlit as st
import streamlit_pandas as sp
import plotly.express as px 


h5_file = "/Users/phad/akita_utils/bin/insert_virtual_flanks_experiment/data/2022-09-30_flank0-30_10motifs_left.h5" 

@st.cache(allow_output_mutation=True)
def get_data(): 
    scd_stats = ["SCD", "INS-16"]
    return akita_utils.h5_to_df(h5_file, scd_stats, drop_duplicates_key=None)
dfs = get_data()


create_data = {
                "chrom": "multiselect",
                "orientation": "multiselect",
                "strand": "multiselect"}

all_widgets = sp.create_widgets(dfs, create_data, ignore_columns=["end","experiment_id"])
res = sp.filter_df(dfs, all_widgets)
st.title("insert virtual flanks experiment")
st.header(f"Original DataFrame n={dfs.shape[0]}")
st.write(dfs)

st.header(f"Result DataFrame n={res.shape[0]}")
st.write(res)

st.markdown("""---""")

# FLANKS BOX PLOT 
fig_flanks = px.box(
    res,
    x="flank_bp",
    y="genomic_SCD",
    title="<b>flanks</b>",
    color_discrete_sequence=["#0083B8"] * res.shape[0],
    template="plotly_white",
)
fig_flanks.update_layout(
    # xaxis=dict(tickmode="linear"),
    # plot_bgcolor="rgba(0,0,0,0)",
    # yaxis=(dict(showgrid=False)),
)

st.plotly_chart(fig_flanks)



# SPACERS BOX PLOT 
fig_spacer = px.box(
    res,
    x="spacer_bp",
    y="genomic_SCD",
    title="<b>spacing</b>",
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


fig = px.scatter_3d(    res,
    x="spacer_bp",
    y="flank_bp",
    z="genomic_SCD",
    color = "genomic_SCD" )
st.plotly_chart(fig)


# ---- HIDE STREAMLIT STYLE ----
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)