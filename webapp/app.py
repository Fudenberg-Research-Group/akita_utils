from pathlib import Path

import streamlit as st

# ---- MAINPAGE ----
st.title(":bar_chart: Akita_utils")
st.markdown("#visualizing various experiments#")

current_file_path = Path(__file__)
intro_markdown = current_file_path.parents[1] / "README.md"

st.markdown(intro_markdown.read_text(), unsafe_allow_html=True)

# ---- HIDE STREAMLIT STYLE ----
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)
