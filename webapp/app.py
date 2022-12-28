from pathlib import Path
import streamlit as st

# ---- MAINPAGE ----
st.title(":bar_chart: Akita_utils")
st.markdown("#visualizing various experiments#")

def read_markdown_file(markdown_file):
    """_summary_

    Args:
        markdown_file (_type_): _description_

    Returns:
        _type_: _description_
    """
    return Path(markdown_file).read_text()

intro_markdown = read_markdown_file("/Users/phad/akita_utils/README.md")
st.markdown(intro_markdown, unsafe_allow_html=True)


# ---- HIDE STREAMLIT STYLE ----
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)
