import streamlit as st
import base64

# working on viewing pdfs in webapp

def show_pdf(file_path):
    with open(file_path,"rb") as f:
        base64_pdf = base64.b64encode(f.read())
        base64_pdf = base64.b64decode(base64_pdf)
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="800" height="800" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

show_pdf('/Users/phad/akita_utils/bin/background_seq_experiments/data/background_seqs/job0/seq_1.pdf')
