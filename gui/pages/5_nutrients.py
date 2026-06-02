import streamlit as st

@st.fragment
def page_nutrients():
    db = st.session_state.datablock

    nutrients = db["nutrients"]

    st.write(nutrients)

page_nutrients()