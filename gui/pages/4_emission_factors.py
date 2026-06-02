import streamlit as st

@st.fragment
def page_emission_factors():
    db = st.session_state.datablock

    emissions = db["emissions"]

    st.write(emissions)

page_emission_factors()