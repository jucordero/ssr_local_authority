import streamlit as st

@st.fragment
def page_inspect_datablock():
    st.title("Inspect Datablock")
    st.write("This page allows you to inspect the contents of the datablock.")
    
    # Display the contents of the datablock
    st.subheader("Datablock Contents")
    st.write(st.session_state.datablock)

page_inspect_datablock()