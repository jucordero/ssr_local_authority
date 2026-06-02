import streamlit as st
from altair_plots import plot_yearly

@st.fragment
def page_population():

    db = st.session_state.datablock

    pop = db["population"]
    c = plot_yearly(
        pop,
        ylabel="[1000 people]",
        mark_total=True,
        )

    st.altair_chart(c, width="stretch")

page_population()