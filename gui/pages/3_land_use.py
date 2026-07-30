import streamlit as st
from streamlit_theme import st_theme
import matplotlib.pyplot as plt
from agrifoodpy.land.land import LandDataArray
from altair_plots import plot_yearly

@st.fragment
def page_land_use():
    theme = st_theme()
    if theme is not None:
        background_color = theme.get("backgroundColor", "#ffffff")
    else:
        background_color = "#ffffff"

    db = st.session_state.datablock
    land = db["land"]

    col_plot, col_info = st.columns(2)

    with col_plot:
        year = st.select_slider(
            "Select year",
            options=list(land.Year.values),
            value=int(land.Year.values[-1]))

        land_year = land.sel(Year=year)
        land_dc = land_year.land.dominant_category(return_index=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        # land_dc.land.plot(ax=ax)
        ax.axis("off")
        fig.patch.set_facecolor(background_color)
        ax.set_ylim(0, land_dc.shape[0]-300)
        ax.imshow(land_dc, origin="lower", interpolation="none")
        plt.xticks(rotation=45)
        st.pyplot(fig, width=500)

    with col_info:
        land_categories = st.multiselect("Select land categories", land.aggregate_class.values)
        if land_categories == []:
            land_categories = land.aggregate_class.values
        land_totals = land.sel(aggregate_class=land_categories).sum(dim=["y", "x"])

        land_use_year_chart = plot_yearly(
            land_totals,
            "aggregate_class",
            "Land use by category")

        st.altair_chart(land_use_year_chart, width="stretch")

page_land_use()