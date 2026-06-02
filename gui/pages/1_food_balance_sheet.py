import streamlit as st
from altair_plots import plot_bars_altair2, plot_yearly
from agrifoodpy.food.food import FoodBalanceSheet

@st.fragment
def page_food_balance_sheet():
    db = st.session_state.datablock

    fbs = db["food"]

    col_cap1, col_cap2, col_cap3 = st.columns(3)
    with col_cap1:
        option_key = st.selectbox(
            "Chart type",
            ["FAOSTAT-style chart", "Yearly chart"]
            )

    with col_cap2:
        dissagregation = st.selectbox(
            "Disaggregation",
            ["Item_origin", "Item_group", "Item_name"]
            )

    with col_cap3:
        item_list = st.multiselect(
            "Items",
            (fbs[dissagregation].values)
            )


    item_selection = {}
    if len(item_list) > 0:
        item_selection = {dissagregation: item_list}

    fbs = fbs.fbs.group_sum(coordinate=dissagregation, new_name=dissagregation)
    fbs = fbs.sel(item_selection)

    if option_key == "FAOSTAT-style chart":
        year = st.slider("Year", 2020, 2100, 2100)
        fbs_chart = plot_bars_altair2(
            fbs.sel(Year=year),
            data_vars=["production", "imports"],
            reversed_vars=["exports", "stock", "feed", "seed", "losses", "other", "processing", "tourist", "food"],
            show=dissagregation,
            x_axis_title="Quantity",
            stacked=True,
            horizontal=True
        )


    elif option_key == "Yearly chart":
        element = st.selectbox(
            "Element",
            [
                "production",
                "imports",
                "exports",
                "stock",
                "food",
                "processing",
                "tourist",
                "feed",
                "seed",
                "losses",
                "other"
            ]
        )
        fbs_chart = plot_yearly(
            fbs[element],
            show=dissagregation
        )

    st.altair_chart(fbs_chart, width="stretch")

page_food_balance_sheet()