import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st


st.set_page_config(page_title="Spotify Dashboard",
page_icon=" :bar_chart:",
layout="wide")

@st.cache_data
def get_data_from_csv():
    df = pd.read_csv("Fixed_Dataset2.csv", index_col="release_date"),
    return df

df = get_data_from_csv()


# ----- SIDEBAR ------

st.sidebar.header("Please Filter Here:")

year_range_select = st.sidebar.slider(
    "Select the Years:", min_value = 1900, max_value = 2021, value = [1900, 2021]
)
"""
only_all_button = st.sidebar.radio(
    "Select the way to filter artists",
    ["Only", "All"]
)

artists_select = st.sidebar.multiselect(
    "Select Artists:",
    options = df["artists"].unique(),
)
"""
"""
artists_select = st.sidebar.text_input(
   "Select Artists separated by |"
)
st.sidebar.write(artists_select)


key_select  = st.sidebar.multiselect(
    "Select Key:",
    options = df["key"].unique(),
    default = df["key"].unique(),
)

popularity_select = st.sidebar.slider(
    "Select Popularity:",
    min_value=0,
    max_value= 100,
    value=[0,100]
)

explicit_select = st.sidebar.multiselect(
    "Explicit Songs:",
    options = [0,1],
    default= [0,1]
)
"""
#st.sidebar.write("You selected:", artists_select)


# ----- MAIN DASHBOARD ------
"""
df_select = df.query("year >= @year_range_select[0] & year <= @year_range_select[1]"
                     "& key == @key_select "
                     #"& @artists_select in artists"
                     "& popularity >= @popularity_select[0] & popularity <= @popularity_select[1]"
                     "& explicit == @explicit_select"
                    )

df_select = df_select[df_select["artists"].str.contains(f"{artists_select}")]

st.dataframe(df_select.sort_values(by="year"))
"""
st.dataframe(df)
st.dataframe(df.describe())