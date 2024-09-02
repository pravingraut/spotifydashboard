import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st

# --- PAGE CONFIG ----

st.set_page_config(page_title="Spotify Dashboard",
page_icon=" :bar_chart:",
layout="wide")


# --- READING DATA ---

df = pd.read_csv("Fixed_Dataset.csv", index_col="release_date")


# --- SIDEBAR ---

st.sidebar.header("Please Filter Here:")

year_range_select = st.sidebar.slider(
    "Select the Years:", min_value = 1900, max_value = 2021, value = [1900, 2021]
)


only_all_button = st.sidebar.radio(
    "Select the way to filter artists",
    ["Only", "All"]
)
sort_button = st.sidebar.radio(
    "Select the way to sort results",
    ["Date", "Popularity"]
)
st.sidebar.write(sort_button)

artists_select = st.sidebar.multiselect(
    "Select Artists:",
    options = df["artists"].unique(),
)

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


# ----- MAIN DASHBOARD ------

def sort_and_display (x):
    if x == "Date":
        st.dataframe(df_select.sort_values(by="year"))
    else:
        st.dataframe(df_select.sort_values(by="popularity", ascending=False))

if len(artists_select) == 0:
    df_select = df.query("year >= @year_range_select[0] & year <= @year_range_select[1]"
                             "& key == @key_select"
                             "& popularity >= @popularity_select[0] & popularity <= @popularity_select[1]"
                             "& explicit == @explicit_select"
                        )

    sort_and_display(sort_button)
else:
    if only_all_button == "Only":
        df_select = df.query("year >= @year_range_select[0] & year <= @year_range_select[1]"
                         "& key == @key_select "
                         "& artists == @artists_select"
                         "& popularity >= @popularity_select[0] & popularity <= @popularity_select[1]"
                         "& explicit == @explicit_select"
                        )
        sort_and_display(sort_button)

    else:
        df_select = df.query("year >= @year_range_select[0] & year <= @year_range_select[1]"
                             "& key == @key_select "
                             #"& artists == @artists_select"
                             "& popularity >= @popularity_select[0] & popularity <= @popularity_select[1]"
                             "& explicit == @explicit_select"
                             )
        df_select = df_select[df_select["artists"].str.contains('|'.join(artists_select), regex = True)]

        sort_and_display(sort_button)

st.dataframe(df.describe())