import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# --- PAGE CONFIG ----

st.set_page_config(page_title="Spotify Dashboard",
page_icon=" :bar_chart:",
layout="wide")


# --- READING DATA ---

df = pd.read_csv("Fixed_Dataset.csv", index_col="release_date")

key_mapping = {
    0: 'C', 1: 'C#', 2: 'D', 3: 'D#', 4: 'E', 5: 'F',
    6: 'F#', 7: 'G', 8: 'G#', 9: 'A', 10: 'A#', 11: 'B'
}

df['key'] = df['key'].map(key_mapping)

# It would have been good if dataset had country of release column

# --- SIDEBAR ---

st.sidebar.header("Please Filter Here:")

year_range_select = st.sidebar.slider(
    "Select the Years:", min_value = 1900, max_value = 2021, value = [1900, 2021]
)


artists_select = st.sidebar.multiselect(
    "Select Artists:",
    options = df["artists"].unique(),
)

only_all_button = st.sidebar.radio(
    "Select the way to filter artists",
    ["Only", "All"]
)
sort_button = st.sidebar.radio(
    "Select the way to sort results",
    ["Date", "Popularity"]
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

st.dataframe(df.describe(include="all"))

nosofsongs = df_select.shape[0]
st.write(f"Number of songs in current selection: {nosofsongs}")

popularsongsdf = df_select[df_select["popularity"] > 75].sort_values(by = "popularity",ascending=False)
popularsongsdf = popularsongsdf[["year", "name", "artists", "popularity"]]
st.dataframe(popularsongsdf)

corr_df = df_select.drop(["name", "artists", "key", "mode", "explicit"], axis = 1).corr(method = "pearson")
corr = plt.figure(figsize = (14,6))
heatmap = sns.heatmap(corr_df,annot = True, fmt = ".1g", vmin = -1, vmax = 1, center = 0, cmap = "inferno", linewidths=1, linecolor = "Black")
heatmap.set_title("Correlation HeatMap between Variables")
heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation = 90)

st.pyplot(corr)


#Creating Line Graphs for understanding the evolution of musical elements with time

sp1, elements = plt.subplots(3, 2) # Initiating subplots

arrElements = ["danceability", "valence", "energy", "speechiness", "acousticness",  "instrumentalness"]
row = 0
col = 0

for element in arrElements: # Creating a line graph for each musical element
    df_select.groupby("year")[element].mean().plot(ax=elements[row][col], figsize=(30,20), color="#bd062a", linewidth=2.5) # Group rows by year and get the mean values for that particular element
    elements[row][col].set_xlabel("decade", fontsize = 30)
    elements[row][col].set_ylabel(element, fontsize = 30)
    elements[row][col].tick_params(labelsize=30)
    elements[row][col].tick_params(labelsize=30)
    if col == 0:
        col = 1
    elif row == 0 and col == 1:
        row = 1
        col = 0
    elif row == 1 and col == 1:
        row = 2
        col = 0

st.pyplot(sp1)


#Number of explicit songs growth by year

"""
explicits = df_select.groupby("explicit")["year"].value_counts() # Get the count between explicit and non-explicit songs by year
df_explicits=pd.DataFrame([explicits[0], explicits[1]], index=["not explicit", "explicit"])# Convert to DataFrame
df_new = df_explicits.T
df_new['year']= df_new.index
df_new.index = np.arange(0, len(df_new))

figexpli = px.scatter(df_new, y='explicit',x='year', title="Growth in Explicit Songs")
st.plotly_chart(figexpli)
"""

explicits = df_select.groupby("explicit")["year"].value_counts().unstack(fill_value=0)

# Convert to DataFrame
df_new = explicits.reset_index()
df_new.columns.name = None  # Remove the columns name

# Reshape DataFrame for plotting
df_new = pd.melt(df_new, id_vars=['explicit'], var_name='year', value_name='count')

# Create the scatter plot
figexpli = px.scatter(df_new, x='year', y='count', color='explicit', title="Growth in Explicit Songs")
st.plotly_chart(figexpli)

# Is there any difference in popularity between explicit and non-explicit songs

expo = px.box(df_select, x='explicit', y='popularity',
                 title="Popularity Distribution by Explicit Content")
expo.update_layout(xaxis_title="Explicit", yaxis_title="Popularity",
                      template="plotly_white")
st.plotly_chart(expo)


"""
figdur = px.line(df_select, x='year', y='duration', title="Years vs Duration")
figdur.update_layout(xaxis_title="Year", yaxis_title="Duration",
                  xaxis_tickangle=60, template="plotly_white")

st.plotly_chart(figdur)
"""

if len(artists_select)>=1:

    df_long = df_select.melt(id_vars='year',
                             value_vars=['acousticness', 'danceability', 'energy',
                                         'instrumentalness', 'speechiness', 'valence'],
                             var_name='attribute', value_name='magnitude')

    # Create the scatter plot with trendlines
    trends = px.scatter(df_long, x='year', y='magnitude', color='attribute',
                     title='Acousticness, Danceability, Energy, Instrumentalness, Speechiness, and Valence Over the Years',
                     trendline='ols',
                     labels={'year': 'Years', 'magnitude': 'Magnitude'})

    # Update layout for better visuals
    trends.update_layout(
        template='plotly_white',
        legend_title='Attributes'
    )

    st.plotly_chart(trends)

else:
    pass

key_df = df_select.groupby('key').agg({'year': 'count'}).reset_index()
#st.dataframe(key_df)
fig_key = px.bar(key_df, x='key', y='year',
             title='Songs by Key',
             labels={'key': 'Key', 'year': 'Songs'},
             color='year')

fig_key.update_layout(template='plotly_white', xaxis_title='Key', yaxis_title='Number of Songs')
st.plotly_chart(fig_key)

key_pop = px.box(df_select, x="key", y='popularity',
                 title="Popularity Distribution by Key")
key_pop.update_layout(xaxis_title="Key", yaxis_title="Popularity",
                      template="plotly_white")
st.plotly_chart(key_pop)