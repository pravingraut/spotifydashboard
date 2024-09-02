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
page_icon="🎶",
layout="wide")


# --- READING DATA ---

@st.cache_data
def get_data():
    df = pd.read_csv("Fixed_Dataset.csv", index_col="release_date")

    key_mapping = {
        0: 'C', 1: 'C#', 2: 'D', 3: 'D#', 4: 'E', 5: 'F',
        6: 'F#', 7: 'G', 8: 'G#', 9: 'A', 10: 'A#', 11: 'B'
    }
    df['key'] = df['key'].map(key_mapping)
    return df

df = get_data()



# It would have been good if dataset had country of release column.
# For example, we have an artist named KK. But there is another artist with the same name in other country.
# So filtering becomes difficult.

# --- SIDEBAR ---

st.sidebar.image(image="logo.jpg")

st.sidebar.header("Please Filter Here:")

year_range_select = st.sidebar.slider(
    "Select the Years:", min_value = 1900, max_value = 2021, value = [1900, 2021]
)


artists_select = st.sidebar.multiselect(
    "Select Artists:",
    options = df["artists"].unique(),
)
artstr = ", ".join(artists_select) # unpacking a list into a string
artsep = artstr.split(sep=", ") # Having all selected artists as separate elements in a list
#st.sidebar.write(artsep)

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

st.title("Spotify Data Analysis")
st.markdown("####")

# Three boxes that will show: Artists selected, Average Popularity in stars, Number of Songs.

left, middle, right = st.columns(3, gap="medium")

st.markdown("#####")

if len(artists_select) == 0:
    st.subheader(f"Songs of All artists", divider=True)
else:
    st.subheader(f"Songs of {artstr}", divider=True)

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



        #df_select = df_select[df_select["artists"].str.contains('|'.join(artsep), regex = True)]
        # this is not working. Shaan, Shreya Ghoshal will not show results with Shreya Ghoshal, Shaan even when pressing all

        for i in artsep:
            df_select = df_select[df_select["artists"].str.contains(fr"\b{i}\b")]


        sort_and_display(sort_button)

st.subheader("Description of all the features of whole Dataset", divider=True)
st.dataframe(df_select.describe(include="all"))
st.dataframe(df.describe(include="all"))


selected_artists = artists_select
avg_popularity = df_select["popularity"].mean()
num_of_songs = df_select.shape[0]

with left:
    if len(artists_select) == 0:
        st.subheader(f"Artists Selected:")
        st.subheader(f"ALL")
    else:
        st.subheader(f"Artists Selected:")
        for i in selected_artists:
            st.subheader(f"{i}")

        #st.write(*selected_artists, sep= "\n")  # asterisk to used to unpack the list in a single line

with middle:
    st.subheader(f"Average Popularity:")
    st.subheader(f"{avg_popularity}")

with right:
    st.subheader(f"Number of Songs:")
    st.subheader(f"{num_of_songs}")

if len(artists_select) == 0:
    st.subheader(f"Songs with mre than 75 popularity rating", divider=True)
else:
    st.subheader(f"{artstr} songs with more than 75 popularity rating", divider=True)

popularsongsdf = df_select[df_select["popularity"] > 75].sort_values(by = "popularity",ascending=False)
popularsongsdf = popularsongsdf[["year", "name", "artists", "popularity"]]
st.dataframe(popularsongsdf)


if len(artists_select) == 0:
    st.subheader(f"Correlation between features for all artists", divider=True)
else:
    st.subheader(f"Correlation between features for {artstr}", divider=True)

corr_df = df_select.drop(["name", "artists", "key", "mode", "explicit"], axis = 1).corr(method = "pearson")
corr = plt.figure(figsize = (14,6))
heatmap = sns.heatmap(corr_df,annot = True, fmt = ".1g", vmin = -1, vmax = 1, center = 0, cmap = "inferno", linewidths=1, linecolor = "Black")
heatmap.set_title("Correlation HeatMap between Variables")
heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation = 90)

st.pyplot(corr)


#Creating Line Graphs for understanding the evolution of musical elements with time

if len(artists_select) == 0:
    st.subheader(f"Evolution of musical elements over time", divider=True)
else:
    st.subheader(f"Evolution of musical elements over time for {artstr}", divider=True)

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

if len(artists_select) == 0:
    st.subheader(f"Growth pf explicit and non-explicit songs", divider=True)
else:
    st.subheader(f"Growth of explicit and non-explicit songs for {artstr}", divider=True)


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

if len(artists_select) == 0:
    st.subheader(f"Popularity difference between explicit and non-explicit songs", divider=True)
else:
    st.subheader(f"Popularity difference between explicit and non-explicit songs for {artstr}", divider=True)

expo = px.box(df_select, x='explicit', y='popularity',
                 title="0 - Non-explicit    &    1 - Explicit")
expo.update_layout(xaxis_title="Explicit", yaxis_title="Popularity",
                      template="plotly_white")
st.plotly_chart(expo)



if len(artists_select)>=1:

    st.subheader(f"Trend of various features of songs over time for {artstr}", divider=True)

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

if len(artists_select) == 0:
    st.subheader(f"Number of songs in each key", divider=True)
else:
    st.subheader(f"Number of songs in each key for {artstr}", divider=True)

key_df = df_select.groupby('key').agg({'year': 'count'}).reset_index()
#st.dataframe(key_df)
fig_key = px.bar(key_df, x='key', y='year',
             labels={'key': 'Key', 'year': 'Songs'},
             color='year')

fig_key.update_layout(template='plotly_white', xaxis_title='Key', yaxis_title='Number of Songs')
st.plotly_chart(fig_key)

if len(artists_select) == 0:
    st.subheader(f"Popularity of songs in each key", divider=True)
else:
    st.subheader(f"Popularity of songs in each key for {artstr}", divider=True)

key_pop = px.box(df_select, x="key", y='popularity')
key_pop.update_layout(xaxis_title="Key", yaxis_title="Popularity",
                      template="plotly_white")
st.plotly_chart(key_pop)