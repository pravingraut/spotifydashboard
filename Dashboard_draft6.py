import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import statsmodels.api as sm

# ---------------------- PAGE CONFIG --------------------------------

st.set_page_config(page_title="Spotify Dashboard",
page_icon="🎶",
layout="wide")


# --------------------- READING DATA ---------------------------------

@st.cache_data
def get_data():
    df = pd.read_csv("Fixed_Dataset.csv", index_col="release_date")
    key_mapping = {
        0: 'C', 1: 'C#', 2: 'D', 3: 'D#', 4: 'E', 5: 'F',
        6: 'F#', 7: 'G', 8: 'G#', 9: 'A', 10: 'A#', 11: 'B'
    }
    df['key'] = df['key'].map(key_mapping)

    explicit_mapping = {
        0: 'Non-explicit', 1: 'Explicit'
    }
    df['explicit'] = df['explicit'].map(explicit_mapping)

    mode_mapping = {
        0: 'Minor scale', 1: 'Major scale'
    }
    df['mode'] = df['mode'].map(mode_mapping)

    df['artists'] = df['artists'].str.strip()

    return df

df = get_data()

# It would have been good if dataset had country of release column.
# For example, we have an artist named KK. But there is another artist with the same name in other country.
# So filtering becomes difficult.

# ---------------------------- SIDEBAR -----------------------------

st.sidebar.image(image="logo.jpg")

st.sidebar.header("Please Filter Here:")

year_range_select = st.sidebar.slider(
    "Select the Years:", min_value = 1922, max_value = 2021, value = [1922, 2021]
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
    options = df["explicit"].unique(),
    default= df["explicit"].unique()
)

mode_select = st.sidebar.multiselect(
    "Scale of Songs:",
    options = df["mode"].unique(),
    default= df["mode"].unique()
)

# ------------------------------- MAIN DASHBOARD -----------------------------------

title, logo = st.columns([0.9, 0.1])
with title:
    st.title("Spotify Data Analysis")
with logo:
    st.image("Black_Logo.png", width = 100)
st.markdown("####")

#-------------------------------------------------------------------------------------#

# Three boxes that will show: Artists selected, Average Popularity of artists, Number of Songs.

left, middle, right = st.columns(3, gap="medium")
st.markdown("#####")

#-------------------------------------------------------------------------------------#


if len(artists_select) == 0:
    st.subheader(f"Songs of All artists", divider=True)
else:
    st.subheader(f"Songs of {artstr}", divider=True)

def sort_and_display (x):
    if x == "Date":
        st.dataframe(df_select.drop(columns="year").sort_index())
    else:
        st.dataframe(df_select.sort_values(by="popularity", ascending=False))

if len(artists_select) == 0:
    df_select = df.query("year >= @year_range_select[0] & year <= @year_range_select[1]"
                             "& key == @key_select"
                             "& popularity >= @popularity_select[0] & popularity <= @popularity_select[1]"
                             "& explicit == @explicit_select"
                             "& mode == @mode_select"
                        )

    sort_and_display(sort_button)
else:
    if only_all_button == "Only":
        df_select = df.query("year >= @year_range_select[0] & year <= @year_range_select[1]"
                         "& key == @key_select "
                         "& artists == @artists_select"
                         "& popularity >= @popularity_select[0] & popularity <= @popularity_select[1]"
                         "& explicit == @explicit_select"
                         "& mode == @mode_select"
                        )
        sort_and_display(sort_button)

    else:
        df_select = df.query("year >= @year_range_select[0] & year <= @year_range_select[1]"
                             "& key == @key_select "
                             #"& artists == @artists_select"
                             "& popularity >= @popularity_select[0] & popularity <= @popularity_select[1]"
                             "& explicit == @explicit_select"
                             "& mode == @mode_select"
                             )

        #df_select = df_select[df_select["artists"].str.contains('|'.join(artsep), regex = True)]
        # this is not working. Shaan, Shreya Ghoshal will not show results with Shreya Ghoshal, Shaan even when pressing all

        for i in artsep:
            df_select = df_select[df_select["artists"].str.contains(fr"^{i}$|^{i},|,\s{i},|',\s{i}$")]

        sort_and_display(sort_button)


#-------------------------------------------------------------------------------------#

selected_artists = artists_select
avg_popularity = int(df_select["popularity"].mean())
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
    st.subheader(f"{avg_popularity}/100")

with right:
    st.subheader(f"Number of Songs:")
    st.subheader(f"{num_of_songs}")


#-------------------------------------------------------------------------------------#

if len(artists_select) == 0:
    st.subheader(f"Top 10 Songs of all time", divider=True)
else:
    st.subheader(f"Top 10 Songs of {artstr}", divider=True)

popularsongsdf = df_select.sort_values(by = "popularity",ascending=False).head(10)
popularsongsdf = popularsongsdf[["name", "artists", "popularity"]]
st.dataframe(popularsongsdf)


#-------------------------------------------------------------------------------------#

if len(artists_select) == 0:
    st.subheader(f"Correlation between features for all artists", divider=True)
else:
    st.subheader(f"Correlation between features for {artstr}", divider=True)

corr_df = df_select.drop(["name", "artists", "key", "mode", "explicit"], axis = 1).corr(method = "pearson")

#corr = plt.figure(figsize = (14,6))
#heatmap = sns.heatmap(corr_df,annot = True, fmt = ".1g", vmin = -1, vmax = 1, center = 0, cmap = "inferno", linewidths=1, linecolor = "Black")
#heatmap.set_title("Correlation HeatMap between Variables")
#heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation = 90)

corr_fig = px.imshow(corr_df, text_auto = True, aspect = "auto", zmin = -1, zmax = 1, color_continuous_scale='RdBu')

st.plotly_chart(corr_fig)

#-------------------------------------------------------------------------------------#

#Creating Line Graphs for understanding the evolution of musical elements with time

if len(artists_select) == 0:
    st.subheader(f"Evolution of musical elements over time", divider=True)
else:
    st.subheader(f"Evolution of musical elements over time for {artstr}", divider=True)

#sp1, elements = plt.subplots(3, 2) # Initiating subplots

evol_fig = make_subplots(rows=3, cols=2,
                         subplot_titles=("Danceability", "Valence", "Energy", "Speechiness", "Acousticness", "Instrumentalness"))

arrElements = ["danceability", "valence", "energy", "speechiness", "acousticness",  "instrumentalness"]
row = 1
col = 1

for element in arrElements: # Creating a line graph for each musical element
    #df_select.groupby("year")[element].mean().plot(ax=elements[row][col], figsize=(30,20), color="#bd062a", linewidth=2.5) # Group rows by year and get the mean values for that particular element
    #elements[row][col].set_xlabel("decade", fontsize = 30)
    #elements[row][col].set_ylabel(element, fontsize = 30)
    #elements[row][col].tick_params(labelsize=30)
    #elements[row][col].tick_params(labelsize=30)

    element_df = df_select.groupby("year")[element].mean()   #whenever we groupby a column, that column becomes the index
    element_df = element_df.reset_index()

    # Making new column called decade and then grouping the dataframe by decade column
    element_df['decade'] = (element_df.year // 10) * 10     # Converts 1943 to 1940 and so on
    decadal_mean = element_df.groupby('decade').mean()

    evol_fig.add_trace(go.Scatter( x = decadal_mean.index, y = decadal_mean[element]), row= row, col= col)
    if col == 1:
        col = 2
    else:
        col = 1
        row += 1

evol_fig.update_layout (height=700, width=800,
                        title_text="Decade-wise change in musical characteristics", showlegend = False)

st.plotly_chart(evol_fig)

#-------------------------------------------------------------------------------------#

#Number of songs released by year

if len(artists_select) == 0:
    st.subheader(f"Year wise release of songs", divider=True)
else:
    st.subheader(f"Year wise release of songs for {artstr}", divider=True)


explicits = df_select.groupby("explicit")["year"].value_counts().unstack(fill_value=0)

# Convert to DataFrame
df_new = explicits.reset_index()
df_new.columns.name = None  # Remove the columns name

# Reshape DataFrame for plotting
df_new = pd.melt(df_new, id_vars=['explicit'], var_name='year', value_name='count')

# Create the scatter plot
figexpli = px.scatter(df_new, x='year', y='count', color='explicit', color_discrete_map={"Non-explicit": "SkyBlue", "Explicit": "BurlyWood"})

st.plotly_chart(figexpli)


#-------------------------------------------------------------------------------------#

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


#-------------------------------------------------------------------------------------#

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


#-------------------------------------------------------------------------------------#

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


#-------------------------------------------------------------------------------------#

if len(artists_select) == 0:
    st.subheader(f"Popularity of songs in each key", divider=True)
else:
    st.subheader(f"Popularity of songs in each key for {artstr}", divider=True)

key_pop = px.box(df_select, x="key", y='popularity')
key_pop.update_layout(xaxis_title="Key", yaxis_title="Popularity",
                      template="plotly_white")
st.plotly_chart(key_pop)

#-------------------------------------------------------------------------------------#

if len(artists_select) == 0:
    st.subheader(f"Popularity of major and minor scale", divider=True)
else:
    st.subheader(f"Popularity of major and minor scale for {artstr}", divider=True)
modepop = px.box(df_select, x="mode", y="popularity")
st.plotly_chart(modepop)

st.subheader("Description of all the features", divider=True)
t1, t2 = st.tabs(["Selected Artists", "Whole Dataset"])
with t1:
    st.dataframe(df_select.describe(include="all"))
with t2:
    st.dataframe(df.describe(include="all"))


#-------------------------------------------------------------------------------------#

st.divider()


col1, col2, col3 = st.columns(3)
with col2:
    st.image("IIMS Logo Light.png", width= 250, caption="Dashboard made for 'Data Analysis using Python' course offered by Prof. Pradeep Dadabada")