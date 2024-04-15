import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

senateWatcherData = pd.read_csv('./SenatorCleaned.csv')
InsiderTrading = pd.read_csv('./all_transactions.csv')
stock_option_rows = InsiderTrading[InsiderTrading['asset_type'] == 'Stock Option']

# Find discrete values and their distribution
categorical_columns = ['owner', 'ticker', 'asset_type', 'type', 'party', 'state', 'industry', 'sector']
discrete_values = {column: InsiderTrading[column].unique() for column in categorical_columns}
distribution = {column: InsiderTrading[column].value_counts() for column in categorical_columns}

# Amount column distribution with ranges
amount_distribution = InsiderTrading['amount'].value_counts()

# Showing distribution for categorical columns
discrete_values = {col: discrete_values[col] for col in ['owner', 'asset_type', 'party', 'sector', 'ticker', 'type', 'state']}
distribution_ex = {col: distribution[col] for col in ['owner', 'asset_type', 'party', 'sector', 'ticker', 'type', 'state']}

# Distribution for Senators
senator_distribution = InsiderTrading['senator'].value_counts()

distribution, senator_distribution.head(10), amount_distribution.head()

# Identify the Top 10 Senators by transaction volume
top_10_senators = InsiderTrading['senator'].value_counts().head(10).index

# Create a dictionary to hold each senator's DataFrame
senators_dfs = {}

for senator in top_10_senators:
    # Filter the original DataFrame for each senator, sorted by largest transaction and assign it to the dictionary
    senators_dfs[senator] = InsiderTrading[InsiderTrading['senator'] == senator].sort_values(by=['amount'], ascending=True)


# Senators_dfs[senator_name] will give you the DataFrame for that senator
david_perdue_df = senators_dfs['David Perdue']
thomas_carper_df = senators_dfs['Thomas R. Carper']
tommy_tuberville_df = senators_dfs['Tommy Tuberville']
sheldon_whitehouse_df = senators_dfs['Sheldon Whitehouse']
pat_roberts_df = senators_dfs['Pat Roberts']
susan_collins_df = senators_dfs['Susan M. Collins']
shelley_capito_df = senators_dfs['Shelley Moore Capito']
kelly_loeffler_df = senators_dfs['Kelly Loeffler']
jack_reed_df = senators_dfs['Jack Reed']
ron_wyden_df = senators_dfs['Ron Wyden']

InsiderTrading.replace('Unknown', np.nan, inplace=True)

# Drop any rows that now contain NaN values, removing rows with originally 'Unknown' values
InsiderTrading.dropna(inplace=True)

# Identifying duplicates
duplicates = InsiderTrading.duplicated()

# Counting the number of duplicate rows
num_duplicates = duplicates.sum()
print(f"Number of duplicate rows: {num_duplicates}")

InsiderTrading_cleaned = InsiderTrading.drop_duplicates(keep ='last')

import plotly.graph_objects as go
import pandas as pd

# Read the data
df = pd.read_csv('Politicians_and_%_Gain.csv')
df_spy = pd.read_csv('congress_vs_spy_full_2023.csv')

# Define color mapping for party affiliations
color_mapping = {'Republican': 'red', 'Democrat': 'blue'}

# Map party affiliations to colors
df['marker_color'] = df['Party'].map(color_mapping)

# Sort the DataFrame by ytd_returns
df = df.sort_values(by='Portfolio Gain %', ascending=False)

# Create a figure
fig1 = go.Figure()

# Add trace for Republicans
fig1.add_trace(go.Bar(
    x=df[df['Party'] == 'Republican']["Name"],
    y=df[df['Party'] == 'Republican']["Portfolio Gain %"],
    marker_color='red',
    hoverinfo="text+y",
    name="Republicans",
    legendgroup="Republicans",
))

# Add trace for Democrats
fig1.add_trace(go.Bar(
    x=df[df['Party'] == 'Democrat']["Name"],
    y=df[df['Party'] == 'Democrat']["Portfolio Gain %"],
    marker_color='blue',
    hoverinfo="text+y",
    name="Democrats",
    legendgroup="Democrats",
))

fig1.add_trace(go.Bar(
    x=df_spy[df_spy['party'] == 'SPY']["name"],
    y=df_spy[df_spy['party'] == 'SPY']["ytd_returns"],
    marker_color='yellow',
    hoverinfo="text+y",
    name="SPY",
    legendgroup="SPY",
))

fig1.update_layout(
    title="Year-to-Date Returns by Politician",
    xaxis_title="Politician",
    yaxis_title="Year-to-Date Returns",
    xaxis={'categoryorder':'total descending'},
    showlegend=True
)

# Show the plot
def vis1():
    st.plotly_chart(fig1, use_container_width=True)


# Read the data
df = pd.read_csv('congress_vs_spy_full_2023.csv')

# Drop rows with missing values
df = df.dropna()

# Define color mapping for party affiliations
color_mapping = {'R': 'rgba(255, 0, 0, 0.5)', 'D': 'rgba(0, 0, 255, 0.5)', 'SPY': 'yellow'}

# Create a figure
fig = go.Figure()

# Add traces for each politician
for politician in df['name'].unique():
    df_politician = df[df['name'] == politician]
    party = df_politician.iloc[0]['party']
    color = color_mapping.get(party, 'black')  # Default to black for unknown parties

    # Calculate percentage change relative to the initial value for each year
    initial_value = df_politician.iloc[0, 3]  # Initial value in 2018
    percentage_change = (df_politician.iloc[0, 3:] / initial_value - 1) * 100

    fig.add_trace(go.Scatter(x=df_politician.columns[3:], y=percentage_change,
                             mode='lines', name=politician, line=dict(color=color),
                             hoverinfo='text', hovertext=[f"{politician}<br>{year}: {growth:.2f}%"
                                                           for year, growth in zip(df_politician.columns[3:], percentage_change)],
                             legendgroup=party if party != 'SPY' else None))

# Update layout
fig.update_layout(
    title="Normalized Portfolio Performance Over Time (2018-2022)",
    xaxis_title="Year",
    yaxis_title="Percentage Growth",
    hovermode="closest",
    hoverlabel=dict(bgcolor="white", font_size=12),
    legend_title="Party",
)

def vis2():
    st.plotly_chart(fig, use_container_width=True)

import pandas as pd
import plotly.express as px

# Load the data containg information of politician's name, party, gain%, committees, and if they're chairmen
data = pd.read_csv('Politicians_and_%_Gain.csv')

# Map full state names to abbreviations
state_abbreviations = {
    'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA',
    'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE', 'Florida': 'FL', 'Georgia': 'GA',
    'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA',
    'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD',
    'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS',
    'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV', 'New Hampshire': 'NH',
    'New Jersey': 'NJ', 'New Mexico': 'NM', 'New York': 'NY', 'North Carolina': 'NC',
    'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK', 'Oregon': 'OR', 'Pennsylvania': 'PA',
    'Rhode Island': 'RI', 'South Carolina': 'SC', 'South Dakota': 'SD', 'Tennessee': 'TN',
    'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT', 'Virginia': 'VA', 'Washington': 'WA',
    'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY'
}
data['State'] = data['State'].map(state_abbreviations).fillna(data['State'])

# Create a new column for hover information
data['hover_info'] = data.apply(lambda x: f"{x['Name']}<br>Party: {x['Party']}<br>Gain: {x['Portfolio Gain %']}<br>Committee: {x['Committee(s)']}<br>Chairman: {x['Chairman?']}", axis=1)

# Group by State and create a summary
grouped_data = data.groupby('State')['hover_info'].apply(lambda x: '<br><br>'.join(x)).reset_index()

# Add a column indicating data presence
grouped_data['has_data'] = 'Data Available'

# Merge this grouped data with a list of all states to identify states without data
all_states = pd.DataFrame({'State': list(state_abbreviations.values())})
grouped_data = all_states.merge(grouped_data, on='State', how='left')

# Fill missing values for states without data
grouped_data['hover_info'] = grouped_data['hover_info'].fillna('No data available')
grouped_data['has_data'] = grouped_data['has_data'].fillna('No Data')

# Create the map with distinct colors for states with and without data
fig2 = px.choropleth(grouped_data,
                    locations='State',  
                    locationmode='USA-states',  
                    color='has_data',  
                    hover_name='State',  
                    hover_data=['hover_info'],  
                    scope="usa",  
                    color_discrete_map={'Data Available': 'green', 'No Data': '#dddddd'} 
                    )

fig2.update_geos(fitbounds="locations", visible=False)  
fig2.update_layout(title_text='US Politicians by State', geo_scope='usa')
fig2.show()

def vis3():
    st.plotly_chart(fig2, use_container_width=True)
