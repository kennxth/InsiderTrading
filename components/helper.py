import streamlit as st
import numpy as np
import pandas as pd

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

def vis3():
    st.plotly_chart(fig2, use_container_width=True)

file_path = 'Politicians_and_%_Gain_FINAL - Sheet1.csv'

politicians_data = pd.read_csv(file_path)

politicians_data['Portfolio Gain %'] = (
    politicians_data['Portfolio Gain %'].str.replace('%', '').astype(float)
)

committee_data = politicians_data['Committee(s)'].str.split(',', expand=True).stack().reset_index(level=1, drop=True)
committee_data.name = 'Committee'  

df_expanded = politicians_data.drop('Committee(s)', axis=1).join(committee_data)

grouped = df_expanded.groupby(['Committee', 'Chairman?'])['Portfolio Gain %'].mean().reset_index()

fig4 = px.density_heatmap(
    grouped, 
    x='Chairman?', 
    y='Committee', 
    z='Portfolio Gain %', 
    category_orders={"Chairman?": ["Y", "N"]},  
    color_continuous_scale='Viridis',  
    title='Heatmap of Average Portfolio Gain % by Committee and Chair Status'
)

fig4.update_layout(
    xaxis_title="Chairman Status",
    yaxis_title="Committee",
    yaxis={'categoryorder':'total ascending'},  
    coloraxis_colorbar=dict(title="Avg Portfolio Gain %"),
    height=700
)

def vis4():
    st.plotly_chart(fig4, use_container_width=True)
    
    import plotly.graph_objects as go

politicians_data['Date Reported'] = pd.to_datetime(politicians_data['Date Reported'], errors='coerce')

events = {
    'Chips Bill Passed': '2023-08-09',
    'Activision-Blizzard Acquisition by MSFT Completed': '2023-10-13',
    'Additional Chips Investment by Government': '2024-03-20',
    'Twitter Acquisition Start': '2022-04-14',
    'Escalation of Worldwide Conflict': '2023-10',
    'Bitcoin Price Surge': '2024-01-01',
    'Initial Semiconductor Stock Boom (NVDA Earnings)': '2023-05-23',
    'Continued Rapid Revenue Growth (NVDA Earnings)': '2023-11-21',
    'Legalization of Sports Betting (Continuing)': '2023-01-01',
    'Surge in Oil Prices': '2023-12-01',
    'Surge in SPY Price:': '2024-01-01',
    'Biden Administration Backs Effort to Standardize Tesla EV Charging Stations': '2023-12-19'
}

fig5 = go.Figure()

fig5.add_trace(go.Scatter(
    x=politicians_data['Date Reported'],
    y=politicians_data['Name'],
    mode='markers',
    marker=dict(size=10, color=politicians_data['Action'].map({'Buy': 'green', 'Sell': 'red'})),
    text=politicians_data['Action'] + " " + politicians_data['Ticker'] + ": " + politicians_data['Amount'],
    hoverinfo='text',
    name='Trades'
))

shapes = []
for i, (event, date) in enumerate(events.items()):
    shapes.append(dict(
        type='line',
        xref='x',
        yref='paper',
        x0=date,
        y0=0,
        x1=date,
        y1=1,
        line=dict(color='blue', width=2),
        opacity=0.5,
        visible=True  
    ))

buttons = []
for i, (event, date) in enumerate(events.items()):
    visible = [False] * len(events)
    visible[i] = True  
    event_date_str = pd.to_datetime(date).strftime('%B %d, %Y') 

    buttons.append(dict(
        label=f"{event} ({event_date_str})",  
        method='update',
        args=[{'visible': [True] + visible},  
              {'title': f"Showing event: {event} on {event_date_str}",
               'shapes': [dict(shapes[i], visible=True if visible[i] else False) for i in range(len(shapes))]}]
    ))

fig5.update_layout(
    updatemenus=[{
        'type': 'dropdown',
        'x': 1.15,
        'y': 1,
        'showactive': True,
        'active': 0,
        'buttons': buttons
    }],
    title="Timeline of Politicians' Recent Large Stock Trades and Key Events",
    xaxis_title="Date",
    yaxis_title="Politician",
    shapes=shapes,
    showlegend=False,
    height=1000
)

def vis5():
    st.plotly_chart(fig5, use_container_width=True)
    
    politicians_data['Portfolio Gain %'] = politicians_data['Portfolio Gain %'].replace('%','',regex=True).astype(float)

treemap_data = politicians_data[['State', 'Party', 'Committee(s)', 'Name', 'Largest Holding', 'Stock Sector', 'Portfolio Gain %']].dropna()
treemap_data['Committee(s)'] = treemap_data['Committee(s)'].str.strip()  


fig6 = px.treemap(treemap_data, 
                 path=['Committee(s)','Stock Sector', 'Largest Holding', 'Name'], 
                 values='Portfolio Gain %',
                 color='Portfolio Gain %',
                 color_continuous_scale='Greens',
                 title=' ')

fig6.update_layout(
    width=1000,  
    height=1000,  
    margin=dict(l=10, r=10, t=50, b=20) 
)

def vis6():
    st.plotly_chart(fig6, use_container_width=True)