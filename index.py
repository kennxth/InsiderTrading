import streamlit as st
from components.helper import vis1, vis2, vis3, vis4, vis5, vis6

#packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("🌚 Insider Trading")

#Intro
st.header('Introduction')
st.write('The goal of this project is to investigate the performance of stock market portfolios held by politicians, particularly those serving in Congress and the Senate, in the context of major legislative bills being passed. By examining the timing of trades executed by these politicians, the study aims to uncover any potential correlations between legislative activities and stock market transactions made by these individuals. This analysis seeks to provide insights into how political knowledge and legislative actions might influence or correlate with the personal financial decisions of lawmakers, potentially shedding light on the integrity and transparency of these transactions.')
st.header('.')
st.header('.')
st.header('.')
st.header('.')
st.image('IMG_6568.jpeg')
st.header('Data Visualizations')

vis3()
vis4()
vis1()
vis2()
vis5()
st.image('IMG_2033.jpeg')
st.image('IMG_2501.jpeg')
vis6()

st.header('Conclusions')
st.markdown("""
- Congress beat SPY, on aggregate, based on party and many members individually.
- Congress has continued to trade despite conflicts and potential bans proposed
- Congress traded thousands of times in a year plagued by high interest rates, war, conflicts, and did well doing so.
""")

st.write('Representative Ro Khanna Khanna agenda includes fighting against market corruption:')
st.markdown("""
            - bans stock trading for Congress + spouses
- bans Congress from lobbying after
- 12 year limits for Congress
- ban lobbyists + PACs donations
            """)

