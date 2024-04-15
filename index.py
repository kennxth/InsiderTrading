import streamlit as st
from components.helper import vis1, vis2, vis3

#packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("🌚 Insider Trading")

#Intro
st.header('Introduction')
st.write('The goal of this project is to investigate the performance of stock market portfolios held by politicians, particularly those serving in Congress and the Senate, in the context of major legislative bills being passed. By examining the timing of trades executed by these politicians, the study aims to uncover any potential correlations between legislative activities and stock market transactions made by these individuals. This analysis seeks to provide insights into how political knowledge and legislative actions might influence or correlate with the personal financial decisions of lawmakers, potentially shedding light on the integrity and transparency of these transactions. ')

vis1()
vis2()
vis3()

