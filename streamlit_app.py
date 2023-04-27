import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

# Function to load time series data
def load_data():
    data = pd.read_csv('shampoo_sales.csv')
    return data

# Function to plot time series data
def plot_time_series(data):
    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(data)
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.set_title('Time Series Data')
    st.pyplot(fig)

# Function to plot ACF and PACF plots
def plot_acf_pacf(data, lag):
    fig, ax = plt.subplots(nrows=2, figsize=(12,6))
    plot_acf(data, lags=lag, ax=ax[0])
    plot_pacf(data, lags=lag, ax=ax[1])
    ax[0].set_title('Autocorrelation Function (ACF)')
    ax[1].set_title('Partial Autocorrelation Function (PACF)')
    st.pyplot(fig)

# Function to train ARIMA model and make predictions
def train_arima(data, p, d, q, steps = 1):
    model = ARIMA(data, order=(p, d, q))
    model_fit = model.fit()
    # print(model_fit.forecast(steps=1))
    # forecast = model_fit.forecast(steps=1)[0]
    forecast = model_fit.forecast(steps)
    return forecast

# Set up the app
st.title('Time Series Analysis with ARIMA Model')
data = load_data()

# Add dropdown to select time series dataset
# data_select = st.sidebar.selectbox('Select Time Series Dataset', ['Dataset 1', 'Dataset 2', 'Dataset 3'])
data_select = st.sidebar.selectbox('Select Time Series Dataset', ['Dataset 1', 'Dataset 2'])

if data_select == 'Dataset 1':
    data = pd.read_csv('daily-minimum-temperatures.csv', index_col=0, parse_dates=True)
# elif data_select == 'Dataset 2':
#     data = pd.read_csv('dataset2.csv', index_col=0, parse_dates=True)
else:
    data = pd.read_csv('shampoo_sales.csv', index_col=0, parse_dates=True)

# Add slider for time lag
lag = st.sidebar.slider('Select Time Lag', min_value=1, max_value=50)

# Plot time series data
plot_time_series(data)

# Plot ACF and PACF plots
plot_acf_pacf(data, lag)

# Add sliders for ARIMA model parameters
p = st.sidebar.slider('Select p value (lag order)', min_value=0, max_value=10)
d = st.sidebar.slider('Select d value (degree of differencing)', min_value=0, max_value=10)
q = st.sidebar.slider('Select q value (moving average wimdow sie)', min_value=0, max_value=10)
steps = st.sidebar.slider('Select number of steps to forecast', min_value=1, max_value=5)

# Train ARIMA model and make predictions
forecast = train_arima(data, p, d, q, steps)

# Display forecasted value
st.write('The forecasted value for the next time step is:', forecast)