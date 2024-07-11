

import streamlit as st

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from datetime import datetime, date

import openpyxl
import warnings

import yfinance as yf

from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm

from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
import statsmodels.tsa.api as smt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

import itertools


warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)

# streamlit run .\stock_pred\stock_prediction_app_str_v2.py



st.title(" 📈 Stock Prediction App")


START = "2000-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

stocks =("AKBNK.IS", "ARCLK.IS", "AYDEM.IS", "EKGYO.IS", "ENJSA.IS","EREGL.IS", "FROTO.IS","GARAN.IS", "GSDHO.IS",
         "HALKB.IS","HEKTS.IS", "ISCTR.IS","ISDMR.IS",  "KRDMD.IS", "KONKA.IS", "KONYA.IS",  "OYAKC.IS","PETKM.IS",
         "PEKGY.IS", "PSDTC.IS", "PGSUS.IS", "SASA.IS", "SISE.IS", "TCELL.IS",  "TUPRS.IS", "TOASO.IS")

# Sidebar setup
st.sidebar.title('Select stock dataset for prediction (BIST 100)')
selected_stocks = st.sidebar.selectbox("", stocks)


# Sidebar navigation
st.sidebar.title('Menu')
options = st.sidebar.radio('', ['Stock data review', 'Dickey-Fuller test',
                                'Decompose the time series into components',
                                'Plotting the ACF and PACF',
                                'Model training, base model building and validation',
                                'Hyper parameter optimization, model building and validation'])

def home():

    if selected_stocks:
        @st.cache_data
        def load_data(ticker):
            data_r = yf.download(ticker, START, TODAY)
            data_r.reset_index(inplace=True)
            return data_r

        data_load_state = st.text("Load data...")
        data_r = load_data(selected_stocks)
        data_r = data_r.fillna(method='bfill')
        st.write("")
        data_load_state.text("Loading data...done!!!!!")

    else:
        st.header('Select please a stock data')

    data = data_r.copy()



    st.write(f"<div style='font-size:25px; color:black; font-weight:bold;'> {selected_stocks} </div>",
             unsafe_allow_html=True)
    st.write("")  # to leave a space in the web page presentation
    st.write("Raw Data : Last 10 days")
    st.write("")  # to leave a space in the web page presentation

    data_1 = data.copy()
    date_col = ["Date"]
    # Loop through the columns and apply date formatting
    for col in date_col:
        data_1[col] = pd.to_datetime(data_1[col]).dt.strftime('%Y-%m-%d')
    st.dataframe(data_1.tail(10))

    # Plot the stock closing prices
    st.write("<div style='font-size:25px; color:black; font-weight:bold;'> 📉 Plot the stock closing prices</div>",
             unsafe_allow_html=True)

    st.write("")  # to leave a space in the web page presentation

    inp_plot_date1 = pd.to_datetime(data["Date"]).min()
    inp_plot_date2 = pd.to_datetime(data["Date"]).max()

    # Writing two dates side by side using HTML and CSS
    st.write(
        f"""
        <div style='display: flex; justify-content: space-between;'>
            <div style='font-size:18px;color:black; font-weight:bold;'> 📆 Start Date : {inp_plot_date1.strftime('%Y-%m-%d')}</div>
            <div style='font-size:18px;color:black; font-weight:bold;'> 📆 Last Date : {inp_plot_date2.strftime('%Y-%m-%d')}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")       # to leave a space in the web page presentation

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data.Date, data['Close'], label='Stock Price Close')
    ax.set_title('Time Series Data')
    ax.set_xlabel('Date')
    ax.set_ylabel('Stock Price Close')
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    st.write("")
    st.write("")

    st.write("<div style='font-size:25px; color:black; font-weight:bold;'> 📉 Stock Data Review  </div>",
             unsafe_allow_html=True)
    st.write("")
    st.write("<div style='font-size:15px; color:red; font-weight:bold;'> 📆 * Select Dates  </div>",
             unsafe_allow_html=True)

    inp_plot_date1 = pd.to_datetime(data["Date"]).min()
    inp_plot_date2 = pd.to_datetime(data["Date"]).max()

    plot_date1 = pd.to_datetime(st.date_input(" Start Date ", inp_plot_date1))
    plot_date2 = pd.to_datetime(st.date_input(" End Date ", inp_plot_date2))

    train_plot = data.loc[(data["Date"] >= plot_date1) & (data["Date"] <= plot_date2)]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(train_plot.Date, train_plot['Close'], label='Stock Close')
    ax.set_title('Time Series Data')
    ax.set_xlabel('Date')
    ax.set_ylabel('Stock Price Close')
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    st.write("")                  # to leave a space in the web page presentation

    data = data[['Date', 'Close']]

    data1 = data.copy()

def dickey_fuller():

    if selected_stocks:
        @st.cache_data
        def load_data(ticker):
            data_r = yf.download(ticker, START, TODAY)
            data_r.reset_index(inplace=True)
            return data_r

        data_load_state = st.text("Load data...")
        data_r = load_data(selected_stocks)
        data_r = data_r.fillna(method='bfill')
        st.write("")
        data_load_state.text("Loading data...done!!!!!")

    else:
        st.header('Select please a stock data')

    data1 = data_r.copy()

    st.write(f"<div style='font-size:25px; color:black; font-weight:bold;'> {selected_stocks} </div>",
             unsafe_allow_html=True)

    st.write("")              # to leave a space in the web page presentation


    # Dickey-Fuller test to check for stationarity

    st.write(
        f"<div style='font-size:20px; color:black; font-weight:bold;'> Dickey-Fuller test to check for stationarity  </div>",
        unsafe_allow_html=True)

    st.write("")  # to leave a space in the web page presentation
    st.write("")  # to leave a space in the web page presentation

    def is_stationary(y):
        result = sm.tsa.adfuller(y)
        p_value = result[1]
        if p_value < 0.05:
            st.write(
                f"<div style='font-size:18px; color:navy; font-weight:bold;'>"
                f"✅ Result : Stationary (H0: Non-Stationary, p-value: {round(p_value, 3)})"
                f"</div>",
                unsafe_allow_html=True
            )
            # st.write(f"Result: Stationary (H0: Non-Stationary, p-value: {round(p_value, 3)})")

        else:
            st.write(
                f"<div style='font-size:18px; color:navy; font-weight:bold;'>"
                f"✅ Result : Non-Stationary (H0: Non-Stationary, p-value: {round(p_value, 3)})"
                f"</div>",
                unsafe_allow_html=True
            )
            # st.write(f"Result: Non-Stationary (H0: Non-Stationary, p-value: {round(p_value, 3)})")

    is_stationary(data1['Close'])

def decompose():

    if selected_stocks:
        @st.cache_data
        def load_data(ticker):
            data_r = yf.download(ticker, START, TODAY)
            data_r.reset_index(inplace=True)
            return data_r

        data_load_state = st.text("Load data...")
        data_r = load_data(selected_stocks)
        data_r = data_r.fillna(method='bfill')
        st.write("")
        data_load_state.text("Loading data...done!!!!!")

    else:
        st.header('Select please a stock data')

    data2 = data_r.copy()

    st.write(f"<div style='font-size:25px; color:black; font-weight:bold;'> {selected_stocks} </div>",
             unsafe_allow_html=True)

    st.write("")              # to leave a space in the web page presentation
    st.write("")              # to leave a space in the web page presentation


    def is_stationary(y):
        result = sm.tsa.adfuller(y)
        p_value = result[1]
        if p_value < 0.05:
            st.write(
                f"<div style='font-size:18px; color:navy; font-weight:bold;'>"
                f"Result : Stationary (H0: Non-Stationary, p-value: {round(p_value, 3)})"
                f"</div>",
                unsafe_allow_html=True
            )
            # st.write(f"Result: Stationary (H0: Non-Stationary, p-value: {round(p_value, 3)})")

        else:
            st.write(
                f"<div style='font-size:18px; color:navy; font-weight:bold;'>"
                f"Result : Non-Stationary (H0: Non-Stationary, p-value: {round(p_value, 3)})"
                f"</div>",
                unsafe_allow_html=True
            )
            # st.write(f"Result: Non-Stationary (H0: Non-Stationary, p-value: {round(p_value, 3)})")


    # Decompose the time series into components

    st.write(
        f"<div style='font-size:25px; color:black; font-weight:bold;'> Decompose the time series into components </div>",
        unsafe_allow_html=True)

    st.write("")          # to leave a space in the web page presentation

    st.write(
        "<div style='font-size:15px; color:red; font-weight:bold;'>* Select time period in days for seasonality </div>",
        unsafe_allow_html=True)

    per = st.selectbox('', [None, 30, 120, 180, 365], index=0)

    if per:

        def ts_decompose(y, model="additive", stationary=False, n_per=7):
            result = seasonal_decompose(y, model=model, period=n_per)
            fig, axes = plt.subplots(4, 1, sharex=True, sharey=False, figsize=(15, 10))

            axes[0].set_title("Decomposition for " + model + " model")
            axes[0].plot(y, label='Original ' + model)
            axes[0].legend(loc='upper left')

            axes[1].plot(result.trend, label='Trend')
            axes[1].legend(loc='upper left')

            axes[2].plot(result.seasonal, 'g', label='Seasonality')
            axes[2].legend(loc='upper left')

            axes[3].plot(result.resid, 'r', label='Residuals')
            axes[3].legend(loc='upper left')
            st.pyplot(fig)

            if stationary:
                st.write("Checking stationarity of the original series after decomposition:")
                is_stationary(y)

        ts_decompose(data2['Close'], stationary=False, n_per=per)

def ACF_PACF():

    if selected_stocks:
        @st.cache_data
        def load_data(ticker):
            data_r = yf.download(ticker, START, TODAY)
            data_r.reset_index(inplace=True)
            return data_r

        data_load_state = st.text("Load data...")
        data_r = load_data(selected_stocks)
        data_r = data_r.fillna(method='bfill')
        st.write("")
        data_load_state.text("Loading data...done ✅")

    else:
        st.header('Select please a stock data')

    data3 = data_r.copy()

    st.write(f"<div style='font-size:25px; color:black; font-weight:bold;'> {selected_stocks} </div>",
             unsafe_allow_html=True)

    st.write("")              # to leave a space in the web page presentation
    st.write("")              # to leave a space in the web page presentation

    st.write(f"<div style='font-size:20px; color:black; font-weight:bold;'> 📉 Plotting the ACF and PACF  </div>",
             unsafe_allow_html=True)

    st.write("")              # to leave a space in the web page presentation
    st.write("")              # to leave a space in the web page presentation

    #from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    # Create a figure with specified size
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Plot ACF on the first subplot
    plot_acf(data3['Close'], ax=ax[0], lags=30)
    ax[0].set_title('Autocorrelation Function (ACF)')

    # Plot PACF on the second subplot
    plot_pacf(data3['Close'], ax=ax[1], lags=30)
    ax[1].set_title('Partial Autocorrelation Function (PACF)')

    # Adjust layout for better fit
    plt.tight_layout()

    # Show grid on both plots
    for a in ax:
        a.grid(True)

    # Display the figure in Streamlit
    st.pyplot(fig)

    st.write("")  # to leave a space in the web page presentation

def model_train():

    if selected_stocks:
        @st.cache_data
        def load_data(ticker):
            data_r = yf.download(ticker, START, TODAY)
            data_r.reset_index(inplace=True)
            return data_r

        data_load_state = st.text("Load data...")
        data_r = load_data(selected_stocks)
        data_r = data_r.fillna(method='bfill')
        st.write("")              # to leave a space in the web page presentation


    else:
        st.header('Select please a stock data')

    data4 = data_r.copy()

    st.write(f"<div style='font-size:25px; color:black; font-weight:bold;'> {selected_stocks} </div>",
             unsafe_allow_html=True)

    st.write("")  # to leave a space in the web page presentation
    st.write("")  # to leave a space in the web page presentation

    data_load_state.text("Loading data ..... done  ✅")

    st.write("")
    st.write("")

    # # Model training , validation and building base model


    st.write(f"<div style='font-size:25px; color:black; font-weight:bold;'>Model training , validation and building base model \
             </div>", unsafe_allow_html=True)

    st.write("")              # to leave a space in the web page presentation
    st.write("")              # to leave a space in the web page presentation


    data4.index = data4["Date"]
    data4.drop("Date", axis=1, inplace=True)
    data4 = data4[['Close']]

    train_df = None  # Initialize with a default value
    n_days = None

    st.write("")              # to leave a space in the web page presentation

    st.write(
        "<div style='font-size:15px; color:red; font-weight:bold;'>* Select the time periyod you want to forecast in days </div>",
        unsafe_allow_html=True)

    n_days = st.selectbox('', [None, 15, 30, 45, 60, 90], index=0)

    st.write("")  # to leave a space in the web page presentation

    if n_days:

        inp_train_date = pd.to_datetime(data4.index.max())

        st.write(
            f"<div style='font-size:18px; color:red; font-weight:bold;'> 🚨 Last Date is : {inp_train_date.strftime('%Y-%m-%d')}</div>",
            unsafe_allow_html=True)

        st.write("")  # to leave a space in the web page presentation
        st.write("")  # to leave a space in the web page presentation

        st.text("Separate the dataset into training and test for model training and validation")

        st.write("")  # to leave a space in the web page presentation

        st.write(" ❗ ... Explanation ... ❗  ")
        st.write("")  # to leave a space in the web page presentation
        st.text("-------------------------------------------------------------------------")
        st.text("The data set covers the working day.")
        st.text("There are no observation values in the dataset for weekend,")
        st.text("public and religious holidays")
        st.text("-------------------------------------------------------------------------")
        st.text(" Example 1 : Last day date is Wednesday , 2024-06-19,")
        st.text("and you want to forecast 30 days from last day (forecast start day is Thursday,")
        st.text("2024-06-20 )")
        st.text("The date one year ago is Tuesday, 2023-06-20,")
        st.text("so choose Thursday 2023-06-22")
        st.write("")  # to leave a space in the web page presentation
        st.text("-------------------------------------------------------------------------")
        st.text(" Example 2 : Last day date is Monday, 2024-06-24,")
        st.text("and you want to forecast 30 days from today (forecast start day is Tuesday")
        st.text("2024-06-25 )")
        st.text("The date one year ago is Sunday, 2023-06-25, ")
        st.text("so choose Tuesday 2023-06-27 ")
        st.write("")  # to leave a space in the web page presentation
        st.text("-------------------------------------------------------------------------")

        st.write("")  # to leave a space in the web page presentation

        st.write(
            "<div style='font-size:15px; color:red; font-weight:bold;'> 📅 * Select the training date for the training dataset </div>",
            unsafe_allow_html=True)

        inp_train_date1 = pd.to_datetime(st.date_input(" ", inp_train_date))

        st.write(
            "<div style='font-size:15px; color:red; font-weight:bold;'> The train date must be the same as the date one year ago </div>",
            unsafe_allow_html=True)

        st.text(" ❗❗❗  Read the explanation above before selecting the training date ❗❗❗ ")


        if inp_train_date :

            start_date = pd.to_datetime(inp_train_date1)
            end_date = start_date + pd.offsets.BDay(n_days)

            train_df = data4.loc[data4.index < start_date]

            st.write("")  # to leave a space in the web page presentation

            # Test veri setini belirleme

            test = data4.loc[(data4.index >= start_date) & (data4.index <= end_date)]
            test_shape = test.shape[0]

            st.write("")  # to leave a space in the web page presentation

            # Triple Exponential Smoothing (Holt-Winters)  ###############################################
            st.write(
                f"<div style='font-size:25px; color:black; font-weight:bold;'> 📄 Triple Exponential Smoothing (Holt-Winters) Method </div>",
                unsafe_allow_html=True)

            st.write("")  # to leave a space in the web page presentation

            # TES = SES + DES + Mevsimsellik

            st.write(
                "<div style='font-size:15px; color:red; font-weight:bold;'> * Select seasonal periods (month) </div>",
                unsafe_allow_html=True)

            seasonal_periods_month = st.selectbox('', [None, 2, 3, 4, 6], index=0)
            seasonal_periods_month_value = seasonal_periods_month

            st.write("")  # to leave a space in the web page presentation

            if seasonal_periods_month:
                st.write(
                    f"<div style='font-size:15px;color:red; font-weight:bold;'> Seasonal periods : {seasonal_periods_month_value}</div>",
                    unsafe_allow_html=True)
                st.write("")
                tes_model = ExponentialSmoothing(train_df,
                                                 trend="add",
                                                 seasonal="add",
                                                 seasonal_periods=seasonal_periods_month_value).fit(smoothing_level=0.5,
                                                                                                    smoothing_slope=0.5,
                                                                                                    smoothing_seasonal=0.5)
                st.write("")  # to leave a space in the web page presentation

                st.text("Base model building ...... done ✅ ")

                st.write("")  # to leave a space in the web page presentation

                y_pred = tes_model.forecast(test_shape)
                y_pred = pd.Series(y_pred.values, index=test.index)

                st.session_state['y_pred'] = y_pred
                mae1 = mean_absolute_error(test, y_pred)
                st.write(f"<div style='font-size:18px;'> ▶️ Base Model MAE (mean absolute error) : {round(mae1, 3)}</div>",
                         unsafe_allow_html=True)

                st.write("")  # to leave a space in the web page presentation

                st.write(
                    "<div style='font-size:20px; color:black; font-weight:bold;'> 📉 Plot of test and predicted values (Base Model) </div>",
                    unsafe_allow_html=True)

                st.write("")  # to leave a space in the web page presentation

                y_pred = st.session_state['y_pred']

                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(test.index, test['Close'], label='Test Data)', color='orange')
                ax.plot(y_pred.index, y_pred, label='Predictions', color='green', linestyle='--')
                ax.set_title('Time Series Plot : Testing and Predictions')
                ax.set_xlabel('Date')
                ax.set_ylabel('Price')
                ax.legend(loc='best')
                ax.grid(True)
                st.pyplot(fig)

                # test ve tahmin veri setlerini gösterme

                st.write("")

                st.write(
                    f"<div style='font-size:20px; color:black; font-weight:bold;'> List of test and predicted values (Base Model) </div>",
                    unsafe_allow_html=True)
                st.write("")

                Comp1 = pd.DataFrame({"Actual": test.squeeze(), "Predict": y_pred.squeeze()})

                Comp1.index = Comp1.index.strftime('%Y-%m-%d')

                st.dataframe(Comp1)

        return data4, train_df, test, test_shape, seasonal_periods_month_value, inp_train_date


def hyper_parameter_optimization():

    data4,train_df, test, test_shape, seasonal_periods_month_value, inp_train_date = model_train()

    # Hyperparameter Optimization

    st.write(
        f"<div style='font-size:25px; color:black; font-weight:bold;'> Hyperparameter Optimization </div>",
        unsafe_allow_html=True)

    st.write("")  # to leave a space in the web page presentation

    alphas = np.arange(0.1, 0.9, 0.05)
    betas = np.arange(0.1, 0.5, 0.05)
    gammas = np.arange(0.1, 0.5, 0.05)

    abg = list(itertools.product(alphas, betas, gammas))

    st.text("Hyper parameter optimization in progress .........  ❗❗❗ ")
    st.write("")  # to leave a space in the web page presentation
    st.text("This will take just a sec  ........  ❗❗❗ ")


    def tes_optimizer(train, abg, step=test_shape):
        best_alpha, best_beta, best_gamma, best_mae = None, None, None, float("inf")
        for comb in abg:
            tes_model = ExponentialSmoothing(train, trend="add", seasonal="add",
                                             seasonal_periods=seasonal_periods_month_value). \
                fit(smoothing_level=comb[0], smoothing_slope=comb[1], smoothing_seasonal=comb[2])
            y_pred = tes_model.forecast(step)
            mae = mean_absolute_error(test, y_pred)
            if mae < best_mae:
                best_alpha, best_beta, best_gamma, best_mae = comb[0], comb[1], comb[2], mae
            print([round(comb[0], 2), round(comb[1], 2), round(comb[2], 2), round(mae, 2)])

        print("best_alpha:", round(best_alpha, 2), "best_beta:", round(best_beta, 2), "best_gamma:",
              round(best_gamma, 2),
              "best_mae:", round(best_mae, 4))

        return best_alpha, best_beta, best_gamma, best_mae

    best_alpha, best_beta, best_gamma, best_mae = tes_optimizer(train_df, abg)


    st.write(f"<div style='font-size:18px;'> best alpha : {round(best_alpha, 3)}</div>",
             unsafe_allow_html=True)
    st.write(f"<div style='font-size:18px;'> best beta : {round(best_beta, 3)}</div>",
             unsafe_allow_html=True)
    st.write(f"<div style='font-size:18px;'> best gamma : {round(best_gamma, 3)}</div>",
             unsafe_allow_html=True)
    st.write(f"<div style='font-size:18px;'> test_shape : {test_shape}</div>",
        unsafe_allow_html=True)

    st.write("")  # to leave a space in the web page presentation

    # Final TES Model

    final_tes_model = ExponentialSmoothing(train_df, trend="add", seasonal="add",
                                           seasonal_periods=seasonal_periods_month_value). \
        fit(smoothing_level=best_alpha, smoothing_trend=best_beta, smoothing_seasonal=best_gamma)

    st.write("")  # to leave a space in the web page presentation

    st.text("Final model building ...... done ✅ ")

    st.write("")  # to leave a space in the web page presentation

    y_pred_f = final_tes_model.forecast(test_shape)

    y_pred_f = pd.Series(y_pred_f.values, index=test.index)
    mae_f = mean_absolute_error(test, y_pred_f)


    st.write(f"<div style='font-size:18px;'> ▶️ Final Model MAE (mean absolute error) : {round(mae_f, 3)}</div>",
             unsafe_allow_html=True)

    st.write("")  # to leave a space in the web page presentation


    st.write(
        f"<div style='font-size:20px; color:black; font-weight:bold;'> 📉 Plot of test and predicted values (Final Model) </div>",
        unsafe_allow_html=True)

    st.write("")  # to leave a space in the web page presentation

    # Plot

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(test.index, test['Close'], label='Test Data)', color='orange')
    ax.plot(y_pred_f.index, y_pred_f, label='Predictions', color='green',
            linestyle='--')
    ax.set_title('Time Series Plot : Testing and Predictions')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend(loc='best')
    ax.grid(True)
    st.pyplot(fig)

    # test ve tahmin veri setlerini gösterme
    st.write("")  # to leave a space in the web page presentation
    st.write(
        f"<div style='font-size:20px; color:black; font-weight:bold;'> List of test and predicted values (Final Model) </div>",
        unsafe_allow_html=True)
    st.write("")
    Compf = pd.DataFrame({"Actual": test.squeeze(), "Predict": y_pred_f.squeeze()})
    Compf.index = Compf.index.strftime('%Y-%m-%d')

    st.dataframe(Compf)

    st.write("")  # to leave a space in the web page presentation
    st.write("")  # to leave a space in the web page presentation

    ######## forcast

    st.write(
        f"<div style='font-size:20px; color:black; font-weight:bold;'> Forecasting Results  </div>",
        unsafe_allow_html=True)

    alfa = best_alpha
    beta = best_beta
    gamma = best_gamma

    seasonal_per_month = seasonal_periods_month_value
    shape = test_shape

    pred_date_input = inp_train_date + pd.offsets.BDay(1)

    final_tes_model_all = ExponentialSmoothing(data4, trend="add", seasonal="add",
                                               seasonal_periods=seasonal_per_month). \
        fit(smoothing_level=alfa, smoothing_trend=beta, smoothing_seasonal=gamma)

    st.write("")  # to leave a space in the web page presentation

    st.text(" Model building ........... done ✅")

    y_pred_all = final_tes_model_all.forecast(shape)

    st.write("")  # to leave a space in the web page presentation

    st.text("Forecasts completed ........  ✅")

    # Define the start date for prediction

    start_date = pd.to_datetime(pred_date_input)

    st.write("")  # to leave a space in the web page presentation

    # Tahmin için tarih aralığını hesaplama

    period = shape
    end_date = start_date + pd.offsets.BDay(period)

    date_range = pd.date_range(start=start_date, end=end_date, freq='B')
    date_range = date_range[:-1]

    st.write("")  # to leave a space in the web page presentation

    st.write(
        f"<div style='font-size:20px; color:black; font-weight:bold;'> List of Forecasts </div>",
        unsafe_allow_html=True)

    st.write("")  # to leave a space in the web page presentation

    Prediction = pd.DataFrame({'Date': date_range,
                               'Prediction Stock Close Price': y_pred_all})
    Prediction.index = Prediction['Date']
    Prediction.drop("Date", axis=1, inplace=True)
    Prediction.index = Prediction.index.strftime('%Y-%m-%d')

    st.dataframe(Prediction)



# Navigation options

if options == 'Stock data review':
    home()

elif options == 'Dickey-Fuller test':
    dickey_fuller()

elif options == 'Decompose the time series into components':
    decompose()

elif options == 'Plotting the ACF and PACF':
    ACF_PACF()

elif options == 'Model training, base model building and validation':
    model_train()

elif options == 'Hyper parameter optimization, model building and validation':
    hyper_parameter_optimization()






# streamlit run .\stock_pred\stock_prediction_app_str_v2.py






