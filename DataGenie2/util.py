import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler



async def load_data(file_name):
    data = pd.read_csv(file_name)
    data = pd.read_csv('sample_1 (1).csv', parse_dates=['point_timestamp'])
    data.set_index('point_timestamp', inplace=True)
    return data


async def preprocess(data):
    data['point_value'] = data['point_value'].interpolate(method='linear')
    data = data.dropna()



async def extract_features(data, period):
    return {
        "seasonality": await f_seasonality(data, period),
        "stationary": await f_stationary(data)
    }


async def f_stationary(data):
    result = adfuller(data['y'])
    if result[1] > 0.05:
        data['y'] = data['y'].diff().dropna()
        result = adfuller(data['y'])
    return result[1]


async def f_autocorrelation(data):
    pass


async def f_seasonality(data, period):
    return seasonal_decompose(data['y'], model='additive', period=period)


async def classify(features):
    seasonality = features["seasonality"]
    stationary = features["stationary"]

    if seasonality.seasonal is not None:
        return "SARIMA"
    elif seasonality.trend is not None:
        return "XG_BOOST"
    else:
        return "PROPHET"


async def fit_and_predict(model, data, period):
    if model == "SARIMA":
        model = SARIMAX(data["y"], order=(1, 1, 0), seasonal_order=(1, 0, 0, 24))
        results = model.fit()
        forecast = results.get_forecast(steps=period)
        # forecast_ci = forecast.conf_int()
        forecast_mean = forecast.predicted_mean
        return forecast_mean.values.tolist()
    elif model == "PROPHET":
        temp_data = data.copy()
        model = Prophet()
        temp_data.reset_index(inplace=True)
        temp_data.rename(columns={'x': 'ds'}, inplace=True)
        model.fit(temp_data)
        future = model.make_future_dataframe(periods=period, freq='D')
        forecast = model.predict(future)
        return forecast["yhat"][-period:].values.tolist()

    elif model == "XGBOOST":
        train_size = int(len(data) * 0.8)
        train_data, test_data = data.iloc[:train_size], data.iloc[train_size:]

        tscv = TimeSeriesSplit(n_splits=5)
        for fold, (train_index, test_index) in enumerate(tscv.split(data)):
            X_train, X_test = data.iloc[train_index], data.iloc[test_index]
            y_train, y_test = data.iloc[train_index]['point_value'], data.iloc[test_index]['point_value']

        # Train XGBoost model
        model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)
        model.fit(X_train, y_train)
        forecast_diff = model.predict(X_test)
        return forecast_diff.values.tolist()



