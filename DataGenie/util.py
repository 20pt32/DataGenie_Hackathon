import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from prophet import Prophet
import matplotlib as plt
# from sklearn.preprocessing import MinMaxScaler


async def load_data(file_name):
    data = pd.read_csv(file_name)
    data['x'] = pd.to_datetime(data['point_timestamp'], format='%Y-%m-%d')
    data = data.drop("point_timestamp", axis=1)
    data['y'] = data["point_value"]
    data = data.drop("point_value", axis=1)
    data.set_index('x', inplace=True)
    data = data.drop(data.columns[0], axis=1)
    return data


async def preprocess(data):
    # log
    data['y'] = data['y'].interpolate(method='linear')
    data = data.dropna()
    return data


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



