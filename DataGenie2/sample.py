import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
data = pd.read_csv('sample_1 (1).csv')

# Convert the timestamp to datetime format
data['point_timestamp'] = pd.to_datetime(data['point_timestamp'], format='%Y-%m-%d')
# Set 'point_timestamp' column as index
data.set_index('point_timestamp', inplace=True)

# Extract features
# Check stationarity using the ADF test
from statsmodels.tsa.stattools import adfuller
data['point_value'] = data['point_value'].interpolate(method='linear')
# Check for any remaining missing values
print(data.isnull().sum())
data = data.dropna()

# Check stationarity using the ADF test
result = adfuller(data['point_value'].interpolate(method='linear').dropna())
if result[1] > 0.05:
    # The series is non-stationary
    # Apply differencing
    diff_data = data['point_value'].diff().dropna()
    result = adfuller(diff_data)
    if result[1] > 0.05:
        print('Differenced series is still non-stationary.')
    else:
        print('Differenced series is stationary.')
else:
    # The series is already stationary
    print('The series is stationary.')


# Check autocorrelation using the ACF and PACF plots
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# Extract the "point_value" column from the "data" DataFrame
point_value = data['point_value']

# Plot the autocorrelation function
plot_acf(point_value, lags=20)
plt.show()
plot_pacf(point_value, lags=20)
plt.show()

# Check seasonality using seasonal decomposition
from statsmodels.tsa.seasonal import seasonal_decompose
# Perform seasonal decomposition
result = seasonal_decompose(point_value, model='additive', period=7)

# Plot the decomposition
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 8))
result.observed.plot(ax=ax1)
ax1.set_ylabel('Observed')
result.trend.plot(ax=ax2)
ax2.set_ylabel('Trend')
result.seasonal.plot(ax=ax3)
ax3.set_ylabel('Seasonal')
result.resid.plot(ax=ax4)
ax4.set_ylabel('Residual')
plt.tight_layout()
plt.show()

# Split the data into training and testing sets
train_size = int(len(data) * 0.8)
train_data, test_data = data.iloc[:train_size], data.iloc[train_size:]

# Define X_train, y_train, X_test, and y_test
X_train = train_data.drop('point_value', axis=1)
y_train = train_data['point_value']
X_test = test_data.drop('point_value', axis=1)
y_test = test_data['point_value']

# Scale the data using MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


#print(data.index.min(), data.index.max())


# Train and evaluate different models based on the extracted features
# Seasonal ARIMA or SARIMA model
if result.seasonal is not None:
    model = SARIMAX(point_value, order=(1,1,0), seasonal_order=(1,0,0,24))
    results = model.fit()
    forecast = results.get_forecast(steps=len(point_value))
    forecast_ci = forecast.conf_int()
    forecast_mean = forecast.predicted_mean
    print(forecast_mean)
    print("SARIMA")


# XGBoost model
elif result.trend is not None:
    # Scale the data using MinMaxScaler
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    train_data_scaled = scaler.fit_transform(train_data)
    test_data_scaled = scaler.transform(test_data)

        # Apply differencing technique
    diff_train_data = train_data_scaled.diff().dropna()
    diff_test_data = test_data_scaled.diff().dropna()

    # Define X_train, y_train, X_test, and y_test
    X_train = diff_train_data.drop('point_value', axis=1)
    y_train = diff_train_data['point_value']
    X_test = diff_test_data.drop('point_value', axis=1)
    y_test = diff_test_data['point_value']

    # Train XGBoost model
    model = XGBRegressor(n_estimators=1000)
    model.fit(X_train, y_train)
    forecast_diff = model.predict(X_test, )

    # Apply inverse differencing
    forecast = scaler.inverse_transform(pd.DataFrame(forecast_diff, columns=['point_value'])).squeeze()

    print("XGBoost")

else:
    model = Prophet()
    data.reset_index(inplace=True)
    data.rename(columns={'point_timestamp':'ds', 'point_value':'y'}, inplace=True)
    model.fit(data)
    future = model.make_future_dataframe(periods=31, freq='D')
    forecast = model.predict(future)
    print("Prophet")

print(forecast)

# Calculate MAPE
def calculate_mape(actual, predicted):
    """
    Calculate Mean Absolute Percentage Error (MAPE) between actual and predicted values
    """
    actual, predicted = np.array(actual), np.array(predicted)
    if np.isnan(actual).any() or np.isnan(predicted).any():
        return np.nan
    else:
        return np.mean(np.abs((actual - predicted) / actual)) * 100

print(y_test.shape)
#print(forecast.shape)


# Calculate MAPE for the model
forecast_mean = forecast.predicted_mean
y_test = y_test[-len(forecast_mean):]  # Use the same number of test data points as the length of forecast
forecast = forecast_mean[:len(y_test)] # Keep only the first len(y_test) elements of forecast
mape = calculate_mape(y_test, forecast)
print('MAPE:', mape)


# Plot the predicted values
plt.figure(figsize=(10, 6))
plt.plot(y_test.index, y_test, label='Actual')
plt.plot(forecast.index, forecast, label='Predicted', color='red')
plt.legend()
plt.title('Forecast vs Actual')
plt.show()



