import warnings
from math import sqrt
import lightgbm as lgb
import matplotlib as mpl
import numpy as np
import pandas as pd
import pmdarima as pm
import shap
import statsmodels as sm
import tensorflow as tf
import xgboost as xgb
from bayes_opt import BayesianOptimization
from prophet import Prophet
from gluonts.dataset.common import ListDataset
from gluonts.evaluation.backtest import make_evaluation_predictions
from matplotlib import pyplot as plt
from pylab import rcParams
from sklearn import linear_model, svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa import api as smt
from statsmodels.tsa.arima_model import ARIMA, ARMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

# We will use deprecated models of statmodels which throw a lot of warnings to use more modern ones
warnings.filterwarnings("ignore")


# Extra settings
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)
plt.style.use('bmh')
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['text.color'] = 'k'


async def load_data(file_name):
    data = pd.read_csv(file_name)
    data = pd.read_csv('sample_1 (1).csv', parse_dates=['point_timestamp'])
    data.set_index('point_timestamp', inplace=True)
    return data


async def preprocess_predict(data):
    plt.figure(num=None, figsize=(30, 10), dpi=80, facecolor='w', edgecolor='k')
    plt.title('Sample', fontsize=30)

    plt.plot(data.point_value)
    # Decomposing our time series
    # Automatic time series decomposition
    rcParams['figure.figsize'] = 18, 8
    plt.figure(num=None, figsize=(50, 20), dpi=80, facecolor='w', edgecolor='k')
    series = data.point_value[:365]
    result = seasonal_decompose(series, model='multiplicative')
    result.plot()
    # Trend
    # Now we will try some methods to check for trend in our series:

    # Automatic decomposing
    # Moving average
    # Fit a linear regression model to identify trend
    fig = plt.figure(figsize=(15, 7))
    layout = (3, 2)
    pm_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
    mv_ax = plt.subplot2grid(layout, (1, 0), colspan=2)
    fit_ax = plt.subplot2grid(layout, (2, 0), colspan=2)

    pm_ax.plot(result.trend)
    pm_ax.set_title("Automatic decomposed trend")

    mm = data.point_value.rolling(12).mean()
    mv_ax.plot(mm)
    mv_ax.set_title("Moving average 12 steps")

    X = [i for i in range(0, len(data.point_value))]
    X = np.reshape(X, (len(X), 1))
    y = data.point_value
    model = LinearRegression()
    model.fit(X, y)
    # calculate trend
    trend = model.predict(X)
    fit_ax.plot(trend)
    fit_ax.set_title("Trend fitted by linear regression")

    plt.tight_layout()

    # Seasonality
    rcParams['figure.figsize'] = 18, 8
    plt.figure(num=None, figsize=(50, 20), dpi=80, facecolor='w', edgecolor='k')
    series = data.point_value[:365]
    result = seasonal_decompose(series, model='multiplicative')
    result.plot()

    rcParams['figure.figsize'] = 18, 8
    plt.figure(num=None, figsize=(50, 20), dpi=80, facecolor='w', edgecolor='k')
    series = data.point_value[-365:]
    result = seasonal_decompose(series, model='multiplicative')
    result.plot()

    # INTERPRETATION
    # Looking for weekly seasonality
    resample = data.resample('W')
    weekly_mean = resample.mean()
    weekly_mean.point_value.plot(label='Weekly mean')
    plt.title("Resampled series to weekly mean values")
    plt.legend()

    # Noise

    fig = plt.figure(figsize=(12, 7))
    layout = (2, 2)
    hist_ax = plt.subplot2grid(layout, (0, 0))
    ac_ax = plt.subplot2grid(layout, (1, 0))
    hist_std_ax = plt.subplot2grid(layout, (0, 1))
    mean_ax = plt.subplot2grid(layout, (1, 1))

    data.point_value.hist(ax=hist_ax)
    hist_ax.set_title("Original series histogram")

    plot_acf(series, lags=30, ax=ac_ax)
    ac_ax.set_title("Autocorrelation")

    mm = data.point_value.rolling(7).std()
    mm.hist(ax=hist_std_ax)
    hist_std_ax.set_title("Standard deviation histogram")

    mm = data.point_value.rolling(30).mean()
    mm.plot(ax=mean_ax)
    mean_ax.set_title("Mean over time")

    # Stationarity
    # Autocorrelation and Partial autocorrelation plots
    plot_acf(series, lags=30)
    plot_pacf(series, lags=30)
    plt.show()

    # Rolling means and standard deviation of our series
    # Determing rolling statistics
    rolmean = data.point_value.rolling(window=12).mean()
    rolstd = data.point_value.rolling(window=12).std()

    # Plot rolling statistics:
    orig = plt.plot(data.point_value, label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)

    # Augmented Dickey-Fuller test
    X = data.point_value.values
    result = adfuller(X)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))

    # Making Time Series Stationary
    data1 = pd.read_csv("sample_1 (1).csv")
    data1.point_value.plot(label='Original')
    data1.point_value.rolling(window=12).mean().plot(
        color='red', label='Windowed mean')
    data1.point_value.rolling(window=12).std().plot(
        color='black', label='Std mean')
    plt.legend()
    plt.title('Original vs Windowed mean vs Windowed std')

    # Difference transform
    def difference(dataset, interval=1, order=1):
        for u in range(order):
            diff = list()
            for i in range(interval, len(dataset)):
                value = dataset[i] - dataset[i - interval]
                diff.append(value)
            dataset = diff
        return diff

    lag1series = pd.Series(difference(data.point_value, interval=1, order=1))
    lag3series = pd.Series(difference(data.point_value, interval=3, order=1))
    lag1order2series = pd.Series(difference(
        data.point_value, interval=1, order=2))

    fig = plt.figure(figsize=(14, 11))
    layout = (3, 2)
    original = plt.subplot2grid(layout, (0, 0), colspan=2)
    lag1 = plt.subplot2grid(layout, (1, 0))
    lag3 = plt.subplot2grid(layout, (1, 1))
    lag1order2 = plt.subplot2grid(layout, (2, 0), colspan=2)

    original.set_title('Original series')
    original.plot(data.point_value, label='Original')
    original.plot(data.point_value.rolling(
        7).mean(), color='red', label='Rolling Mean')
    original.plot(data.point_value.rolling(7).std(),
                  color='black', label='Rolling Std')
    original.legend(loc='best')

    lag1.set_title('Difference series with lag 1 order 1')
    lag1.plot(lag1series, label="Lag1")
    lag1.plot(lag1series.rolling(7).mean(), color='red', label='Rolling Mean')
    lag1.plot(lag1series.rolling(7).std(), color='black', label='Rolling Std')
    lag1.legend(loc='best')

    lag3.set_title('Difference series with lag 3 order 1')
    lag3.plot(lag3series, label="Lag3")
    lag3.plot(lag3series.rolling(7).mean(), color='red', label='Rolling Mean')
    lag3.plot(lag3series.rolling(7).std(), color='black', label='Rolling Std')
    lag3.legend(loc='best')

    lag1order2.set_title('Difference series with lag 1 order 2')
    lag1order2.plot(lag1order2series, label="Lag1order2")
    lag1order2.plot(lag1order2series.rolling(7).mean(),
                    color='red', label='Rolling Mean')
    lag1order2.plot(lag1order2series.rolling(7).std(),
                    color='black', label='Rolling Std')
    lag1order2.legend(loc='best')

    # Log scale transformation
    ts_log = np.log(data.point_value)
    ts_log.plot(label='Log scale result')
    ts_log.rolling(window=12).mean().plot(color='red', label='Windowed mean')
    ts_log.rolling(window=12).std().plot(color='black', label='Std mean')
    plt.legend()
    plt.title('Log scale transformation into original series')

    # Smoothing
    avg = pd.Series(ts_log).rolling(12).mean()
    plt.plot(avg, label='Log scale smoothed with windows 12')
    avg.rolling(window=12).mean().plot(color='red', label='Windowed mean')
    avg.rolling(window=12).std().plot(color='black', label='Std mean')
    plt.legend()

    ts_log_moving_avg_diff = ts_log - avg

    ts_log_moving_avg_diff.plot(label='Original')
    ts_log_moving_avg_diff.rolling(12).mean().plot(
        color='red', label="Rolling mean")
    ts_log_moving_avg_diff.rolling(12).std().plot(
        color='black', label="Rolling mean")
    plt.legend(loc='best')

    # Methods for time series forecasting

    # We split our dataset to be able to evaluate our models

    resultsDict = {}
    predictionsDict = {}

    split_fraction = 0.8  # 80% for training, 20% for testing
    n_obs = len(data)
    split_index = int(split_fraction * n_obs)
    split_date = data.index[split_index]

    df_training = data.loc[data.index <= split_date]
    df_test = data.loc[data.index > split_date]

    print(f"{len(df_training)} days of training data \n {len(df_test)} days of testing data ")

    # Univariate-time-series-forecasting
    from tqdm import tqdm

    # Simple Exponential Smoothing (SES)

    # Walk throught the test data, training and predicting 1 day ahead for all the test data
    index = len(df_training)
    yhat = list()
    for t in tqdm(range(len(df_test.point_value))):
        temp_train = data[:len(df_training) + t]
        model = SimpleExpSmoothing(temp_train.point_value)
        model_fit = model.fit()
        predictions = model_fit.predict(start=len(temp_train), end=len(temp_train))
        yhat = yhat + [predictions]

    yhat = pd.concat(yhat)
    resultsDict['SES'] = evaluate(df_test.point_value, yhat.values)
    predictionsDict['SES'] = yhat.values
    # print(predictionsDict['SES'])

    # Univariate-time-series-forecasting
    from tqdm import tqdm

    # Simple Exponential Smoothing (SES)

    # Walk throught the test data, training and predicting 1 day ahead for all the test data
    index = len(df_training)
    yhat = list()
    for t in tqdm(range(len(df_test.point_value))):
        temp_train = data[:len(df_training) + t]
        model = SimpleExpSmoothing(temp_train.point_value)
        model_fit = model.fit()
        predictions = model_fit.predict(start=len(temp_train), end=len(temp_train))
        yhat = yhat + [predictions]

    yhat = pd.concat(yhat)
    resultsDict['SES'] = evaluate(df_test.point_value, yhat.values)
    predictionsDict['SES'] = yhat.values
    # print(predictionsDict['SES'])

    # Holt Winter’s Exponential Smoothing (HWES)

    # Walk throught the test data, training and predicting 1 day ahead for all the test data
    index = len(df_training)
    yhat = list()
    for t in tqdm(range(len(df_test.point_value))):
        temp_train = data[:len(df_training) + t]
        model = ExponentialSmoothing(temp_train.point_value)
        model_fit = model.fit()
        predictions = model_fit.predict(start=len(temp_train), end=len(temp_train))
        yhat = yhat + [predictions]

    yhat = pd.concat(yhat)
    resultsDict['HWES'] = evaluate(df_test.point_value, yhat.values)
    predictionsDict['HWES'] = yhat.values

    # ARIMA

    from statsmodels.tsa.arima.model import ARIMA

    # Walk throught the test data, training and predicting 1 day ahead for all the test data
    index = len(df_training)
    yhat = list()
    for t in tqdm(range(len(df_test.point_value))):
        temp_train = data[:len(df_training) + t]
        model = ARIMA(temp_train.point_value, order=(1, 0, 0))
        model_fit = model.fit()
        predictions = model_fit.predict(
            start=len(temp_train), end=len(temp_train), dynamic=False)
        yhat = yhat + [predictions]

    yhat = pd.concat(yhat)
    resultsDict['ARIMA'] = evaluate(df_test.point_value, yhat.values)
    predictionsDict['ARIMA'] = yhat.values

    plt.plot(df_test.point_value.values, label='Original')
    plt.plot(yhat.values, color='red', label='ARIMA predicted')
    plt.legend()

    # Auto ARIMA

    # building the model

    autoModel = pm.auto_arima(df_training.point_value, trace=True,
                              error_action='ignore', suppress_warnings=True, seasonal=False)
    autoModel.fit(df_training.point_value)

    order = autoModel.order
    yhat = list()
    for t in tqdm(range(len(df_test.point_value))):
        temp_train = data[:len(df_training) + t]
        model = ARIMA(temp_train.point_value, order=order)
        model_fit = model.fit()
        predictions = model_fit.predict(
            start=len(temp_train), end=len(temp_train), dynamic=False)
        yhat = yhat + [predictions]

    yhat = pd.concat(yhat)
    resultsDict['AutoARIMA {0}'.format(order)] = evaluate(
        df_test.point_value, yhat)
    predictionsDict['AutoARIMA {0}'.format(order)] = yhat.values

    plt.plot(df_test.point_value.values, label='Original')
    plt.plot(yhat.values, color='red', label='AutoARIMA {0}'.format(order))
    plt.legend()

    # Seasonal Autoregressive Integrated Moving-Average (SARIMA)

    # SARIMA example

    # Walk throught the test data, training and predicting 1 day ahead for all the test data
    index = len(df_training)
    yhat = list()
    for t in tqdm(range(len(df_test.point_value))):
        temp_train = data[:len(df_training) + t]
        model = SARIMAX(temp_train.point_value, order=(
            1, 0, 0), seasonal_order=(0, 0, 0, 3))
        model_fit = model.fit(disp=False)
        predictions = model_fit.predict(
            start=len(temp_train), end=len(temp_train), dynamic=False)
        yhat = yhat + [predictions]

    yhat = pd.concat(yhat)
    resultsDict['SARIMAX'] = evaluate(df_test.point_value, yhat.values)
    predictionsDict['SARIMAX'] = yhat.values

    plt.plot(df_test.point_value.values, label='Original')
    plt.plot(yhat.values, color='red', label='SARIMAX')
    plt.legend()

    # prophet

    # Prophet needs some specifics data stuff, coment it here
    prophet_training = df_training.rename(
        columns={'point_value': 'y'})  # old method
    prophet_training['ds'] = prophet_training.index
    prophet_training.index = pd.RangeIndex(len(prophet_training.index))

    prophet_test = df_test.rename(columns={'point_value': 'y'})  # old method
    prophet_test['ds'] = prophet_test.index
    prophet_test.index = pd.RangeIndex(len(prophet_test.index))

    prophet = Prophet(
        growth='linear',
        seasonality_mode='multiplicative',
        holidays_prior_scale=20,
        daily_seasonality=False,
        weekly_seasonality=False,
        yearly_seasonality=False
    ).add_seasonality(
        name='monthly',
        period=30.5,
        fourier_order=55
    ).add_seasonality(
        name='daily',
        period=1,
        fourier_order=15
    ).add_seasonality(
        name='weekly',
        period=7,
        fourier_order=25
    ).add_seasonality(
        name='yearly',
        period=365.25,
        fourier_order=20
    ).add_seasonality(
        name='quarterly',
        period=365.25 / 4,
        fourier_order=55
    ).add_country_holidays(country_name='China')

    prophet.fit(prophet_training)
    yhat = prophet.predict(prophet_test)
    resultsDict['Prophet univariate'] = evaluate(
        df_test.point_value, yhat.yhat.values)
    predictionsDict['Prophet univariate'] = yhat.yhat.values

    plt.plot(df_test.point_value.values, label='Original')
    plt.plot(yhat.yhat, color='red', label='Prophet univariate')
    plt.legend()

    # Multivariate time series forecasting
    # ADD time features to our model
    def create_time_features(df, target=None):
        """
        Creates time series features from datetime index
        """
        df['date'] = df.index
        df['hour'] = df['date'].dt.hour
        df['dayofweek'] = df['date'].dt.dayofweek
        df['quarter'] = df['date'].dt.quarter
        df['month'] = df['date'].dt.month
        df['year'] = df['date'].dt.year
        df['dayofyear'] = df['date'].dt.dayofyear
        df['sin_day'] = np.sin(df['dayofyear'])
        df['cos_day'] = np.cos(df['dayofyear'])
        df['dayofmonth'] = df['date'].dt.day
        df['weekofyear'] = df['date'].dt.weekofyear
        X = df.drop(['date'], axis=1)
        if target:
            y = df[target]
            X = X.drop([target], axis=1)
            return X, y

        return X

    X_train_df, y_train = create_time_features(
        df_training, target='point_value')
    X_test_df, y_test = create_time_features(df_test, target='point_value')
    scaler = StandardScaler()
    scaler.fit(X_train_df)  # No cheating, never scale on the training+test!
    X_train = scaler.transform(X_train_df)
    X_test = scaler.transform(X_test_df)

    X_train_df = pd.DataFrame(X_train, columns=X_train_df.columns)
    X_test_df = pd.DataFrame(X_test, columns=X_test_df.columns)

    # Linear models

    # Bayesian regression
    reg = linear_model.BayesianRidge()
    reg.fit(X_train, y_train)
    yhat = reg.predict(X_test)
    resultsDict['BayesianRidge'] = evaluate(df_test.point_value, yhat)
    predictionsDict['BayesianRidge'] = yhat

    # Tree models
    # Randomforest
    reg = RandomForestRegressor(max_depth=2, random_state=0)
    reg.fit(X_train, y_train)
    yhat = reg.predict(X_test)
    resultsDict['Randomforest'] = evaluate(df_test.point_value, yhat)
    predictionsDict['Randomforest'] = yhat

    # XGBoost
    reg = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000)
    reg.fit(X_train, y_train,
            verbose=False)  # Change verbose to True if you want to see it train
    yhat = reg.predict(X_test)
    resultsDict['XGBoost'] = evaluate(df_test.point_value, yhat)
    predictionsDict['XGBoost'] = yhat
    plt.plot(df_test.point_value.values, label='Original')
    plt.plot(yhat, color='red', label='XGboost')
    plt.legend()

    # For our dl model we will create windows of data that will be feeded into the datasets, for each timestemp T we will append the data from T-7 to T to the Xdata with target Y(t)
    BATCH_SIZE = 64
    BUFFER_SIZE = 100
    WINDOW_LENGTH = 24

    def window_data(X, Y, window=7):
        '''
        The dataset length will be reduced to guarante all samples have the window, so new length will be len(dataset)-window
        '''
        x = []
        y = []
        for i in range(window - 1, len(X)):
            x.append(X[i - window + 1:i + 1])
            y.append(Y[i])
        return np.array(x), np.array(y)

    # Since we are doing sliding, we need to join the datasets again of train and test
    X_w = np.concatenate((X_train, X_test))
    y_w = np.concatenate((y_train, y_test))

    X_w, y_w = window_data(X_w, y_w, window=WINDOW_LENGTH)
    X_train_w = X_w[:-len(X_test)]
    y_train_w = y_w[:-len(X_test)]
    X_test_w = X_w[-len(X_test):]
    y_test_w = y_w[-len(X_test):]

    # Check we will have same test set as in the previous models, make sure we didnt screw up on the windowing
    print(f"Test set equal: {np.array_equal(y_test_w, y_test)}")

    train_data = tf.data.Dataset.from_tensor_slices((X_train_w, y_train_w))
    train_data = train_data.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

    val_data = tf.data.Dataset.from_tensor_slices((X_test_w, y_test_w))
    val_data = val_data.batch(BATCH_SIZE).repeat()

    dropout = 0.0
    simple_lstm_model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(
            128, input_shape=X_train_w.shape[-2:], dropout=dropout),
        tf.keras.layers.Dense(128),
        tf.keras.layers.Dense(128),
        tf.keras.layers.Dense(1)
    ])

    simple_lstm_model.compile(optimizer='rmsprop', loss='mae')

    EVALUATION_INTERVAL = 200
    EPOCHS = 5

    model_history = simple_lstm_model.fit(train_data, epochs=EPOCHS,
                                          steps_per_epoch=EVALUATION_INTERVAL,
                                          validation_data=val_data, validation_steps=50)

    return bar_metrics(resultsDict)






