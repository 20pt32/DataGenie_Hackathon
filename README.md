# DataGenie_Hackathon

I have implemented these step by step process in the code :

Time series analysis and transforms :

-->Time series decomposition
Level
Trend
Seasonality
Noise

-->Stationarity
AC and PAC plots
Rolling mean and std
Dickey-Fuller test

-->Making our time series stationary
Difference transform
Log scale
Smoothing
Moving average

-->Decomposing our time series

Additive Model:
y(t) = Level + Trend + Seasonality + Noise

Multiplicative model
y(t) = Level * Trend * Seasonality * Noise

-->Trend

will try some methods to check for trend in our series:

Automatic decomposing
Moving average
Fit a linear regression model to identify trend

-->Seasonality
Seasonality is observed when there is a distinct repeated pattern observed between regular intervals due to seasonal factors

-->Noise
Check our series histogram, does it look like a Gaussian distribution? Mean=0 and constand std
Correlation plots
Standard deviation distribution, is it a Gaussian distribution?
Does the mean or level change over time?

-->Stationarity
-->Check for sationarity
Autocorrelation and Partial autocorrelation plots
Rolling means and standard deviation of our series
Augmented Dickey-Fuller test
Making Time Series Stationary
-->Difference transform
difference(t) = observation(t) - observation(t-1)
-->Log scale transformation
LogScaleTransform(t)= Log(t)

-->Smoothing

-->Methods for time series forecasting
-->Univariate-time-series-forecasting
Simple Exponential Smoothing (SES)
Holt Winter’s Exponential Smoothing (HWES)
Autoregressive integrated moving average (ARIMA)
Auto ARIMA
Seasonal Autoregressive Integrated Moving-Average (SARIMA)
Prophet

-->Multivariate time series forecasting
-->Linear models
Bayesian regression
-->Tree models
Randomforest
XGBoost

->Deep learning
Tensorlfow LSTM

-->Finally, predicted which the best model using a bar chart




