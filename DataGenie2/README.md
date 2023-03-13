
This code reads a CSV file, preprocesses the data, checks for stationarity, autocorrelation, and seasonality of the time series, scales the data, and trains three different models to make a prediction. It also defines a function to calculate the Mean Absolute Percentage Error (MAPE) between the actual and predicted values. The models used are Seasonal ARIMA, XGBoost, and Prophet.

The first section of the code reads the CSV file, converts the timestamp to datetime format, sets the 'point_timestamp' column as the index, and then checks for missing values. It also applies linear interpolation to fill any missing values. After that, it checks the stationarity of the time series using the ADF test. If the series is non-stationary, it applies differencing and checks again for stationarity. If the differenced series is still non-stationary, it prints a message saying so; otherwise, it prints a message saying the differenced series is stationary.

The second section of the code checks for autocorrelation using the ACF and PACF plots. It then checks for seasonality using seasonal decomposition and plots the decomposition.

The third section of the code splits the data into training and testing sets, defines the X_train, y_train, X_test, and y_test, and scales the data using MinMaxScaler.

The fourth section of the code trains and evaluates different models based on the extracted features. If the seasonal component exists, it trains a Seasonal ARIMA or SARIMA model, otherwise, if the trend exists, it trains an XGBoost model, and finally, if none of them exist, it trains a Prophet model.

The last section of the code defines a function to calculate MAPE between the actual and predicted values.

The output is :

![Screenshot (5)](https://user-images.githubusercontent.com/76140010/224596679-25a0f2de-129e-4a60-b1c7-e669b7b050f0.png)

![Screenshot (6)](https://user-images.githubusercontent.com/76140010/224596740-1215219c-b5dc-4997-b429-fbfdc75256c9.png)

![Screenshot (7)](https://user-images.githubusercontent.com/76140010/224596770-0ad0fe27-fc59-4779-abd8-0f7479bb665a.png)

![Screenshot (8)](https://user-images.githubusercontent.com/76140010/224596808-570c9e78-1264-48d6-a18a-17fbde4301e7.png)


