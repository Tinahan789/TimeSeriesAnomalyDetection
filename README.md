# Time Series Anomaly Detection
This is a repository dedicated to sharing easy to use anomaly detection methods for time series data and model outputs.
*Note that Tensorflow (required to run the LSTM based method) only works with Python 3.9–3.12.*
## Contents
- **arima_anomaly_detection.ipynb**: Simple example of using an ARIMA model for anomaly detection in simulated model outputs.
- **ensemble_anomaly_detection.ipynb**: Example of using ensembled time series models for anomaly detection with google trends data.
- **lstm_autoencoder.ipynb**: Example using an LSTM autoencoder for anomaly detection.
- **google_trends_redfin.csv**: Data used for ensemble_anomaly_detection.ipynb
- **pmms_data.csv**: Data used for arima_anomaly_detection.ipynb
- **requirements.txt**: requirements file
## Prerequisite knowledge
### Time series data
A time series $Y = \{y_1, y_2, ..., y_n\}$ is a set of observations $y_t$ recorded at time $t$. They can be univariate or multivariate, discrete or continuous. Time series can be used in regression (e.g. forecasting stock prices) or classification (e.g. identifying a cardiac event). This repository focuses on using forecasting models for anomaly detection in the discrete univariate case.
### Anomaly detection
Anomaly is sometimes used interchangably with outliers, and frequently the same methods can be used to find them. However, they are distinct concepts. Outliers are data points that significantly deviate from the majority of the data set, frequently the use case in identifying outliers is for data cleaning or use them to explain a statistic. Anomalies are events in the data that does not fit the expected behavior which help identify significant events like fraud or a heart attack. Anomaly detection is the automated identification of anomalies frequently utilizing machine learning methods.

In this image it is easy to visually inspect the plot and find an obvious anomaly in February 2026. Not all anomalies are this easy to spot.
![anomaly example](images/anomaly.png)
### Prediction intervals
Many of the techniques included in this repository rely on the use of prediction intervals. The prediction interval is a range of values that is likely to contian an individual value.

Note that in some places we use the term confidence interval, which is a similar concept, but technically incorrect and I'm too lazy to go change it.

In this image, the prediction interval is indicated by the grey area around the predicted values and true values.
![prediction interval example](images/prediction_interval.png)
## Time series models and python libraries used in this repository
Many models assume the data is stationary. In this repository we have ignored these standard checks.
### ARIMA
https://pypi.org/project/pmdarima/
### Exponential smoothing
https://www.statsmodels.org/stable/generated/statsmodels.tsa.holtwinters.ExponentialSmoothing.html
### Prophet
https://facebook.github.io/prophet/
### LSTM
https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM
