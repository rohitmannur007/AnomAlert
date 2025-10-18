import os
import pandas as pd
import numpy as np
import joblib
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA

def train_prophet(ts_df, model_dir="./models"):
    """
    ts_df: DataFrame with index timestamp and a numeric column 'y' (already aggregated anomaly score)
    """
    model_dir = model_dir
    df = ts_df.reset_index().rename(columns={'timestamp':'ds', 'y':'y'})[['ds','y']]
    m = Prophet()
    m.fit(df)
    os.makedirs(model_dir, exist_ok=True)
    m.save(os.path.join(model_dir, "prophet_model"))
    return m

def predict_prophet(m, periods=60):
    future = m.make_future_dataframe(periods=periods, freq='T')
    forecast = m.predict(future)
    return forecast

def train_arima(ts_series, model_dir="./models", order=(5,1,0)):
    m = ARIMA(ts_series, order=order).fit()
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(m, os.path.join(model_dir, "arima_model.joblib"))
    return m

def predict_arima(m, steps=60):
    pred = m.forecast(steps=steps)
    return pred
