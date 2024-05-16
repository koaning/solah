# pip install openmeteo-requests
# pip install requests-cache retry-requests numpy pandas

import openmeteo_requests

import requests_cache
import pandas as pd
from retry_requests import retry
from pathlib import Path

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

# Make sure all required weather variables are listed here
# The order of variables in hourly or daily is important to assign them correctly below
url = "https://api.open-meteo.com/v1/forecast"
params = {
	"latitude": 52.3808,
	"longitude": 4.6368,
	"daily": ["temperature_2m_max", "temperature_2m_min", "sunrise", "sunset", "sunshine_duration", "precipitation_hours", "shortwave_radiation_sum"],
	"past_days": 92,
	"forecast_days": 1
}
responses = openmeteo.weather_api(url, params=params)

# Process first location. Add a for-loop for multiple locations or weather models
response = responses[0]

# Process daily data. The order of variables needs to be the same as requested.
daily = response.Daily()
daily_temperature_2m_max = daily.Variables(0).ValuesAsNumpy()
daily_temperature_2m_min = daily.Variables(1).ValuesAsNumpy()
daily_sunrise = daily.Variables(2).ValuesAsNumpy()
daily_sunset = daily.Variables(3).ValuesAsNumpy()
daily_sunshine_duration = daily.Variables(4).ValuesAsNumpy()
daily_precipitation_hours = daily.Variables(5).ValuesAsNumpy()
daily_shortwave_radiation_sum = daily.Variables(6).ValuesAsNumpy()

daily_data = {"date": pd.date_range(
	start = pd.to_datetime(daily.Time(), unit = "s", utc = True),
	end = pd.to_datetime(daily.TimeEnd(), unit = "s", utc = True),
	freq = pd.Timedelta(seconds = daily.Interval()),
	inclusive = "left"
)}
daily_data["temperature_2m_max"] = daily_temperature_2m_max
daily_data["temperature_2m_min"] = daily_temperature_2m_min
daily_data["sunrise"] = daily_sunrise
daily_data["sunset"] = daily_sunset
daily_data["sunshine_duration"] = daily_sunshine_duration
daily_data["precipitation_hours"] = daily_precipitation_hours
daily_data["shortwave_radiation_sum"] = daily_shortwave_radiation_sum

daily_dataframe = pd.DataFrame(data = daily_data).assign(date=lambda d: d['date'].dt.strftime('%Y-%m-%d'))

# Check if it already exists 
path = Path("data/history.csv")
if not path.exists():
    daily_dataframe.to_csv('data/history.csv', index=False)

# Merge and overwrite
existing_dataframe = pd.read_csv('data/history.csv')
daily_dataframe = pd.concat([existing_dataframe, daily_dataframe]).drop_duplicates(subset=['date'], keep='last')
daily_dataframe.to_csv('data/history.csv', index=False)
