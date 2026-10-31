# AIM: Analyze Weather Data Using OpenWeatherMap API
# and Perform Data Modeling and Visualization.

# Installation commands:
# Windows:
# python -m pip install requests pandas numpy matplotlib
# python practical_weather_api.py
#
# Ubuntu/Linux:
# pip install requests pandas numpy matplotlib
# python3 -m pip install requests pandas numpy matplotlib
# python3 practical_weather_api.py


import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Enter your OpenWeatherMap API key

API_KEY = "343da8727c19266740f1fc90388fc343"

# Specify city

CITY = "Pune"
COUNTRY = "IN"

# API URL

URL = "https://api.openweathermap.org/data/2.5/forecast"

# API parameters

params = {
    "q": CITY + "," + COUNTRY,
    "appid": API_KEY,
    "units": "metric"
}

# 1. Retrieve weather data from API

response = requests.get(URL, params=params, timeout=20)

if response.status_code != 200:
    print("Error:", response.status_code)
    print(response.json())
    raise SystemExit("Unable to retrieve weather data. Check API key and city.")

data = response.json()

print("Weather Data Retrieved Successfully!")

print("\nCity:", data["city"]["name"])

# 2. Extract geographical information

latitude = data["city"]["coord"]["lat"]

longitude = data["city"]["coord"]["lon"]

print("\nLatitude:", latitude)

print("Longitude:", longitude)

# 3. Extract weather attributes

weather_records = []

for item in data["list"]:

    weather_records.append({
        "Date": item["dt_txt"],
        "Temperature": item["main"].get("temp"),
        "Humidity": item["main"].get("humidity"),
        "Wind_Speed": item["wind"].get("speed"),
        "Rainfall": item.get("rain", {}).get("3h", 0),
        "Weather": item["weather"][0]["description"]
    })

# Convert records into DataFrame

df = pd.DataFrame(weather_records)

print("\nWeather Dataset:")

print(df.head())

# 4. Explore dataset

print("\nDataset Information:")

df.info()

print("\nMissing Values:")

print(df.isnull().sum())

# 5. Data cleaning

df["Date"] = pd.to_datetime(
    df["Date"],
    errors="coerce"
)

numeric_columns = [
    "Temperature",
    "Humidity",
    "Wind_Speed",
    "Rainfall"
]

for column in numeric_columns:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

df[numeric_columns] = df[numeric_columns].replace(
    [np.inf, -np.inf],
    np.nan
)

df = df.dropna(
    subset=["Date", "Temperature", "Humidity", "Wind_Speed"]
)

df["Rainfall"] = df["Rainfall"].fillna(0)

df = df.drop_duplicates()

# 6. Data transformation

df["Day"] = df["Date"].dt.date

df["Month"] = df["Date"].dt.month_name()

df["Hour"] = df["Date"].dt.hour

print("\nCleaned Weather Dataset:")

print(df.head())

# 7. Weather data analysis

print("\nDescriptive Statistics:")

print(df[numeric_columns].describe())

print("\nAverage Temperature:",
      df["Temperature"].mean(), "°C")

print("Maximum Temperature:",
      df["Temperature"].max(), "°C")

print("Minimum Temperature:",
      df["Temperature"].min(), "°C")

print("Average Humidity:",
      df["Humidity"].mean(), "%")

print("Average Wind Speed:",
      df["Wind_Speed"].mean(), "m/s")

print("Total Forecast Rainfall:",
      df["Rainfall"].sum(), "mm")

# 8. Daily aggregation

daily_weather = df.groupby("Day").agg({
    "Temperature": "mean",
    "Humidity": "mean",
    "Wind_Speed": "mean",
    "Rainfall": "sum"
})

print("\nDaily Weather Summary:")

print(daily_weather)

# 9. Save weather data

df.to_csv("weather_data.csv", index=False)

daily_weather.to_csv("daily_weather_summary.csv")

print("\nWeather data saved successfully!")

# 10. Temperature line chart

plt.figure(figsize=(10, 5))

plt.plot(
    df["Date"],
    df["Temperature"],
    marker="o"
)

plt.title("Temperature Changes Over Time")

plt.xlabel("Date and Time")

plt.ylabel("Temperature (°C)")

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()

# 11. Humidity bar chart

plt.figure(figsize=(10, 5))

plt.bar(
    df["Date"].dt.strftime("%d-%m %H:%M"),
    df["Humidity"]
)

plt.title("Humidity Variation")

plt.xlabel("Date and Time")

plt.ylabel("Humidity (%)")

plt.xticks(rotation=90)

plt.tight_layout()

plt.show()

# 12. Rainfall visualization

plt.figure(figsize=(9, 5))

plt.bar(
    df["Date"].dt.strftime("%d-%m %H:%M"),
    df["Rainfall"]
)

plt.title("Rainfall Forecast")

plt.xlabel("Date and Time")

plt.ylabel("Rainfall (mm in 3 hours)")

plt.xticks(rotation=90)

plt.tight_layout()

plt.show()

# 13. Wind speed visualization

plt.figure(figsize=(9, 5))

plt.plot(
    df["Date"],
    df["Wind_Speed"],
    marker="o"
)

plt.title("Wind Speed Variation")

plt.xlabel("Date and Time")

plt.ylabel("Wind Speed (m/s)")

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()

# 14. Correlation analysis

correlation = df[numeric_columns].corr()

print("\nWeather Attributes Correlation:")

print(correlation)

# Correlation heatmap

plt.figure(figsize=(7, 5))

plt.imshow(
    correlation,
    cmap="coolwarm",
    vmin=-1,
    vmax=1
)

plt.colorbar()

plt.xticks(
    range(len(numeric_columns)),
    numeric_columns,
    rotation=45
)

plt.yticks(
    range(len(numeric_columns)),
    numeric_columns
)

for i in range(len(numeric_columns)):
    for j in range(len(numeric_columns)):
        plt.text(
            j,
            i,
            round(correlation.iloc[i, j], 2),
            ha="center",
            va="center"
        )

plt.title("Weather Attributes Correlation Heatmap")

plt.tight_layout()

plt.show()

# 15. Scatter plot - Temperature and Humidity

plt.figure(figsize=(7, 5))

plt.scatter(
    df["Temperature"],
    df["Humidity"]
)

plt.title("Temperature vs Humidity")

plt.xlabel("Temperature (°C)")

plt.ylabel("Humidity (%)")

plt.tight_layout()

plt.show()

# 16. Geographical visualization

plt.figure(figsize=(6, 5))

plt.scatter(
    longitude,
    latitude,
    s=200,
    marker="o"
)

plt.text(
    longitude,
    latitude,
    CITY
)

plt.title("Weather Location Map")

plt.xlabel("Longitude")

plt.ylabel("Latitude")

plt.grid()

plt.tight_layout()

plt.show()

print("\nWeather Data Analysis Completed Successfully!")


# ============================================================
# EXPLANATION LINE BY LINE
# ============================================================

# import requests
# Imports the Requests library to communicate with the API.

# import pandas as pd
# Imports Pandas to create and process DataFrames.

# import numpy as np
# Imports NumPy for numerical calculations.

# import matplotlib.pyplot as plt
# Imports Matplotlib to create visualizations.

# API_KEY
# Stores the API key obtained from OpenWeatherMap.

# CITY
# Specifies the city whose weather data is required.

# COUNTRY
# Specifies the country code.

# URL
# Stores the OpenWeatherMap forecast API endpoint.

# params
# Stores the city, API key and measurement units.

# requests.get()
# Sends a GET request to the weather API.

# response.status_code
# Checks whether the API request was successful.

# response.json()
# Converts the API response into a Python dictionary.

# data["city"]["name"]
# Retrieves the city name from the API response.

# data["city"]["coord"]["lat"]
# Retrieves the latitude of the city.

# data["city"]["coord"]["lon"]
# Retrieves the longitude of the city.

# weather_records
# Creates an empty list to store weather records.

# for item in data["list"]
# Iterates through the forecast records.

# item["dt_txt"]
# Retrieves the forecast date and time.

# item["main"].get("temp")
# Retrieves the temperature in Celsius.

# item["main"].get("humidity")
# Retrieves the humidity percentage.

# item["wind"].get("speed")
# Retrieves the wind speed in metres per second.

# item.get("rain", {}).get("3h", 0)
# Retrieves rainfall for three hours.
# If rainfall information is unavailable, it uses zero.

# item["weather"][0]["description"]
# Retrieves the weather description.

# pd.DataFrame()
# Converts weather records into a DataFrame.

# df.head()
# Displays the first five weather records.

# df.info()
# Displays dataset structure and data types.

# df.isnull().sum()
# Counts missing values in each column.

# pd.to_datetime()
# Converts the date column into datetime format.

# pd.to_numeric()
# Converts weather attributes into numeric values.

# replace()
# Replaces infinite values with missing values.

# dropna()
# Removes records with missing important values.

# fillna()
# Replaces missing rainfall values with zero.

# drop_duplicates()
# Removes duplicate records.

# dt.date
# Extracts the date from the datetime column.

# dt.month_name()
# Extracts the month name.

# dt.hour
# Extracts the hour from the forecast time.

# describe()
# Calculates count, mean, standard deviation, minimum and maximum.

# mean()
# Calculates the average weather value.

# max()
# Finds the maximum weather value.

# min()
# Finds the minimum weather value.

# sum()
# Calculates the total forecast rainfall.

# groupby()
# Groups weather records according to date.

# agg()
# Calculates daily average and total weather values.

# to_csv()
# Saves weather data into CSV files.

# plt.figure()
# Creates a new figure for visualization.

# plt.plot()
# Creates a line chart.

# plt.bar()
# Creates a bar chart.

# plt.scatter()
# Creates a scatter plot.

# corr()
# Calculates the correlation between weather attributes.

# plt.imshow()
# Displays the correlation matrix as a heatmap.

# plt.colorbar()
# Displays the heatmap color scale.

# plt.text()
# Displays correlation values inside the heatmap.

# plt.title()
# Sets the graph title.

# plt.xlabel()
# Sets the horizontal axis label.

# plt.ylabel()
# Sets the vertical axis label.

# plt.xticks()
# Sets and rotates horizontal axis labels.

# plt.yticks()
# Sets vertical axis labels.

# plt.grid()
# Displays grid lines.

# plt.tight_layout()
# Adjusts the graph layout.

# plt.show()
# Displays the visualization.