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
# AIM:
# Analyze Weather Data Using OpenWeatherMap API
# and Perform Data Modeling and Visualization.
#
# Retrieve weather forecast data using an API.
# Perform data cleaning, transformation and analysis.
# Generate graphs and study relationships between
# different weather attributes.
# ============================================================


# ============================================================
# 1. INSTALLATION AND EXECUTION COMMANDS
# ============================================================

# Windows:
# python -m pip install requests pandas numpy matplotlib
# python practical_weather.py

# Ubuntu/Linux:
# python3 -m pip install requests pandas numpy matplotlib
# python3 practical_weather.py


# ============================================================
# 2. IMPORTING LIBRARIES
# ============================================================

# import requests
# Imports the Requests library.
# It is used to send HTTP requests to websites and APIs.
# Here, it retrieves weather information from OpenWeatherMap.


# import pandas as pd
# Imports the Pandas library.
# It is used to store, clean and analyze weather data.
# pd is the short name used for Pandas.


# import numpy as np
# Imports the NumPy library.
# It is used for numerical calculations and array operations.
# Here, it helps identify and replace infinite values.


# import matplotlib.pyplot as plt
# Imports the Pyplot module from Matplotlib.
# It is used to create graphs and visualizations.
# plt is the short name used for Pyplot.


# ============================================================
# 3. ENTERING THE OPENWEATHERMAP API KEY
# ============================================================

# API_KEY = "YOUR_NEW_API_KEY"
# Stores the API key obtained from OpenWeatherMap.
#
# An API key is a unique identification code.
# It is used to authenticate requests to the weather API.
#
# Replace YOUR_NEW_API_KEY with your newly generated key.
#
# Security:
# Do not share your API key publicly.
# For better security, store it in an environment variable
# instead of writing it directly in the program.


# ============================================================
# 4. SPECIFYING CITY AND COUNTRY
# ============================================================

# CITY = "Pune"
# Specifies the city for which weather data is required.
# Here, the selected city is Pune.


# COUNTRY = "IN"
# Specifies the country code.
# IN represents India.
#
# The city and country code are combined
# to identify the required location.


# ============================================================
# 5. DEFINING API URL
# ============================================================

# URL = "https://api.openweathermap.org/data/2.5/forecast"
# Stores the URL of the OpenWeatherMap forecast API.
#
# This endpoint provides weather forecast information.
#
# It generally provides forecasts at three-hour intervals
# for up to five days.
#
# The API returns the response in JSON format.


# ============================================================
# 6. DEFINING API PARAMETERS
# ============================================================

# params = {
#     "q": CITY + "," + COUNTRY,
#     "appid": API_KEY,
#     "units": "metric"
# }
#
# Creates a dictionary containing API request parameters.
#
# q:
# Specifies the city and country.
# Example: Pune,IN.
#
# appid:
# Sends the API key for authentication.
#
# units:
# Specifies the measurement system.
#
# metric means:
# Temperature is returned in Celsius.
# Wind speed is returned in metres per second.
#
# These parameters are sent along with the API request.


# ============================================================
# 7. RETRIEVING WEATHER DATA FROM API
# ============================================================

# response = requests.get(URL, params=params, timeout=20)
# Sends an HTTP GET request to the OpenWeatherMap API.
#
# URL:
# Specifies the API endpoint.
#
# params:
# Sends the city, API key and measurement units.
#
# timeout=20:
# Allows a maximum waiting time of 20 seconds.
#
# The server sends a response containing weather information.
# The response is stored in the response variable.


# if response.status_code != 200:
# Checks whether the API request was unsuccessful.
#
# HTTP status code 200 means the request was successful.
#
# If the status code is not 200,
# the program enters the error-handling section.


#     print("Error:", response.status_code)
# Displays the HTTP error status code.
#
# Examples:
# 401 = Unauthorized or invalid API key.
# 404 = Requested location or resource not found.
# 429 = Too many requests or rate limit exceeded.
# 500 = Internal server error.


#     print(response.json())
# Converts the error response into JSON format
# and displays the error details returned by the API.


#     raise SystemExit("Unable to retrieve weather data. Check API key and city.")
# Stops the program if weather data cannot be retrieved.
#
# This prevents the remaining code from running
# without valid weather information.


# ============================================================
# 8. CONVERTING API RESPONSE INTO JSON DATA
# ============================================================

# data = response.json()
# Converts the successful API response into Python data.
#
# The returned JSON information is converted into
# Python dictionaries and lists.
#
# This makes it easier to access weather information.


# print("Weather Data Retrieved Successfully!")
# Displays a success message after retrieving weather data.


# print("\nCity:", data["city"]["name"])
# Accesses the city information from the API response.
#
# data["city"]:
# Contains geographical and city-related information.
#
# ["name"]:
# Retrieves the city name.
#
# Displays the name of the selected city.


# ============================================================
# 9. EXTRACTING GEOGRAPHICAL INFORMATION
# ============================================================

# latitude = data["city"]["coord"]["lat"]
# Extracts the latitude of the selected city.
#
# Latitude represents the north-south geographical position.
# It is measured in degrees.


# longitude = data["city"]["coord"]["lon"]
# Extracts the longitude of the selected city.
#
# Longitude represents the east-west geographical position.
# It is measured in degrees.


# print("\nLatitude:", latitude)
# Displays the latitude value.


# print("Longitude:", longitude)
# Displays the longitude value.


# ============================================================
# 10. EXTRACTING WEATHER ATTRIBUTES
# ============================================================

# weather_records = []
# Creates an empty Python list.
#
# It will store the extracted weather information.
# Each forecast record will be stored as a dictionary.


# for item in data["list"]:
# Starts a loop through all forecast records.
#
# data["list"] contains the forecast entries returned by the API.
#
# The standard forecast API generally provides
# weather information at three-hour intervals.


#     weather_records.append({
#         "Date": item["dt_txt"],
#         "Temperature": item["main"].get("temp"),
#         "Humidity": item["main"].get("humidity"),
#         "Wind_Speed": item["wind"].get("speed"),
#         "Rainfall": item.get("rain", {}).get("3h", 0),
#         "Weather": item["weather"][0]["description"]
#     })
#
# Extracts the required weather information
# from each forecast record.
#
# Date:
# Extracts the forecast date and time.
# dt_txt contains the date and time in text format.
#
# Temperature:
# Extracts the forecast temperature in Celsius.
#
# Humidity:
# Extracts the relative humidity percentage.
#
# Wind_Speed:
# Extracts wind speed in metres per second.
#
# Rainfall:
# Extracts the rainfall amount for a three-hour period.
#
# item.get("rain", {}):
# Checks whether rainfall information is available.
# If the rain field is missing, it uses an empty dictionary.
#
# .get("3h", 0):
# Retrieves the rainfall amount for three hours.
# If it is unavailable, the value is set to zero.
#
# Weather:
# Extracts the description of weather conditions.
#
# Example:
# Clear sky, light rain or overcast clouds.
#
# append():
# Adds each extracted record to the weather_records list.


# ============================================================
# 11. CONVERTING RECORDS INTO A DATAFRAME
# ============================================================

# df = pd.DataFrame(weather_records)
# Converts the list of dictionaries into a Pandas DataFrame.
#
# Each dictionary becomes one row.
# Dictionary keys become column names.
#
# The resulting DataFrame contains:
# Date.
# Temperature.
# Humidity.
# Wind_Speed.
# Rainfall.
# Weather.


# print("\nWeather Dataset:")
# Displays the heading for the weather dataset.


# print(df.head())
# Displays the first five records of the DataFrame.
#
# head() is used to view the first five rows by default.
# It helps us understand the structure of the dataset.


# ============================================================
# 12. EXPLORING THE DATASET
# ============================================================

# print("\nDataset Information:")
# Displays the dataset information heading.


# df.info()
# Displays information about the DataFrame.
#
# It shows:
# Number of rows.
# Column names.
# Number of non-null values.
# Data types.
# Memory usage.


# print("\nMissing Values:")
# Displays the heading for missing-value information.


# print(df.isnull().sum())
# isnull() identifies missing values.
#
# sum() counts the missing values in each column.
#
# This helps identify incomplete weather records.


# ============================================================
# 13. DATA CLEANING - CONVERTING DATE
# ============================================================

# df["Date"] = pd.to_datetime(
#     df["Date"],
#     errors="coerce"
# )
#
# Converts the Date column into Pandas datetime format.
#
# pd.to_datetime():
# Converts text dates into date and time values.
#
# errors="coerce":
# Converts invalid date values into NaT.
#
# NaT means Not a Time.
#
# Datetime format allows us to extract
# day, month and hour information.


# ============================================================
# 14. IDENTIFYING NUMERIC COLUMNS
# ============================================================

# numeric_columns = [
#     "Temperature",
#     "Humidity",
#     "Wind_Speed",
#     "Rainfall"
# ]
#
# Creates a list of columns containing numerical weather data.
#
# These columns are used for:
# Data cleaning.
# Statistical analysis.
# Correlation analysis.
# Visualization.


# ============================================================
# 15. CONVERTING NUMERIC COLUMNS
# ============================================================

# for column in numeric_columns:
# Starts a loop through all numerical weather columns.


#     df[column] = pd.to_numeric(
#         df[column],
#         errors="coerce"
#     )
#
# Converts each selected column into numeric format.
#
# pd.to_numeric():
# Converts values into numbers.
#
# errors="coerce":
# Converts invalid values into NaN.
#
# NaN represents a missing numerical value.


# ============================================================
# 16. HANDLING INFINITE VALUES
# ============================================================

# df[numeric_columns] = df[numeric_columns].replace(
#     [np.inf, -np.inf],
#     np.nan
# )
#
# Checks the numerical columns for infinite values.
#
# np.inf:
# Represents positive infinity.
#
# -np.inf:
# Represents negative infinity.
#
# replace():
# Replaces both types of infinite values with NaN.
#
# This helps make the dataset suitable
# for statistical calculations and analysis.


# ============================================================
# 17. REMOVING IMPORTANT MISSING RECORDS
# ============================================================

# df = df.dropna(
#     subset=["Date", "Temperature", "Humidity", "Wind_Speed"]
# )
#
# Removes rows containing missing values
# in the specified important columns.
#
# subset:
# Specifies the columns to check.
#
# A record is removed if its date, temperature,
# humidity or wind speed is missing.
#
# Rainfall is not included because it is handled separately.


# ============================================================
# 18. FILLING MISSING RAINFALL VALUES
# ============================================================

# df["Rainfall"] = df["Rainfall"].fillna(0)
# Replaces missing rainfall values with zero.
#
# fillna() fills missing values.
#
# This program assumes missing rainfall values
# can be treated as zero for the analysis.
#
# Note:
# A missing rainfall field in the API is commonly treated
# as no reported rain, but missing data is not always
# proof that rainfall was exactly zero.


# ============================================================
# 19. REMOVING DUPLICATE RECORDS
# ============================================================

# df = df.drop_duplicates()
# Removes repeated records from the DataFrame.
#
# This helps prevent duplicate forecast records
# from affecting statistical calculations.


# ============================================================
# 20. DATA TRANSFORMATION - EXTRACTING DAY
# ============================================================

# df["Day"] = df["Date"].dt.date
# Extracts the calendar date from the Date column.
#
# .dt accesses datetime-related properties.
#
# .date extracts only the date without the time.
#
# Example:
# 2026-01-05 09:00:00 becomes 2026-01-05.
#
# The result is stored in a new column called Day.


# ============================================================
# 21. EXTRACTING MONTH
# ============================================================

# df["Month"] = df["Date"].dt.month_name()
# Extracts the full month name from the date.
#
# Example:
# 2026-01-05 becomes January.
#
# The result is stored in a new column called Month.


# ============================================================
# 22. EXTRACTING HOUR
# ============================================================

# df["Hour"] = df["Date"].dt.hour
# Extracts the hour from the forecast date and time.
#
# Example:
# 2026-01-05 15:00:00 gives 15.
#
# The result is stored in a new column called Hour.


# ============================================================
# 23. DISPLAYING THE CLEANED DATASET
# ============================================================

# print("\nCleaned Weather Dataset:")
# Displays the heading for the cleaned weather data.


# print(df.head())
# Displays the first five rows after cleaning
# and transformation.


# ============================================================
# 24. WEATHER DATA STATISTICAL ANALYSIS
# ============================================================

# print("\nDescriptive Statistics:")
# Displays the statistical analysis heading.


# print(df[numeric_columns].describe())
# Calculates descriptive statistics for numerical columns.
#
# count:
# Number of available records.
#
# mean:
# Average value.
#
# std:
# Standard deviation, which measures data spread.
#
# min:
# Minimum value.
#
# 25%:
# First quartile.
#
# 50%:
# Median or middle value.
#
# 75%:
# Third quartile.
#
# max:
# Maximum value.


# ============================================================
# 25. CALCULATING AVERAGE TEMPERATURE
# ============================================================

# print("\nAverage Temperature:",
#       df["Temperature"].mean(), "°C")
#
# mean() calculates the average temperature.
#
# Average Temperature =
# Sum of temperatures / Number of valid temperature records.
#
# The result is displayed in degrees Celsius.


# ============================================================
# 26. CALCULATING MAXIMUM TEMPERATURE
# ============================================================

# print("Maximum Temperature:",
#       df["Temperature"].max(), "°C")
#
# max() finds the highest temperature
# in the forecast dataset.


# ============================================================
# 27. CALCULATING MINIMUM TEMPERATURE
# ============================================================

# print("Minimum Temperature:",
#       df["Temperature"].min(), "°C")
#
# min() finds the lowest temperature
# in the forecast dataset.


# ============================================================
# 28. CALCULATING AVERAGE HUMIDITY
# ============================================================

# print("Average Humidity:",
#       df["Humidity"].mean(), "%")
#
# Calculates the average relative humidity.
#
# Humidity represents the amount of water vapour
# present in the air relative to the maximum amount
# the air can hold at that temperature.


# ============================================================
# 29. CALCULATING AVERAGE WIND SPEED
# ============================================================

# print("Average Wind Speed:",
#       df["Wind_Speed"].mean(), "m/s")
#
# Calculates the average wind speed.
#
# m/s means metres per second.


# ============================================================
# 30. CALCULATING TOTAL FORECAST RAINFALL
# ============================================================

# print("Total Forecast Rainfall:",
#       df["Rainfall"].sum(), "mm")
#
# sum() adds all rainfall values in the dataset.
#
# The rainfall values are generally measured
# for individual three-hour forecast periods.
#
# The result represents the sum of the forecast
# three-hour rainfall amounts in the retrieved records.
#
# It is not the rainfall amount for one particular day.


# ============================================================
# 31. DAILY WEATHER AGGREGATION
# ============================================================

# daily_weather = df.groupby("Day").agg({
#     "Temperature": "mean",
#     "Humidity": "mean",
#     "Wind_Speed": "mean",
#     "Rainfall": "sum"
# })
#
# Groups weather records according to the calendar day.
#
# groupby("Day"):
# Creates a separate group for each day.
#
# agg():
# Performs different calculations on different columns.
#
# Temperature: mean
# Calculates the average daily temperature.
#
# Humidity: mean
# Calculates the average daily humidity.
#
# Wind_Speed: mean
# Calculates the average daily wind speed.
#
# Rainfall: sum
# Calculates the total forecast rainfall for each day.
#
# The result is stored in daily_weather.


# ============================================================
# 32. DISPLAYING DAILY WEATHER SUMMARY
# ============================================================

# print("\nDaily Weather Summary:")
# Displays the daily weather summary heading.


# print(daily_weather)
# Displays the aggregated weather information
# for each forecast day.


# ============================================================
# 33. SAVING WEATHER DATA
# ============================================================

# df.to_csv("weather_data.csv", index=False)
# Saves the cleaned weather dataset into a CSV file.
#
# weather_data.csv:
# Contains the complete cleaned forecast records.
#
# index=False:
# Prevents the DataFrame index from being saved
# as an additional column.


# daily_weather.to_csv("daily_weather_summary.csv")
# Saves the daily aggregated weather information
# into another CSV file.
#
# The daily DataFrame index is included by default.


# print("\nWeather data saved successfully!")
# Displays a confirmation message after saving the files.


# ============================================================
# 34. TEMPERATURE LINE CHART
# ============================================================

# plt.figure(figsize=(10, 5))
# Creates a new graph.
#
# figsize=(10, 5):
# Sets the graph width to 10 inches
# and height to 5 inches.


# plt.plot(
#     df["Date"],
#     df["Temperature"],
#     marker="o"
# )
#
# Creates a line graph.
#
# X-axis:
# Date and time.
#
# Y-axis:
# Temperature.
#
# marker="o":
# Displays circular markers at the data points.
#
# This graph shows how forecast temperature
# changes over time.


# plt.title("Temperature Changes Over Time")
# Sets the graph title.


# plt.xlabel("Date and Time")
# Sets the X-axis label.


# plt.ylabel("Temperature (°C)")
# Sets the Y-axis label.


# plt.xticks(rotation=45)
# Rotates date and time labels by 45 degrees.
#
# This improves readability and prevents label overlap.


# plt.tight_layout()
# Automatically adjusts graph spacing.


# plt.show()
# Displays the temperature line chart.


# ============================================================
# 35. HUMIDITY BAR CHART
# ============================================================

# plt.figure(figsize=(10, 5))
# Creates a new graph with width 10 inches
# and height 5 inches.


# plt.bar(
#     df["Date"].dt.strftime("%d-%m %H:%M"),
#     df["Humidity"]
# )
#
# Creates a bar chart for humidity.
#
# X-axis:
# Date and time converted into readable text.
#
# strftime("%d-%m %H:%M"):
# Formats the date and time.
#
# %d = Day.
# %m = Month.
# %H = Hour.
# %M = Minute.
#
# Y-axis:
# Humidity percentage.
#
# Each bar represents the humidity at a forecast time.


# plt.title("Humidity Variation")
# Sets the chart title.


# plt.xlabel("Date and Time")
# Sets the X-axis label.


# plt.ylabel("Humidity (%)")
# Sets the Y-axis label.


# plt.xticks(rotation=90)
# Rotates the date and time labels by 90 degrees.


# plt.tight_layout()
# Adjusts the spacing of the graph.


# plt.show()
# Displays the humidity bar chart.


# ============================================================
# 36. RAINFALL VISUALIZATION
# ============================================================

# plt.figure(figsize=(9, 5))
# Creates a new graph with width 9 inches
# and height 5 inches.


# plt.bar(
#     df["Date"].dt.strftime("%d-%m %H:%M"),
#     df["Rainfall"]
# )
#
# Creates a bar chart for forecast rainfall.
#
# X-axis:
# Date and time.
#
# Y-axis:
# Rainfall amount in millimetres.
#
# Each bar represents the rainfall amount
# for the corresponding three-hour forecast period.


# plt.title("Rainfall Forecast")
# Sets the graph title.


# plt.xlabel("Date and Time")
# Sets the X-axis label.


# plt.ylabel("Rainfall (mm in 3 hours)")
# Sets the Y-axis label.
#
# mm means millimetres.


# plt.xticks(rotation=90)
# Rotates the date and time labels vertically.


# plt.tight_layout()
# Adjusts graph spacing.


# plt.show()
# Displays the rainfall visualization.


# ============================================================
# 37. WIND SPEED VISUALIZATION
# ============================================================

# plt.figure(figsize=(9, 5))
# Creates a new graph with width 9 inches
# and height 5 inches.


# plt.plot(
#     df["Date"],
#     df["Wind_Speed"],
#     marker="o"
# )
#
# Creates a line chart for wind speed.
#
# X-axis:
# Date and time.
#
# Y-axis:
# Wind speed in metres per second.
#
# Circular markers represent individual forecast values.


# plt.title("Wind Speed Variation")
# Sets the chart title.


# plt.xlabel("Date and Time")
# Sets the X-axis label.


# plt.ylabel("Wind Speed (m/s)")
# Sets the Y-axis label.


# plt.xticks(rotation=45)
# Rotates the date labels by 45 degrees.


# plt.tight_layout()
# Adjusts the graph layout.


# plt.show()
# Displays the wind speed line chart.


# ============================================================
# 38. CORRELATION ANALYSIS
# ============================================================

# correlation = df[numeric_columns].corr()
# Calculates the correlation between numerical weather attributes.
#
# corr() calculates the Pearson correlation coefficient
# by default.
#
# It measures the strength and direction
# of the linear relationship between two numerical variables.
#
# Correlation values range from -1 to +1.
#
# +1:
# Perfect positive correlation.
#
# 0:
# No linear correlation.
#
# -1:
# Perfect negative correlation.
#
# The result is stored in correlation.


# print("\nWeather Attributes Correlation:")
# Displays the correlation analysis heading.


# print(correlation)
# Displays the correlation matrix.
#
# It shows the correlation between:
# Temperature.
# Humidity.
# Wind speed.
# Rainfall.


# ============================================================
# 39. CREATING CORRELATION HEATMAP
# ============================================================

# plt.figure(figsize=(7, 5))
# Creates a new graph with width 7 inches
# and height 5 inches.


# plt.imshow(
#     correlation,
#     cmap="coolwarm",
#     vmin=-1,
#     vmax=1
# )
#
# Displays the correlation matrix as a heatmap.
#
# imshow():
# Displays numerical values as colored cells.
#
# cmap="coolwarm":
# Uses a color map that represents negative
# and positive correlation with different colors.
#
# vmin=-1:
# Sets the minimum color scale value to -1.
#
# vmax=1:
# Sets the maximum color scale value to +1.


# plt.colorbar()
# Displays a color scale beside the heatmap.
# It helps understand the meaning of the colors.


# plt.xticks(
#     range(len(numeric_columns)),
#     numeric_columns,
#     rotation=45
# )
#
# Sets the X-axis tick positions and labels.
#
# Displays the numerical column names.
# Rotates the labels by 45 degrees.


# plt.yticks(
#     range(len(numeric_columns)),
#     numeric_columns
# )
#
# Sets the Y-axis tick positions and labels.
# Displays the same numerical column names.


# ============================================================
# 40. DISPLAYING CORRELATION VALUES INSIDE HEATMAP
# ============================================================

# for i in range(len(numeric_columns)):
#     for j in range(len(numeric_columns)):
#
# Starts two nested loops.
#
# The outer loop selects each row.
# The inner loop selects each column.
#
# This visits every cell of the correlation matrix.


#         plt.text(
#             j,
#             i,
#             round(correlation.iloc[i, j], 2),
#             ha="center",
#             va="center"
#         )
#
# Displays the numerical correlation value inside each heatmap cell.
#
# j:
# Horizontal position of the text.
#
# i:
# Vertical position of the text.
#
# correlation.iloc[i, j]:
# Retrieves the correlation value at row i and column j.
#
# round(..., 2):
# Rounds the value to two decimal places.
#
# ha="center":
# Aligns the text horizontally in the center.
#
# va="center":
# Aligns the text vertically in the center.


# plt.title("Weather Attributes Correlation Heatmap")
# Sets the heatmap title.


# plt.tight_layout()
# Adjusts the heatmap layout.


# plt.show()
# Displays the correlation heatmap.


# ============================================================
# 41. SCATTER PLOT - TEMPERATURE AND HUMIDITY
# ============================================================

# plt.figure(figsize=(7, 5))
# Creates a new graph with width 7 inches
# and height 5 inches.


# plt.scatter(
#     df["Temperature"],
#     df["Humidity"]
# )
#
# Creates a scatter plot.
#
# X-axis:
# Temperature in degrees Celsius.
#
# Y-axis:
# Humidity in percentage.
#
# Each point represents one forecast record.
#
# It helps study the relationship
# between temperature and humidity.


# plt.title("Temperature vs Humidity")
# Sets the graph title.


# plt.xlabel("Temperature (°C)")
# Sets the X-axis label.


# plt.ylabel("Humidity (%)")
# Sets the Y-axis label.


# plt.tight_layout()
# Adjusts the graph layout.


# plt.show()
# Displays the scatter plot.


# ============================================================
# 42. GEOGRAPHICAL VISUALIZATION
# ============================================================

# plt.figure(figsize=(6, 5))
# Creates a new graph with width 6 inches
# and height 5 inches.


# plt.scatter(
#     longitude,
#     latitude,
#     s=200,
#     marker="o"
# )
#
# Creates a scatter plot using geographical coordinates.
#
# X-axis:
# Longitude.
#
# Y-axis:
# Latitude.
#
# s=200:
# Sets the marker size to 200.
#
# marker="o":
# Uses a circular marker.
#
# It represents the geographical location
# of the selected city.


# plt.text(
#     longitude,
#     latitude,
#     CITY
# )
#
# Displays the city name near the plotted point.
#
# longitude:
# Horizontal coordinate.
#
# latitude:
# Vertical coordinate.
#
# CITY:
# Displays the selected city name.


# plt.title("Weather Location Map")
# Sets the graph title.


# plt.xlabel("Longitude")
# Sets the X-axis label.


# plt.ylabel("Latitude")
# Sets the Y-axis label.


# plt.grid()
# Displays grid lines on the graph.
# This helps read coordinate positions.


# plt.tight_layout()
# Adjusts the graph layout.


# plt.show()
# Displays the geographical scatter plot.
#
# Important:
# This is a coordinate plot, not an actual geographical map.
# It shows only one location point.


# ============================================================
# 43. DISPLAYING COMPLETION MESSAGE
# ============================================================

# print("\nWeather Data Analysis Completed Successfully!")
# Displays the final completion message.
#
# It indicates that data retrieval, cleaning,
# analysis and visualization have been completed.


# ============================================================
# 44. IMPORTANT VIVA QUESTIONS
# ============================================================

# Q1. What is an API?
# API stands for Application Programming Interface.
# It allows two software applications to communicate
# and exchange information.

# Q2. What is OpenWeatherMap?
# OpenWeatherMap is an online weather service
# that provides current weather and forecast information
# through APIs.

# Q3. What is an API key?
# An API key is a unique code used to authenticate
# requests made to an API.

# Q4. What is the purpose of the Requests library?
# It is used to send HTTP requests
# and receive responses from web services.

# Q5. What is an HTTP GET request?
# A GET request is used to retrieve information
# from a server.

# Q6. What is JSON?
# JSON stands for JavaScript Object Notation.
# It is a format used to store and exchange structured data.

# Q7. What does status_code 200 mean?
# It means that the HTTP request was successful.

# Q8. What does status_code 401 mean?
# It generally indicates an authentication problem,
# such as an invalid or unauthorized API key.

# Q9. What does status_code 404 mean?
# It indicates that the requested resource was not found.

# Q10. What does status_code 429 mean?
# It indicates that too many requests were made
# or an API rate limit was exceeded.

# Q11. What is a DataFrame?
# A DataFrame is a two-dimensional table
# containing rows and columns.

# Q12. What is pd.DataFrame()?
# It converts data into a Pandas DataFrame.

# Q13. What is pd.to_datetime()?
# It converts values into date and time format.

# Q14. What is pd.to_numeric()?
# It converts values into numerical format.

# Q15. What is NaN?
# NaN means Not a Number.
# It represents missing or invalid numerical data.

# Q16. What is NaT?
# NaT means Not a Time.
# It represents a missing or invalid date-time value.

# Q17. What is fillna()?
# It replaces missing values with a specified value.

# Q18. What is dropna()?
# It removes rows or columns containing missing values.

# Q19. What is drop_duplicates()?
# It removes repeated records from a DataFrame.

# Q20. What is groupby()?
# It groups records according to a specified column.

# Q21. What is aggregation?
# Aggregation means calculating summary values
# from a group of records.
# Examples include sum, mean, minimum and maximum.

# Q22. What is describe()?
# It calculates descriptive statistics such as
# count, mean, standard deviation, minimum,
# maximum and quartiles.

# Q23. What is correlation?
# Correlation measures the strength and direction
# of a relationship between two numerical variables.

# Q24. What is the range of correlation?
# Correlation ranges from -1 to +1.

# Q25. What is positive correlation?
# When one variable increases, the other variable
# tends to increase.

# Q26. What is negative correlation?
# When one variable increases, the other variable
# tends to decrease.

# Q27. What is a heatmap?
# A heatmap represents numerical information
# using different colors.

# Q28. What is a scatter plot?
# A scatter plot displays individual data points
# to study the relationship between two numerical variables.

# Q29. What is a line chart?
# A line chart connects data points using lines.
# It is useful for showing changes over time.

# Q30. What is a bar chart?
# A bar chart uses rectangular bars
# to compare different values.

# Q31. What is latitude?
# Latitude represents the north-south geographical position
# of a location.

# Q32. What is longitude?
# Longitude represents the east-west geographical position
# of a location.

# Q33. What is rainfall measured in?
# Rainfall is generally measured in millimetres.

# Q34. What is humidity?
# Humidity is the amount of water vapour present in the air
# relative to the maximum amount the air can hold
# at a particular temperature.

# Q35. What is wind speed measured in?
# In this program, wind speed is measured
# in metres per second.

# Q36. What is the purpose of timeout=20?
# It limits how long the program waits
# for the API response before timing out.

# Q37. What is the purpose of errors="coerce"?
# It converts invalid values into missing values
# instead of stopping the program with a conversion error.

# Q38. What is the purpose of np.inf?
# It represents positive infinity in NumPy.

# Q39. What is the purpose of np.nan?
# It represents a missing numerical value.

# Q40. What is the purpose of to_csv()?
# It saves a DataFrame into a CSV file.

# Q41. What is the difference between current weather
# and forecast weather?
# Current weather describes the weather at the present time.
# Forecast weather provides predicted weather conditions
# for future time periods.

# Q42. What is the OpenWeatherMap forecast API?
# It is an API that provides forecast weather information
# for a selected location.

# Q43. What are the applications of weather data analysis?
# Weather forecasting.
# Agriculture.
# Disaster management.
# Transportation.
# Travel planning.
# Environmental monitoring.


# ============================================================
# 45. CONCLUSION
# ============================================================

# In this practical, weather data is retrieved
# from the OpenWeatherMap API for Pune, India.

# The API response is converted into a Pandas DataFrame.

# Data cleaning is performed by converting date and numerical
# columns, handling missing values and removing duplicates.

# Data transformation is performed by extracting
# day, month and hour from the date.

# Statistical analysis is performed to calculate
# average, maximum and minimum temperature,
# average humidity, wind speed and total forecast rainfall.

# Daily aggregation is performed using groupby() and agg().

# Finally, different visualizations are created:
# Temperature line chart.
# Humidity bar chart.
# Rainfall bar chart.
# Wind speed line chart.
# Correlation heatmap.
# Temperature versus humidity scatter plot.
# Geographical coordinate plot.

# The cleaned weather data and daily summary
# are saved into CSV files.

# Therefore, this practical demonstrates how API data
# can be retrieved, cleaned, analyzed and visualized
# using Python libraries.
