
# AIM: Perform Time Series Data Analysis on Stock Market Data.

# Installation commands:
# Windows:
# python -m pip install pandas numpy matplotlib statsmodels scikit-learn
# python practical_stock_analysis.py
#
# Ubuntu/Linux:
# pip install pandas numpy matplotlib statsmodels scikit-learn
# python3 practical_stock_analysis.py


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error


# 1. Import dataset

df = pd.read_csv("Stock_Prices.csv")

print("First Five Records:")
print(df.head())


# 2. Explore dataset

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
df.info()

print("\nStatistical Summary:")
print(df.describe(include="all"))

print("\nMissing Values:")
print(df.isnull().sum())


# 3. Clean column names

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

print("\nColumn Names:")
print(df.columns.tolist())


# 4. Identify date and closing price columns

date_candidates = [
    "date",
    "datetime",
    "timestamp"
]

price_candidates = [
    "close",
    "closing_price",
    "close_price",
    "adj_close",
    "adj_close_price"
]

date_column = next(
    (col for col in date_candidates if col in df.columns),
    None
)

price_column = next(
    (col for col in price_candidates if col in df.columns),
    None
)

if date_column is None or price_column is None:
    raise ValueError(
        "Dataset must contain a Date column and a Close or Adj Close column. "
        "Available columns: " + str(df.columns.tolist())
    )


# 5. Convert date and price columns

df[date_column] = pd.to_datetime(
    df[date_column],
    errors="coerce"
)

df[price_column] = pd.to_numeric(
    df[price_column].astype(str).str.replace(",", "", regex=False),
    errors="coerce"
)

if "volume" in df.columns:
    df["volume"] = pd.to_numeric(
        df["volume"].astype(str).str.replace(",", "", regex=False),
        errors="coerce"
    )


# 6. Handle missing values and clean dataset

df = df.dropna(
    subset=[date_column, price_column]
)

df = df.drop_duplicates()

df = df.sort_values(date_column)

df = df.set_index(date_column)

df = df[df[price_column] > 0]

if df.empty:
    raise ValueError("No valid stock price records found.")

print("\nCleaned Dataset:")
print(df.head())


# 7. Plot historical stock prices

plt.figure(figsize=(10, 5))

plt.plot(
    df.index,
    df[price_column],
    color="blue",
    label="Closing Price"
)

plt.title("Historical Stock Price Trend")
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# 8. Calculate moving averages

df["MA_7"] = df[price_column].rolling(window=7).mean()

df["MA_30"] = df[price_column].rolling(window=30).mean()

plt.figure(figsize=(10, 5))

plt.plot(
    df.index,
    df[price_column],
    label="Closing Price"
)

plt.plot(
    df.index,
    df["MA_7"],
    label="7-Day Moving Average"
)

plt.plot(
    df.index,
    df["MA_30"],
    label="30-Day Moving Average"
)

plt.title("Stock Price with Moving Averages")
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# 9. Weekly seasonality analysis

df["Day"] = df.index.day_name()

weekly_pattern = df.groupby("Day")[price_column].mean()

day_order = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday"
]

weekly_pattern = weekly_pattern.reindex(
    [day for day in day_order if day in weekly_pattern.index]
)

print("\nAverage Stock Price by Day:")
print(weekly_pattern)

plt.figure(figsize=(9, 5))

weekly_pattern.plot(kind="bar", color="orange")

plt.title("Weekly Stock Price Pattern")
plt.xlabel("Day")
plt.ylabel("Average Closing Price")
plt.tight_layout()
plt.show()


# 10. Monthly seasonality analysis

df["Month"] = df.index.month

monthly_pattern = df.groupby("Month")[price_column].mean()

print("\nAverage Stock Price by Month:")
print(monthly_pattern)

plt.figure(figsize=(9, 5))

monthly_pattern.plot(kind="bar", color="green")

plt.title("Monthly Stock Price Pattern")
plt.xlabel("Month")
plt.ylabel("Average Closing Price")
plt.tight_layout()
plt.show()


# 11. Yearly trend analysis

yearly_pattern = df[price_column].resample("YE").mean()

print("\nYearly Average Stock Price:")
print(yearly_pattern)

plt.figure(figsize=(9, 5))

plt.plot(
    yearly_pattern.index,
    yearly_pattern.values,
    marker="o",
    color="purple"
)

plt.title("Yearly Average Stock Price")
plt.xlabel("Year")
plt.ylabel("Average Closing Price")
plt.grid(True)
plt.tight_layout()
plt.show()


# 12. Correlation analysis

numeric_df = df.select_dtypes(include=np.number)

correlation = numeric_df.corr()

print("\nCorrelation Matrix:")
print(correlation)

plt.figure(figsize=(9, 6))

plt.imshow(
    correlation,
    cmap="coolwarm",
    aspect="auto"
)

plt.colorbar(label="Correlation")

plt.xticks(
    range(len(correlation.columns)),
    correlation.columns,
    rotation=45
)

plt.yticks(
    range(len(correlation.columns)),
    correlation.columns
)

plt.title("Stock Market Correlation Heatmap")
plt.tight_layout()
plt.show()


# 13. Analyze stock price and volume relationship

if "volume" in df.columns:

    plt.figure(figsize=(8, 5))

    plt.scatter(
        df["volume"],
        df[price_column],
        alpha=0.6,
        color="red"
    )

    plt.title("Stock Price vs Trading Volume")
    plt.xlabel("Trading Volume")
    plt.ylabel("Closing Price")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# 14. Prepare data for ARIMA model

# Use numerical index to avoid unsupported date index errors

stock_data = df[price_column].dropna().reset_index(drop=True)

if len(stock_data) < 20:
    raise ValueError(
        "At least 20 valid stock price records are required."
    )

# Split data into training and testing sets

split_point = int(len(stock_data) * 0.8)

train_data = stock_data.iloc[:split_point]

test_data = stock_data.iloc[split_point:]

print("\nTraining Records:", len(train_data))
print("Testing Records:", len(test_data))


# 15. Build ARIMA model

model = ARIMA(
    train_data,
    order=(1, 1, 1)
)

model_fit = model.fit()

print("\nARIMA Model Summary:")
print(model_fit.summary())


# 16. Forecast testing period

test_predictions = model_fit.forecast(
    steps=len(test_data)
)

test_predictions = pd.Series(
    np.asarray(test_predictions),
    index=test_data.index
)

# Calculate prediction errors

mae = mean_absolute_error(
    test_data,
    test_predictions
)

rmse = np.sqrt(
    mean_squared_error(
        test_data,
        test_predictions
    )
)

print("\nModel Evaluation:")
print("Mean Absolute Error:", mae)
print("Root Mean Squared Error:", rmse)


# 17. Plot actual and predicted stock prices

date_index = df.index

plt.figure(figsize=(10, 5))

plt.plot(
    date_index[:split_point],
    train_data,
    label="Training Data"
)

plt.plot(
    date_index[split_point:],
    test_data,
    label="Actual Stock Price"
)

plt.plot(
    date_index[split_point:],
    test_predictions,
    label="Predicted Stock Price",
    color="red"
)

plt.title("Actual vs Predicted Stock Prices")
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# 18. Forecast future stock prices

final_model = ARIMA(
    stock_data,
    order=(1, 1, 1)
)

final_model_fit = final_model.fit()

future_days = 30

future_predictions = final_model_fit.forecast(
    steps=future_days
)

future_dates = pd.bdate_range(
    start=df.index[-1] + pd.offsets.BDay(1),
    periods=future_days
)

future_predictions = pd.Series(
    np.asarray(future_predictions),
    index=future_dates
)

print("\nNext 30 Business Day Stock Price Forecast:")
print(future_predictions)


# 19. Plot future stock price forecast

plt.figure(figsize=(10, 5))

plt.plot(
    df.index,
    df[price_column],
    label="Historical Stock Price"
)

plt.plot(
    future_predictions.index,
    future_predictions,
    label="Future Forecast",
    color="red",
    marker="o"
)

plt.title("Future Stock Price Forecast using ARIMA")
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# 20. Export forecast data

future_predictions.to_csv(
    "stock_price_forecast.csv",
    header=["Predicted_Price"]
)

df.to_csv("cleaned_stock_data.csv")

print("\nStock Market Time Series Analysis Completed Successfully!")

# ============================================================
# AIM:
# To perform stock market time series analysis and forecasting.
#
# 1. Import historical stock price data.
# 2. Explore and clean the dataset.
# 3. Visualize historical stock prices.
# 4. Calculate 7-day and 30-day moving averages.
# 5. Analyze weekly, monthly and yearly patterns.
# 6. Perform correlation analysis.
# 7. Study the relationship between stock price and volume.
# 8. Build an ARIMA forecasting model.
# 9. Evaluate the model using MAE and RMSE.
# 10. Forecast future stock prices.
# 11. Export cleaned data and forecast results.
# ============================================================


# ============================================================
# INSTALLATION AND EXECUTION COMMANDS
# ============================================================

# Windows:
# python -m pip install pandas numpy matplotlib statsmodels scikit-learn
# python practical_stock_prices.py

# Ubuntu/Linux:
# python3 -m pip install pandas numpy matplotlib statsmodels scikit-learn
# python3 practical_stock_prices.py

# Keep Stock_Prices.csv in the same folder
# as the Python program.


# ============================================================
# 1. IMPORTING LIBRARIES
# ============================================================

# import pandas as pd
# Imports Pandas for loading, cleaning and analyzing data.
# pd is the short name used for Pandas.


# import numpy as np
# Imports NumPy for numerical calculations.
# It is used here to convert predictions into arrays
# and calculate the square root of an error value.


# import matplotlib.pyplot as plt
# Imports Pyplot for creating graphs and charts.
# plt is the short name used for Pyplot.


# from statsmodels.tsa.arima.model import ARIMA
# Imports the ARIMA time series forecasting model.
#
# ARIMA stands for:
# AutoRegressive Integrated Moving Average.
#
# It uses historical values and patterns
# to model a time series and generate forecasts.


# from sklearn.metrics import mean_absolute_error, mean_squared_error
# Imports two functions used to evaluate predictions.
#
# mean_absolute_error:
# Calculates the average absolute prediction error.
#
# mean_squared_error:
# Calculates the average squared prediction error.
#
# These functions help measure how far predictions
# are from actual stock prices.


# ============================================================
# 2. IMPORTING THE DATASET
# ============================================================

# df = pd.read_csv("Stock_Prices.csv")
# Loads historical stock price data from a CSV file.
#
# df stores the data in a Pandas DataFrame.
#
# The dataset may contain:
# Date.
# Open price.
# High price.
# Low price.
# Closing price.
# Adjusted closing price.
# Trading volume.


# print("First Five Records:")
# Displays a heading for the first five records.


# print(df.head())
# Displays the first five rows of the dataset.
#
# head() returns five rows by default.
# It helps us understand the dataset structure.


# ============================================================
# 3. EXPLORING THE DATASET
# ============================================================

# print("\nDataset Shape:")
# Displays the dataset dimensions heading.


# print(df.shape)
# Displays the number of rows and columns.
#
# Example:
# (500, 7)
#
# This means 500 rows and 7 columns.
# Actual values depend on the CSV file.


# print("\nDataset Information:")
# Displays the dataset information heading.


# df.info()
# Displays:
# Column names.
# Data types.
# Number of non-missing values.
# Memory usage.
#
# It helps identify missing values
# and incorrect data types.


# print("\nStatistical Summary:")
# Displays the statistical summary heading.


# print(df.describe(include="all"))
# Displays descriptive statistics for all columns.
#
# For numerical columns:
# count = Number of non-missing values.
# mean = Average.
# std = Standard deviation.
# min = Minimum.
# max = Maximum.
# 25%, 50%, 75% = Quartiles.
#
# include="all":
# Includes numerical and non-numerical columns.


# print("\nMissing Values:")
# Displays the missing-value heading.


# print(df.isnull().sum())
# Counts missing values in every column.
#
# isnull() identifies missing values.
# sum() counts them column by column.


# ============================================================
# 4. CLEANING COLUMN NAMES
# ============================================================

# df.columns = (
#     df.columns
#     .str.strip()
#     .str.lower()
#     .str.replace(" ", "_")
# )
#
# Standardizes the column names.
#
# .str.strip():
# Removes spaces from the beginning and end.
#
# .str.lower():
# Converts names to lowercase.
#
# .str.replace(" ", "_"):
# Replaces spaces with underscores.
#
# Example:
# "Closing Price" becomes "closing_price".
# "Adj Close" becomes "adj_close".
#
# This makes column names easier to access.


# print("\nColumn Names:")
# Displays the column names heading.


# print(df.columns.tolist())
# Converts the column names into a Python list
# and displays them.


# ============================================================
# 5. IDENTIFYING DATE AND PRICE COLUMNS
# ============================================================

# date_candidates = [
#     "date",
#     "datetime",
#     "timestamp"
# ]
#
# Creates a list of possible date column names.
#
# The dataset can use any of these names
# for its date information.


# price_candidates = [
#     "close",
#     "closing_price",
#     "close_price",
#     "adj_close",
#     "adj_close_price"
# ]
#
# Creates a list of possible closing price column names.
#
# Closing price:
# The stock price at the end of a trading session.
#
# Adjusted closing price:
# A closing price adjusted for certain corporate actions,
# such as stock splits and dividends.


# date_column = next(
#     (col for col in date_candidates if col in df.columns),
#     None
# )
#
# Finds the first available date column.
#
# next():
# Returns the first matching value.
#
# if col in df.columns:
# Checks whether a candidate column exists.
#
# If a match is found:
# Its name is stored in date_column.
#
# Otherwise:
# None is returned.


# price_column = next(
#     (col for col in price_candidates if col in df.columns),
#     None
# )
#
# Finds the first available closing price column.
#
# Its name is stored in price_column.
#
# If no matching column exists:
# None is returned.


# if date_column is None or price_column is None:
# Checks whether either required column is missing.
#
# The program requires:
# A date column.
# A closing price or adjusted closing price column.


#     raise ValueError(
#         "Dataset must contain a Date column and a Close or Adj Close column. "
#         "Available columns: " + str(df.columns.tolist())
#     )
#
# Raises an error and stops execution
# if the required columns are not found.
#
# str(df.columns.tolist()):
# Converts the available column names into text.
#
# This helps identify which column names
# are present in the dataset.


# ============================================================
# 6. CONVERTING DATE AND PRICE COLUMNS
# ============================================================

# df[date_column] = pd.to_datetime(
#     df[date_column],
#     errors="coerce"
# )
#
# Converts the date column into datetime format.
#
# pd.to_datetime():
# Converts date values into dates.
#
# errors="coerce":
# Converts invalid dates into NaT.
#
# NaT means Not a Time.
#
# Datetime format supports sorting and time-based analysis.


# df[price_column] = pd.to_numeric(
#     df[price_column].astype(str).str.replace(",", "", regex=False),
#     errors="coerce"
# )
#
# Converts the selected closing price column into numbers.
#
# astype(str):
# Converts values into text temporarily.
#
# str.replace(",", "", regex=False):
# Removes commas from price values.
#
# Example:
# "1,250.50" becomes "1250.50".
#
# pd.to_numeric():
# Converts the cleaned text into a number.
#
# errors="coerce":
# Converts invalid values into NaN.


# if "volume" in df.columns:
# Checks whether a volume column exists.
#
# Volume represents the number of shares traded
# during a particular trading period.


#     df["volume"] = pd.to_numeric(
#         df["volume"].astype(str).str.replace(",", "", regex=False),
#         errors="coerce"
#     )
#
# Converts volume values into numerical format.
#
# Commas are removed before conversion.
# Invalid values become NaN.
#
# The conversion is performed only when
# the volume column exists.


# ============================================================
# 7. HANDLING MISSING VALUES AND CLEANING DATA
# ============================================================

# df = df.dropna(
#     subset=[date_column, price_column]
# )
#
# Removes rows where the date or price is missing.
#
# Both date and closing price are required
# for time series analysis.


# df = df.drop_duplicates()
# Removes duplicate rows from the dataset.
#
# This prevents repeated records from affecting
# the analysis and forecasting.


# df = df.sort_values(date_column)
# Sorts the records by date in ascending order.
#
# The oldest record appears first.
# The newest record appears last.
#
# Time series data should be arranged chronologically.


# df = df.set_index(date_column)
# Makes the date column the DataFrame index.
#
# The dates are now used as row labels.
#
# This makes time-based plotting and resampling easier.


# df = df[df[price_column] > 0]
# Retains only records with positive closing prices.
#
# Zero and negative prices are removed
# because they are not valid stock prices
# for this analysis.


# if df.empty:
#     raise ValueError("No valid stock price records found.")
#
# Checks whether the dataset has any rows remaining.
#
# If no valid records remain:
# An error is raised and the program stops.


# print("\nCleaned Dataset:")
# Displays the cleaned dataset heading.


# print(df.head())
# Displays the first five cleaned records.


# ============================================================
# 8. PLOTTING HISTORICAL STOCK PRICES
# ============================================================

# plt.figure(figsize=(10, 5))
# Creates a graph with width 10 inches
# and height 5 inches.


# plt.plot(
#     df.index,
#     df[price_column],
#     color="blue",
#     label="Closing Price"
# )
#
# Creates a line graph of historical closing prices.
#
# X-axis:
# Date.
#
# Y-axis:
# Closing price.
#
# color="blue":
# Sets the line color to blue.
#
# label="Closing Price":
# Names the line for the legend.
#
# This graph shows how the stock price
# changed over the recorded period.


# plt.title("Historical Stock Price Trend")
# Sets the graph title.


# plt.xlabel("Date")
# Labels the X-axis as Date.


# plt.ylabel("Stock Price")
# Labels the Y-axis as Stock Price.


# plt.legend()
# Displays the legend.


# plt.grid(True)
# Displays grid lines.


# plt.tight_layout()
# Adjusts the layout to prevent labels
# from being cut off.


# plt.show()
# Displays the historical stock price graph.


# ============================================================
# 9. CALCULATING MOVING AVERAGES
# ============================================================

# df["MA_7"] = df[price_column].rolling(window=7).mean()
#
# Calculates the 7-observation moving average.
#
# rolling(window=7):
# Creates a moving window containing seven observations.
#
# mean():
# Calculates the average within each window.
#
# MA_7:
# Stores the 7-observation moving average.
#
# It helps smooth short-term price fluctuations.
#
# The first six rows will generally contain NaN
# because seven observations are needed
# for the first complete window.
#
# Important:
# This means seven observations, not necessarily
# seven calendar days.


# df["MA_30"] = df[price_column].rolling(window=30).mean()
#
# Calculates the 30-observation moving average.
#
# The average is calculated using the latest
# 30 observations in each window.
#
# It helps identify the broader price trend.
#
# The first 29 rows will generally contain NaN.


# plt.figure(figsize=(10, 5))
# Creates a new graph.


# plt.plot(
#     df.index,
#     df[price_column],
#     label="Closing Price"
# )
#
# Plots the original closing prices.


# plt.plot(
#     df.index,
#     df["MA_7"],
#     label="7-Day Moving Average"
# )
#
# Plots the 7-observation moving average.
#
# It helps show short-term changes
# with less day-to-day fluctuation.


# plt.plot(
#     df.index,
#     df["MA_30"],
#     label="30-Day Moving Average"
# )
#
# Plots the 30-observation moving average.
#
# It helps show the longer-term trend.


# plt.title("Stock Price with Moving Averages")
# Sets the graph title.


# plt.xlabel("Date")
# Labels the X-axis.


# plt.ylabel("Stock Price")
# Labels the Y-axis.


# plt.legend()
# Identifies all three plotted lines.


# plt.grid(True)
# Displays grid lines.


# plt.tight_layout()
# Adjusts the graph layout.


# plt.show()
# Displays the stock prices and moving averages graph.


# ============================================================
# 10. WEEKLY SEASONALITY ANALYSIS
# ============================================================

# df["Day"] = df.index.day_name()
#
# Extracts the weekday name from each date.
#
# Example:
# Monday, Tuesday, Wednesday.
#
# The result is stored in a new column named Day.


# weekly_pattern = df.groupby("Day")[price_column].mean()
#
# Groups stock prices by weekday.
#
# groupby("Day"):
# Groups all Monday records together,
# all Tuesday records together, and so on.
#
# mean():
# Calculates the average closing price
# for each weekday.
#
# The result is stored in weekly_pattern.


# day_order = [
#     "Monday",
#     "Tuesday",
#     "Wednesday",
#     "Thursday",
#     "Friday",
#     "Saturday",
#     "Sunday"
# ]
#
# Creates a list containing weekdays
# in calendar order.


# weekly_pattern = weekly_pattern.reindex(
#     [day for day in day_order if day in weekly_pattern.index]
# )
#
# Reorders the weekday results into calendar order.
#
# reindex():
# Changes the order of the index labels.
#
# if day in weekly_pattern.index:
# Includes only days that are actually present.
#
# This prevents weekdays from being displayed
# in alphabetical order.


# print("\nAverage Stock Price by Day:")
# Displays the weekday analysis heading.


# print(weekly_pattern)
# Displays average stock prices by weekday.


# plt.figure(figsize=(9, 5))
# Creates a new graph.


# weekly_pattern.plot(kind="bar", color="orange")
#
# Creates a bar chart of average prices by weekday.
#
# kind="bar":
# Creates a bar chart.
#
# color="orange":
# Sets the bar color to orange.


# plt.title("Weekly Stock Price Pattern")
# Sets the graph title.


# plt.xlabel("Day")
# Labels the X-axis as Day.


# plt.ylabel("Average Closing Price")
# Labels the Y-axis.


# plt.tight_layout()
# Adjusts the graph layout.


# plt.show()
# Displays the weekday analysis graph.
#
# Note:
# Stock exchanges are normally closed on weekends.
# Saturday and Sunday may therefore be absent.


# ============================================================
# 11. MONTHLY SEASONALITY ANALYSIS
# ============================================================

# df["Month"] = df.index.month
#
# Extracts the month number from each date.
#
# January = 1.
# February = 2.
# March = 3.
# ...
# December = 12.
#
# The result is stored in a new Month column.


# monthly_pattern = df.groupby("Month")[price_column].mean()
#
# Groups stock prices by month number.
#
# Calculates the average closing price
# for each month number across all years.
#
# Example:
# All January observations are grouped together.


# print("\nAverage Stock Price by Month:")
# Displays the monthly analysis heading.


# print(monthly_pattern)
# Displays the average closing price for each month.


# plt.figure(figsize=(9, 5))
# Creates a new graph.


# monthly_pattern.plot(kind="bar", color="green")
#
# Creates a bar chart of average prices by month.
#
# The month numbers are used as the categories.
#
# color="green":
# Sets the bar color to green.


# plt.title("Monthly Stock Price Pattern")
# Sets the graph title.


# plt.xlabel("Month")
# Labels the X-axis as Month.


# plt.ylabel("Average Closing Price")
# Labels the Y-axis.


# plt.tight_layout()
# Adjusts the graph layout.


# plt.show()
# Displays the monthly pattern graph.
#
# Important:
# This compares month numbers across years.
# It does not compare each individual calendar month
# such as January 2024 with January 2025 separately.


# ============================================================
# 12. YEARLY TREND ANALYSIS
# ============================================================

# yearly_pattern = df[price_column].resample("YE").mean()
#
# Groups the time series into calendar-year periods.
#
# resample("YE"):
# Uses year-end frequency.
#
# mean():
# Calculates the average closing price
# for each year.
#
# The result is stored in yearly_pattern.


# print("\nYearly Average Stock Price:")
# Displays the yearly analysis heading.


# print(yearly_pattern)
# Displays average closing prices for each year.


# plt.figure(figsize=(9, 5))
# Creates a new graph.


# plt.plot(
#     yearly_pattern.index,
#     yearly_pattern.values,
#     marker="o",
#     color="purple"
# )
#
# Creates a line graph of yearly average prices.
#
# X-axis:
# Year-end dates.
#
# Y-axis:
# Average closing price for the year.
#
# marker="o":
# Displays circular markers.
#
# color="purple":
# Sets the line color to purple.


# plt.title("Yearly Average Stock Price")
# Sets the graph title.


# plt.xlabel("Year")
# Labels the X-axis as Year.


# plt.ylabel("Average Closing Price")
# Labels the Y-axis.


# plt.grid(True)
# Displays grid lines.


# plt.tight_layout()
# Adjusts the graph layout.


# plt.show()
# Displays the yearly trend graph.


# ============================================================
# 13. CORRELATION ANALYSIS
# ============================================================

# numeric_df = df.select_dtypes(include=np.number)
#
# Selects only numerical columns from the DataFrame.
#
# include=np.number:
# Includes numerical data types.
#
# Examples:
# Closing price.
# Volume.
# Moving averages.
# Month number.
#
# Text columns such as Day are excluded.


# correlation = numeric_df.corr()
#
# Calculates the correlation matrix
# between numerical columns.
#
# corr():
# Measures pairwise linear relationships.
#
# Correlation values range from -1 to +1.
#
# +1 = Perfect positive linear correlation.
# -1 = Perfect negative linear correlation.
#  0 = No linear correlation.
#
# Correlation does not prove causation.


# print("\nCorrelation Matrix:")
# Displays the correlation matrix heading.


# print(correlation)
# Displays the correlation values.


# plt.figure(figsize=(9, 6))
# Creates a new graph.


# plt.imshow(
#     correlation,
#     cmap="coolwarm",
#     aspect="auto"
# )
#
# Displays the correlation matrix as a heatmap.
#
# imshow():
# Converts numerical matrix values into colored cells.
#
# cmap="coolwarm":
# Uses colors to distinguish correlation values.
#
# aspect="auto":
# Automatically adjusts the cell proportions.


# plt.colorbar(label="Correlation")
# Displays a color scale representing correlation values.


# plt.xticks(
#     range(len(correlation.columns)),
#     correlation.columns,
#     rotation=45
# )
#
# Sets the X-axis labels to column names.
#
# range():
# Creates positions for the labels.
#
# rotation=45:
# Rotates labels to improve readability.


# plt.yticks(
#     range(len(correlation.columns)),
#     correlation.columns
# )
#
# Sets the Y-axis labels to the same column names.


# plt.title("Stock Market Correlation Heatmap")
# Sets the graph title.


# plt.tight_layout()
# Adjusts the graph layout.


# plt.show()
# Displays the correlation heatmap.


# ============================================================
# 14. ANALYZING STOCK PRICE AND TRADING VOLUME
# ============================================================

# if "volume" in df.columns:
#
# Checks whether the dataset contains a volume column.
#
# If volume is unavailable, the scatter plot is skipped.


#     plt.figure(figsize=(8, 5))
# Creates a new graph.


#     plt.scatter(
#         df["volume"],
#         df[price_column],
#         alpha=0.6,
#         color="red"
#     )
#
# Creates a scatter plot of trading volume
# against closing price.
#
# X-axis:
# Trading volume.
#
# Y-axis:
# Closing price.
#
# alpha=0.6:
# Makes the points partially transparent.
#
# color="red":
# Sets the points to red.
#
# This graph helps investigate whether
# volume and closing price show a relationship.


#     plt.title("Stock Price vs Trading Volume")
# Sets the graph title.


#     plt.xlabel("Trading Volume")
# Labels the X-axis.


#     plt.ylabel("Closing Price")
# Labels the Y-axis.


#     plt.grid(True)
# Displays grid lines.


#     plt.tight_layout()
# Adjusts the graph layout.


#     plt.show()
# Displays the stock price versus volume scatter plot.


# ============================================================
# 15. PREPARING DATA FOR THE ARIMA MODEL
# ============================================================

# stock_data = df[price_column].dropna().reset_index(drop=True)
#
# Selects the closing price series.
#
# dropna():
# Removes any remaining missing closing prices.
#
# reset_index(drop=True):
# Replaces the date index with a simple numerical index.
#
# The index becomes:
# 0, 1, 2, 3, ...
#
# This avoids some date-index frequency issues
# when fitting and forecasting with ARIMA.
#
# The observations must still remain
# in chronological order.


# if len(stock_data) < 20:
#
# Checks whether fewer than 20 valid price records exist.


#     raise ValueError(
#         "At least 20 valid stock price records are required."
#     )
#
# Stops the program if there are fewer than 20 records.
#
# This is a minimum record-count check
# chosen by the program.
#
# Twenty observations do not guarantee
# that the model will forecast accurately.


# ============================================================
# 16. SPLITTING DATA INTO TRAINING AND TESTING SETS
# ============================================================

# split_point = int(len(stock_data) * 0.8)
#
# Calculates the position that divides the dataset.
#
# 0.8 means 80 percent.
#
# int():
# Converts the result into an integer index.
#
# Example:
# If there are 100 observations,
# split_point will be 80.


# train_data = stock_data.iloc[:split_point]
#
# Selects the first 80 percent of the observations.
#
# iloc:
# Selects data by integer position.
#
# These records are used to train the model.


# test_data = stock_data.iloc[split_point:]
#
# Selects the remaining 20 percent of the observations.
#
# These records are used to evaluate predictions
# against actual stock prices.


# print("\nTraining Records:", len(train_data))
# Displays the number of training observations.


# print("Testing Records:", len(test_data))
# Displays the number of testing observations.
#
# A time series is split chronologically,
# rather than randomly, to preserve time order.


# ============================================================
# 17. BUILDING THE ARIMA MODEL
# ============================================================

# model = ARIMA(
#     train_data,
#     order=(1, 1, 1)
# )
#
# Creates an ARIMA model using the training data.
#
# order=(p, d, q):
#
# p = Number of autoregressive terms.
# d = Number of differencing operations.
# q = Number of moving-average error terms.
#
# order=(1, 1, 1) means:
#
# p=1:
# Uses one autoregressive term.
#
# d=1:
# Applies differencing once.
#
# q=1:
# Uses one moving-average error term.
#
# ARIMA models past values and patterns
# to generate forecasts.
#
# The order is a model configuration.
# It is not automatically the best order
# for every stock dataset.


# model_fit = model.fit()
#
# Fits the ARIMA model to the training data.
#
# fit():
# Estimates the model parameters from the data.
#
# The fitted model is stored in model_fit.


# print("\nARIMA Model Summary:")
# Displays the model summary heading.


# print(model_fit.summary())
# Displays statistical information about the fitted model.
#
# The summary may include:
# Model configuration.
# Estimated coefficients.
# Standard errors.
# Statistical tests.
# Information criteria.
#
# These details help examine the fitted model.


# ============================================================
# 18. FORECASTING THE TESTING PERIOD
# ============================================================

# test_predictions = model_fit.forecast(
#     steps=len(test_data)
# )
#
# Forecasts the same number of observations
# as there are in the testing dataset.
#
# steps:
# Specifies how many future observations to predict.
#
# The predictions are generated from the training data.


# test_predictions = pd.Series(
#     np.asarray(test_predictions),
#     index=test_data.index
# )
#
# Converts the predictions into a Pandas Series.
#
# np.asarray():
# Converts the prediction output into a NumPy array.
#
# index=test_data.index:
# Assigns the testing data's numerical index
# to the predictions.
#
# This aligns actual and predicted values
# for error calculations.


# ============================================================
# 19. CALCULATING PREDICTION ERRORS
# ============================================================

# mae = mean_absolute_error(
#     test_data,
#     test_predictions
# )
#
# Calculates Mean Absolute Error.
#
# MAE measures the average absolute difference
# between actual and predicted values.
#
# Formula:
# MAE = Average of |Actual - Predicted|.
#
# Lower MAE generally means smaller prediction errors.
#
# The MAE is expressed in the same units
# as the stock price.


# rmse = np.sqrt(
#     mean_squared_error(
#         test_data,
#         test_predictions
#     )
# )
#
# Calculates Root Mean Squared Error.
#
# Step 1:
# mean_squared_error() calculates MSE.
#
# MSE:
# Average of squared prediction errors.
#
# Step 2:
# np.sqrt() calculates the square root of MSE.
#
# RMSE:
# Root Mean Squared Error.
#
# RMSE penalizes large errors more strongly
# than MAE because the errors are squared first.
#
# Lower RMSE generally indicates smaller errors.
#
# Both metrics are expressed in stock price units
# after the square root is taken.


# print("\nModel Evaluation:")
# Displays the model evaluation heading.


# print("Mean Absolute Error:", mae)
# Displays the MAE value.


# print("Root Mean Squared Error:", rmse)
# Displays the RMSE value.
#
# The actual values depend on the dataset
# and the fitted model.


# ============================================================
# 20. PLOTTING ACTUAL AND PREDICTED STOCK PRICES
# ============================================================

# date_index = df.index
#
# Stores the dates from the cleaned DataFrame.
#
# These dates are used for the forecast comparison graph.


# plt.figure(figsize=(10, 5))
# Creates a new graph.


# plt.plot(
#     date_index[:split_point],
#     train_data,
#     label="Training Data"
# )
#
# Plots the training observations.
#
# X-axis:
# Dates belonging to the training period.
#
# Y-axis:
# Actual closing prices.


# plt.plot(
#     date_index[split_point:],
#     test_data,
#     label="Actual Stock Price"
# )
#
# Plots the actual stock prices in the testing period.
#
# These values are compared with model predictions.


# plt.plot(
#     date_index[split_point:],
#     test_predictions,
#     label="Predicted Stock Price",
#     color="red"
# )
#
# Plots the predicted prices over the testing dates.
#
# color="red":
# Sets the prediction line to red.
#
# Comparing the actual and predicted lines
# helps visually assess forecast performance.


# plt.title("Actual vs Predicted Stock Prices")
# Sets the graph title.


# plt.xlabel("Date")
# Labels the X-axis as Date.


# plt.ylabel("Stock Price")
# Labels the Y-axis as Stock Price.


# plt.legend()
# Displays the legend.


# plt.grid(True)
# Displays grid lines.


# plt.tight_layout()
# Adjusts the graph layout.


# plt.show()
# Displays the actual-versus-predicted graph.


# ============================================================
# 21. FORECASTING FUTURE STOCK PRICES
# ============================================================

# final_model = ARIMA(
#     stock_data,
#     order=(1, 1, 1)
# )
#
# Creates a new ARIMA model using all available
# cleaned closing price observations.
#
# This model is used for the future forecast,
# after the test evaluation has been performed.


# final_model_fit = final_model.fit()
#
# Fits the final ARIMA model to all available data.


# future_days = 30
#
# Sets the number of future observations to forecast
# to 30.


# future_predictions = final_model_fit.forecast(
#     steps=future_days
# )
#
# Forecasts the next 30 observations.
#
# The forecast is based on the fitted ARIMA model.
#
# These are estimates, not guaranteed future prices.


# future_dates = pd.bdate_range(
#     start=df.index[-1] + pd.offsets.BDay(1),
#     periods=future_days
# )
#
# Generates 30 future business dates.
#
# df.index[-1]:
# Gets the last date in the historical dataset.
#
# pd.offsets.BDay(1):
# Moves forward by one business day.
#
# pd.bdate_range():
# Generates dates on weekdays, normally Monday to Friday.
#
# periods=future_days:
# Creates 30 dates.
#
# Important:
# Business days do not account for every exchange holiday.
# Also, the ARIMA forecast steps represent observations,
# so the generated dates should match the frequency
# of the original stock observations.


# future_predictions = pd.Series(
#     np.asarray(future_predictions),
#     index=future_dates
# )
#
# Converts future predictions into a Pandas Series.
#
# np.asarray():
# Converts predictions into a NumPy array.
#
# index=future_dates:
# Assigns future business dates to the predictions.
#
# The result contains forecast dates and prices.


# print("\nNext 30 Business Day Stock Price Forecast:")
# Displays the future forecast heading.


# print(future_predictions)
# Displays the predicted stock prices for the future dates.


# ============================================================
# 22. PLOTTING THE FUTURE FORECAST
# ============================================================

# plt.figure(figsize=(10, 5))
# Creates a new graph.


# plt.plot(
#     df.index,
#     df[price_column],
#     label="Historical Stock Price"
# )
#
# Plots historical closing prices.


# plt.plot(
#     future_predictions.index,
#     future_predictions,
#     label="Future Forecast",
#     color="red",
#     marker="o"
# )
#
# Plots the forecasted stock prices.
#
# X-axis:
# Future business dates.
#
# Y-axis:
# Predicted stock prices.
#
# color="red":
# Sets the forecast line to red.
#
# marker="o":
# Displays circular markers at forecast points.


# plt.title("Future Stock Price Forecast using ARIMA")
# Sets the graph title.


# plt.xlabel("Date")
# Labels the X-axis.


# plt.ylabel("Stock Price")
# Labels the Y-axis.


# plt.legend()
# Identifies the historical and forecast lines.


# plt.grid(True)
# Displays grid lines.


# plt.tight_layout()
# Adjusts the graph layout.


# plt.show()
# Displays the historical prices and future forecast.


# ============================================================
# 23. EXPORTING FORECAST DATA
# ============================================================

# future_predictions.to_csv(
#     "stock_price_forecast.csv",
#     header=["Predicted_Price"]
# )
#
# Saves the future forecast into a CSV file.
#
# Output file:
# stock_price_forecast.csv
#
# header=["Predicted_Price"]:
# Sets the output column name to Predicted_Price.
#
# The forecast dates are saved as the index
# by default because index=False is not specified.


# df.to_csv("cleaned_stock_data.csv")
#
# Saves the cleaned stock dataset.
#
# Output file:
# cleaned_stock_data.csv
#
# The DataFrame index, containing dates,
# is also saved by default.


# print("\nStock Market Time Series Analysis Completed Successfully!")
#
# Displays the final completion message.
#
# It indicates that the analysis and forecasting
# steps have finished.
