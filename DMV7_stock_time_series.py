
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
# EXPLANATION LINE BY LINE
# ============================================================

# pandas: Used for loading and processing stock market data.
# numpy: Used for numerical calculations.
# matplotlib: Used for creating graphs.
# ARIMA: Used for time series forecasting.
# mean_absolute_error: Calculates average absolute prediction error.
# mean_squared_error: Calculates average squared prediction error.

# read_csv(): Loads the CSV dataset.
# head(): Displays first five records.
# shape: Displays rows and columns.
# info(): Displays dataset information.
# describe(): Displays statistical summary.
# isnull().sum(): Counts missing values.

# str.strip(): Removes extra spaces.
# str.lower(): Converts column names into lowercase.
# next(): Finds the first matching column.
# to_datetime(): Converts values into date format.
# to_numeric(): Converts values into numeric format.
# dropna(): Removes missing values.
# drop_duplicates(): Removes duplicate records.
# sort_values(): Sorts records by date.
# set_index(): Sets date as the DataFrame index.

# rolling(window=7): Calculates 7-record moving average.
# rolling(window=30): Calculates 30-record moving average.
# day_name(): Extracts weekday names.
# groupby(): Groups records according to a column.
# resample("YE"): Calculates yearly averages.
# corr(): Calculates correlation between numerical columns.
# imshow(): Displays correlation as a heatmap.
# scatter(): Creates a scatter plot.

# reset_index(): Replaces the date index with numerical index.
# iloc[]: Selects records by integer position.
# ARIMA(1,1,1): Creates the ARIMA forecasting model.
# fit(): Trains the ARIMA model.
# forecast(): Predicts future stock prices.
# np.asarray(): Converts forecast results into a NumPy array.
# mean_absolute_error(): Calculates average absolute error.
# mean_squared_error(): Calculates average squared error.
# np.sqrt(): Calculates square root.
# bdate_range(): Generates future business dates.
# Series(): Creates a Pandas series.
# to_csv(): Saves results into CSV files.

# figure(): Creates a new graph.
# plot(): Creates a line graph.
# bar(): Creates a bar chart.
# title(): Sets graph title.
# xlabel(): Sets horizontal axis label.
# ylabel(): Sets vertical axis label.
# legend(): Displays graph labels.
# grid(): Displays grid lines.
# tight_layout(): Adjusts graph layout.
# show(): Displays the graph.
