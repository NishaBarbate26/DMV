# AIM: Analyze Air Quality Index (AQI) Trends using Matplotlib.

# Installation commands:
# Windows:
# python -m pip install pandas numpy matplotlib
# python practical_air_quality.py
#
# Ubuntu/Linux:
# pip install pandas numpy matplotlib
# python3 practical_air_quality.py


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Import dataset

df = pd.read_csv("City_Air_Quality.csv")

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
    .str.replace(".", "", regex=False)
)

print("\nColumn Names:")

print(df.columns.tolist())

# 4. Identify important columns

date_candidates = [
    "date",
    "datetime",
    "timestamp"
]

aqi_candidates = [
    "aqi",
    "air_quality_index"
]

date_column = next(
    (col for col in date_candidates if col in df.columns),
    None
)

aqi_column = next(
    (col for col in aqi_candidates if col in df.columns),
    None
)

if date_column is None or aqi_column is None:

    raise ValueError(
        "Dataset must contain Date and AQI columns."
    )

# 5. Convert date and numerical columns

df[date_column] = pd.to_datetime(
    df[date_column],
    errors="coerce"
)

pollutant_candidates = [
    "pm25",
    "pm10",
    "co",
    "no2",
    "so2",
    "o3"
]

pollutants = [
    col for col in pollutant_candidates
    if col in df.columns
]

numeric_columns = [aqi_column] + pollutants

for column in numeric_columns:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

# 6. Handle missing values

df = df.dropna(
    subset=[date_column, aqi_column]
)

for column in pollutants:

    df[column] = df[column].fillna(
        df[column].median()
    )

df = df.drop_duplicates()

# Sort records by date

df = df.sort_values(date_column)

print("\nCleaned Dataset:")

print(df.head())

print("\nAvailable Pollutants:")

print(pollutants)

# 7. Overall AQI trend using line plot

plt.figure(figsize=(10, 5))

plt.plot(
    df[date_column],
    df[aqi_column],
    marker="o",
    color="blue",
    label="AQI"
)

plt.title("Overall AQI Trend Over Time")

plt.xlabel("Date")

plt.ylabel("Air Quality Index")

plt.legend()

plt.xticks(rotation=45)

plt.grid(True)

plt.tight_layout()

plt.show()

# 8. Individual pollutant trend plots

for pollutant in pollutants:

    plt.figure(figsize=(10, 5))

    plt.plot(
        df[date_column],
        df[pollutant],
        marker="o",
        label=pollutant.upper()
    )

    plt.title(
        pollutant.upper() + " Levels Over Time"
    )

    plt.xlabel("Date")

    plt.ylabel("Pollutant Level")

    plt.legend()

    plt.xticks(rotation=45)

    plt.grid(True)

    plt.tight_layout()

    plt.show()

# 9. Bar plot to compare AQI values

plt.figure(figsize=(10, 5))

plt.bar(
    df[date_column].dt.strftime("%Y-%m-%d"),
    df[aqi_column],
    color="orange"
)

plt.title("AQI Comparison Across Dates")

plt.xlabel("Date")

plt.ylabel("AQI Value")

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()

# 10. Box plot for AQI distribution

plt.figure(figsize=(7, 5))

plt.boxplot(
    df[aqi_column].dropna(),
    patch_artist=True
)

plt.title("Distribution of AQI Values")

plt.ylabel("AQI")

plt.grid(True)

plt.tight_layout()

plt.show()

# 11. Box plot for pollutant distributions

if pollutants:

    plt.figure(figsize=(9, 5))

    plt.boxplot(
        [df[col].dropna() for col in pollutants],
        tick_labels=[col.upper() for col in pollutants]
    )

    plt.title("Distribution of Pollutant Levels")

    plt.xlabel("Pollutants")

    plt.ylabel("Pollutant Levels")

    plt.grid(True)

    plt.tight_layout()

    plt.show()

# 12. Scatter plots showing AQI and pollutant relationship

for pollutant in pollutants:

    plt.figure(figsize=(7, 5))

    plt.scatter(
        df[pollutant],
        df[aqi_column],
        color="green",
        alpha=0.7
    )

    plt.title(
        "AQI vs " + pollutant.upper()
    )

    plt.xlabel(pollutant.upper() + " Level")

    plt.ylabel("AQI")

    plt.grid(True)

    plt.tight_layout()

    plt.show()

# 13. Bubble chart for AQI and PM2.5

if "pm25" in df.columns:

    plt.figure(figsize=(8, 5))

    plt.scatter(
        df["pm25"],
        df[aqi_column],
        s=df[aqi_column].clip(lower=1) * 2,
        c=df[aqi_column],
        cmap="viridis",
        alpha=0.6
    )

    plt.title("Bubble Chart: PM2.5 vs AQI")

    plt.xlabel("PM2.5 Level")

    plt.ylabel("AQI")

    plt.colorbar(label="AQI")

    plt.tight_layout()

    plt.show()

# 14. Display AQI statistics

print("\nAQI Statistics:")

print("Maximum AQI:", df[aqi_column].max())

print("Minimum AQI:", df[aqi_column].min())

print("Average AQI:", df[aqi_column].mean())

print("\nPollutant Statistics:")

print(df[pollutants].describe())

# 15. Correlation between AQI and pollutants

if pollutants:

    correlation = df[
        [aqi_column] + pollutants
    ].corr()

    print("\nCorrelation Matrix:")

    print(correlation)

    plt.figure(figsize=(8, 6))

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

    plt.title("AQI and Pollutant Correlation")

    plt.tight_layout()

    plt.show()

# 16. Export cleaned dataset

df.to_csv(
    "cleaned_air_quality.csv",
    index=False
)

print("\nCleaned Dataset Saved Successfully!")

print("\nAir Quality Visualization Completed Successfully!")


# ============================================================
# EXPLANATION LINE BY LINE
# ============================================================

# import pandas as pd
# Imports Pandas for loading and handling data.

# import numpy as np
# Imports NumPy for numerical operations.

# import matplotlib.pyplot as plt
# Imports Matplotlib for creating graphs.

# pd.read_csv()
# Loads the CSV dataset.

# df.head()
# Displays the first five records.

# df.shape
# Displays the number of rows and columns.

# df.info()
# Displays column names, data types and non-null values.

# describe(include="all")
# Displays statistical information about the dataset.

# isnull().sum()
# Counts missing values in each column.

# str.strip()
# Removes unnecessary spaces from column names.

# str.lower()
# Converts column names into lowercase.

# str.replace()
# Replaces spaces and special characters in column names.

# next()
# Finds the first matching column name.

# pd.to_datetime()
# Converts date values into datetime format.

# pd.to_numeric()
# Converts pollutant and AQI values into numbers.

# errors="coerce"
# Converts invalid values into missing values.

# dropna()
# Removes records containing missing values.

# median()
# Calculates the middle value of numerical data.

# fillna()
# Replaces missing pollutant values with their median.

# drop_duplicates()
# Removes duplicate records.

# sort_values()
# Arranges records in ascending date order.

# plt.figure()
# Creates a new graph.

# plt.plot()
# Creates a line graph.

# plt.bar()
# Creates a bar graph.

# plt.boxplot()
# Displays the distribution and outliers in data.

# plt.scatter()
# Creates a scatter plot.

# s
# Controls the size of bubbles in a bubble chart.

# c
# Controls the color of the bubbles.

# cmap
# Selects the color scheme.

# plt.colorbar()
# Displays the color scale.

# max()
# Finds the maximum AQI value.

# min()
# Finds the minimum AQI value.

# mean()
# Calculates the average AQI value.

# corr()
# Calculates the correlation between numerical columns.

# plt.imshow()
# Displays correlation values as a heatmap.

# plt.xlabel()
# Sets the horizontal axis label.

# plt.ylabel()
# Sets the vertical axis label.

# plt.title()
# Sets the graph title.

# plt.legend()
# Displays the graph legend.

# plt.grid()
# Displays grid lines.

# plt.xticks()
# Rotates date labels for better readability.

# plt.tight_layout()
# Adjusts the graph layout.

# plt.show()
# Displays the graph.

# to_csv()
# Saves the cleaned dataset into a CSV file.