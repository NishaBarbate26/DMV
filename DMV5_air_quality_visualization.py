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
# AIM:
# To analyze and visualize city air quality data.
#
# 1. Import the air quality dataset.
# 2. Explore the dataset.
# 3. Clean column names.
# 4. Identify date, AQI and pollutant columns.
# 5. Convert columns into correct data types.
# 6. Handle missing values.
# 7. Visualize AQI trends.
# 8. Visualize individual pollutant trends.
# 9. Compare AQI values using bar charts.
# 10. Study AQI and pollutant distributions.
# 11. Analyze relationships using scatter plots.
# 12. Create a bubble chart.
# 13. Calculate descriptive statistics.
# 14. Perform correlation analysis.
# 15. Export the cleaned dataset.
# ============================================================


# ============================================================
# INSTALLATION AND EXECUTION COMMANDS
# ============================================================

# Windows:
# python -m pip install pandas numpy matplotlib
# python practical_air_quality.py

# Ubuntu/Linux:
# python3 -m pip install pandas numpy matplotlib
# python3 practical_air_quality.py

# Keep City_Air_Quality.csv in the same folder
# as the Python program.


# ============================================================
# 1. IMPORTING LIBRARIES
# ============================================================

# import pandas as pd
# Imports the Pandas library.
# It is used to load, clean and analyze the dataset.
# pd is the short name used for Pandas.


# import numpy as np
# Imports the NumPy library.
# It is used for numerical operations.
# NumPy is also useful for handling numerical data.


# import matplotlib.pyplot as plt
# Imports the Pyplot module from Matplotlib.
# It is used to create different types of graphs.
# plt is the short name used for Pyplot.


# ============================================================
# 2. IMPORTING THE DATASET
# ============================================================

# df = pd.read_csv("City_Air_Quality.csv")
# Reads the air quality dataset from a CSV file.
#
# pd.read_csv():
# Loads CSV data into a Pandas DataFrame.
#
# df:
# Stores the complete dataset.
#
# The dataset may contain:
# Date.
# AQI.
# PM2.5.
# PM10.
# CO.
# NO2.
# SO2.
# O3.


# print("First Five Records:")
# Displays a heading for the first five records.


# print(df.head())
# Displays the first five rows of the dataset.
#
# head() displays five rows by default.
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
# (100, 8)
#
# This means:
# 100 rows = 100 air quality records.
# 8 columns = 8 attributes.
#
# Actual values depend on the dataset.


# print("\nDataset Information:")
# Displays the dataset information heading.


# df.info()
# Displays:
# Column names.
# Number of non-missing values.
# Data types.
# Memory usage.
#
# It helps identify missing values
# and incorrect data types.


# print("\nStatistical Summary:")
# Displays the statistical summary heading.


# print(df.describe(include="all"))
# Generates descriptive statistics for all columns.
#
# For numerical columns:
# count = Number of non-missing values.
# mean = Average value.
# std = Standard deviation.
# min = Minimum value.
# max = Maximum value.
# 25%, 50%, 75% = Quartiles.
#
# include="all":
# Includes numerical and categorical columns.


# print("\nMissing Values:")
# Displays the missing-value heading.


# print(df.isnull().sum())
# isnull() identifies missing values.
#
# sum() counts missing values in each column.
#
# This helps identify incomplete records.


# ============================================================
# 4. CLEANING COLUMN NAMES
# ============================================================

# df.columns = (
#     df.columns
#     .str.strip()
#     .str.lower()
#     .str.replace(" ", "_")
#     .str.replace(".", "", regex=False)
# )
#
# Standardizes all column names.
#
# df.columns:
# Accesses the names of all columns.
#
# .str.strip():
# Removes unwanted spaces from the beginning
# and end of column names.
#
# .str.lower():
# Converts all column names into lowercase.
#
# Example:
# "AQI" becomes "aqi".
#
# .str.replace(" ", "_"):
# Replaces spaces with underscores.
#
# Example:
# "Air Quality Index" becomes "air_quality_index".
#
# .str.replace(".", "", regex=False):
# Removes full stops from column names.
#
# regex=False:
# Treats the full stop as a normal character
# instead of a regular expression pattern.
#
# This makes column names consistent
# and easier to access in Python.


# print("\nColumn Names:")
# Displays the column names heading.


# print(df.columns.tolist())
# Converts column names into a Python list
# and displays them.


# ============================================================
# 5. IDENTIFYING IMPORTANT COLUMNS
# ============================================================

# date_candidates = [
#     "date",
#     "datetime",
#     "timestamp"
# ]
#
# Creates a list of possible date column names.
#
# Different datasets may use different names
# for the date column.


# aqi_candidates = [
#     "aqi",
#     "air_quality_index"
# ]
#
# Creates a list of possible AQI column names.
#
# AQI stands for Air Quality Index.
#
# It represents the overall air quality level.


# date_column = next(
#     (col for col in date_candidates if col in df.columns),
#     None
# )
#
# Searches for the first matching date column.
#
# next():
# Returns the first available matching value.
#
# if col in df.columns:
# Checks whether the candidate column exists.
#
# If a matching column is found:
# Its name is stored in date_column.
#
# Otherwise:
# None is returned.


# aqi_column = next(
#     (col for col in aqi_candidates if col in df.columns),
#     None
# )
#
# Searches for the first matching AQI column.
#
# The selected column name is stored in aqi_column.
#
# If no matching column exists:
# None is returned.


# if date_column is None or aqi_column is None:
# Checks whether the date column or AQI column is missing.
#
# or:
# Means either one of the conditions can be True.


#     raise ValueError(
#         "Dataset must contain Date and AQI columns."
#     )
#
# Raises an error and stops the program
# if the required columns are not available.
#
# The program requires both date and AQI
# to perform its analysis.


# ============================================================
# 6. CONVERTING DATE AND NUMERICAL COLUMNS
# ============================================================

# df[date_column] = pd.to_datetime(
#     df[date_column],
#     errors="coerce"
# )
#
# Converts the date column into Pandas datetime format.
#
# pd.to_datetime():
# Converts date values into date and time format.
#
# errors="coerce":
# Converts invalid date values into NaT.
#
# NaT means Not a Time.
#
# This allows date-based operations
# such as sorting and extracting year or month.


# ============================================================
# 7. IDENTIFYING POLLUTANT COLUMNS
# ============================================================

# pollutant_candidates = [
#     "pm25",
#     "pm10",
#     "co",
#     "no2",
#     "so2",
#     "o3"
# ]
#
# Creates a list of possible pollutant column names.
#
# PM2.5 = Fine particulate matter.
# PM10 = Particulate matter.
# CO = Carbon Monoxide.
# NO2 = Nitrogen Dioxide.
# SO2 = Sulfur Dioxide.
# O3 = Ozone.


# pollutants = [
#     col for col in pollutant_candidates
#     if col in df.columns
# ]
#
# Checks which pollutant columns are available
# in the dataset.
#
# Only existing pollutant columns are selected.
#
# The selected column names are stored in pollutants.
#
# This makes the program flexible.
# It can work even if some pollutants are missing.


# numeric_columns = [aqi_column] + pollutants
#
# Creates a list containing the AQI column
# and all available pollutant columns.
#
# Example:
# ["aqi", "pm25", "pm10", "no2", "so2"].
#
# These columns must contain numerical values.


# ============================================================
# 8. CONVERTING NUMERICAL COLUMNS
# ============================================================

# for column in numeric_columns:
# Starts a loop through the AQI and pollutant columns.


#     df[column] = pd.to_numeric(
#         df[column],
#         errors="coerce"
#     )
#
# Converts the selected columns into numerical format.
#
# pd.to_numeric():
# Converts values into numbers.
#
# errors="coerce":
# Converts invalid values into NaN.
#
# Example:
# "45.5" becomes 45.5.
#
# An invalid value such as "unknown"
# becomes NaN.
#
# Numerical values are required for
# statistical calculations and graphs.


# ============================================================
# 9. HANDLING MISSING VALUES
# ============================================================

# df = df.dropna(
#     subset=[date_column, aqi_column]
# )
#
# Removes records where the date or AQI is missing.
#
# subset:
# Specifies the columns to check.
#
# A valid date and AQI are necessary
# for the main air quality analysis.
#
# Therefore, incomplete records are removed.


# ============================================================
# 10. FILLING MISSING POLLUTANT VALUES
# ============================================================

# for column in pollutants:
# Starts a loop through all available pollutant columns.


#     df[column] = df[column].fillna(
#         df[column].median()
#     )
#
# Fills missing pollutant values using the median.
#
# median():
# Finds the middle value after arranging
# numerical values in ascending order.
#
# fillna():
# Replaces missing values with the median.
#
# Median is less affected by extreme values
# than the mean.
#
# This helps retain records instead of deleting
# every row containing a missing pollutant value.


# ============================================================
# 11. REMOVING DUPLICATE RECORDS
# ============================================================

# df = df.drop_duplicates()
#
# Removes duplicate rows from the dataset.
#
# drop_duplicates():
# Keeps the first occurrence and removes repeated rows.
#
# This prevents the same air quality observation
# from being counted multiple times.


# ============================================================
# 12. SORTING RECORDS BY DATE
# ============================================================

# df = df.sort_values(date_column)
#
# Sorts the dataset according to the date column.
#
# sort_values():
# Arranges records in ascending order by default.
#
# The earliest date appears first.
# The latest date appears last.
#
# Sorting is important for time-series analysis
# because it displays changes in chronological order.


# print("\nCleaned Dataset:")
# Displays the cleaned dataset heading.


# print(df.head())
# Displays the first five records
# after cleaning and sorting.


# print("\nAvailable Pollutants:")
# Displays the pollutant heading.


# print(pollutants)
# Displays the pollutant columns available
# in the dataset.


# ============================================================
# 13. OVERALL AQI TREND USING LINE PLOT
# ============================================================

# plt.figure(figsize=(10, 5))
# Creates a new graph.
#
# figsize=(10, 5):
# Sets graph width to 10 inches
# and height to 5 inches.


# plt.plot(
#     df[date_column],
#     df[aqi_column],
#     marker="o",
#     color="blue",
#     label="AQI"
# )
#
# Creates a line graph showing AQI changes over time.
#
# df[date_column]:
# Represents the X-axis.
#
# df[aqi_column]:
# Represents the Y-axis.
#
# marker="o":
# Displays circular markers for each observation.
#
# color="blue":
# Sets the line color to blue.
#
# label="AQI":
# Gives the line a name for the legend.
#
# The graph helps us understand
# how air quality changes over time.


# plt.title("Overall AQI Trend Over Time")
# Sets the graph title.


# plt.xlabel("Date")
# Labels the X-axis as Date.


# plt.ylabel("Air Quality Index")
# Labels the Y-axis as Air Quality Index.


# plt.legend()
# Displays the legend.
#
# It identifies the plotted AQI line.


# plt.xticks(rotation=45)
# Rotates date labels by 45 degrees.
#
# This makes date labels easier to read.


# plt.grid(True)
# Displays grid lines on the graph.
#
# Grid lines help compare AQI values.


# plt.tight_layout()
# Automatically adjusts the graph layout
# to prevent labels from being cut off.


# plt.show()
# Displays the AQI line graph.


# ============================================================
# 14. INDIVIDUAL POLLUTANT TREND PLOTS
# ============================================================

# for pollutant in pollutants:
# Repeats the plotting process for each available pollutant.
#
# Example:
# First PM2.5.
# Then PM10.
# Then CO.
# Then NO2, and so on.


#     plt.figure(figsize=(10, 5))
# Creates a separate graph for the current pollutant.


#     plt.plot(
#         df[date_column],
#         df[pollutant],
#         marker="o",
#         label=pollutant.upper()
#     )
#
# Creates a line graph for the selected pollutant.
#
# X-axis:
# Date.
#
# Y-axis:
# Pollutant concentration.
#
# marker="o":
# Displays circular markers.
#
# pollutant.upper():
# Converts the pollutant name into uppercase.
#
# Example:
# pm25 becomes PM25.


#     plt.title(
#         pollutant.upper() + " Levels Over Time"
#     )
#
# Sets the title according to the pollutant.
#
# Example:
# PM25 Levels Over Time.


#     plt.xlabel("Date")
# Labels the X-axis as Date.


#     plt.ylabel("Pollutant Level")
# Labels the Y-axis as Pollutant Level.


#     plt.legend()
# Displays the legend.


#     plt.xticks(rotation=45)
# Rotates date labels by 45 degrees.


#     plt.grid(True)
# Displays grid lines.


#     plt.tight_layout()
# Adjusts the graph layout.


#     plt.show()
# Displays the current pollutant graph.
#
# A separate graph is created for every pollutant.


# ============================================================
# 15. BAR PLOT TO COMPARE AQI VALUES
# ============================================================

# plt.figure(figsize=(10, 5))
# Creates a new graph with width 10 and height 5.


# plt.bar(
#     df[date_column].dt.strftime("%Y-%m-%d"),
#     df[aqi_column],
#     color="orange"
# )
#
# Creates a bar graph comparing AQI values across dates.
#
# df[date_column].dt:
# Accesses datetime properties.
#
# strftime("%Y-%m-%d"):
# Converts dates into readable text.
#
# %Y = Year.
# %m = Month.
# %d = Day.
#
# df[aqi_column]:
# Provides the height of each bar.
#
# color="orange":
# Sets the bars to orange.
#
# The graph helps compare AQI values
# on different dates.


# plt.title("AQI Comparison Across Dates")
# Sets the graph title.


# plt.xlabel("Date")
# Labels the X-axis.


# plt.ylabel("AQI Value")
# Labels the Y-axis.


# plt.xticks(rotation=45)
# Rotates date labels by 45 degrees.


# plt.tight_layout()
# Adjusts the graph layout.


# plt.show()
# Displays the AQI comparison bar graph.


# ============================================================
# 16. BOX PLOT FOR AQI DISTRIBUTION
# ============================================================

# plt.figure(figsize=(7, 5))
# Creates a new graph.


# plt.boxplot(
#     df[aqi_column].dropna(),
#     patch_artist=True
# )
#
# Creates a box plot for AQI values.
#
# dropna():
# Removes missing values before plotting.
#
# patch_artist=True:
# Allows the box area to be filled with color.
#
# A box plot displays:
# Minimum value excluding outliers.
# First quartile (Q1).
# Median (Q2).
# Third quartile (Q3).
# Maximum value excluding outliers.
# Potential outliers.
#
# It helps understand the spread
# and distribution of AQI values.


# plt.title("Distribution of AQI Values")
# Sets the graph title.


# plt.ylabel("AQI")
# Labels the Y-axis.


# plt.grid(True)
# Displays grid lines.


# plt.tight_layout()
# Adjusts the graph layout.


# plt.show()
# Displays the AQI box plot.


# ============================================================
# 17. BOX PLOT FOR POLLUTANT DISTRIBUTIONS
# ============================================================

# if pollutants:
# Checks whether at least one pollutant column is available.
#
# If no pollutant columns exist,
# the box plot is skipped.


#     plt.figure(figsize=(9, 5))
# Creates a new graph.


#     plt.boxplot(
#         [df[col].dropna() for col in pollutants],
#         tick_labels=[col.upper() for col in pollutants]
#     )
#
# Creates box plots for all available pollutants.
#
# [df[col].dropna() for col in pollutants]:
# Collects the values of each pollutant.
#
# dropna():
# Removes missing values.
#
# tick_labels:
# Provides pollutant names on the X-axis.
#
# Example:
# PM25, PM10, CO, NO2.
#
# This allows us to compare pollutant distributions.


#     plt.title("Distribution of Pollutant Levels")
# Sets the graph title.


#     plt.xlabel("Pollutants")
# Labels the X-axis.


#     plt.ylabel("Pollutant Levels")
# Labels the Y-axis.


#     plt.grid(True)
# Displays grid lines.


#     plt.tight_layout()
# Adjusts the graph layout.


#     plt.show()
# Displays the pollutant box plots.


# ============================================================
# 18. SCATTER PLOTS: AQI AND POLLUTANT RELATIONSHIP
# ============================================================

# for pollutant in pollutants:
# Repeats the scatter plot process
# for every available pollutant.


#     plt.figure(figsize=(7, 5))
# Creates a new graph.


#     plt.scatter(
#         df[pollutant],
#         df[aqi_column],
#         color="green",
#         alpha=0.7
#     )
#
# Creates a scatter plot.
#
# X-axis:
# Pollutant concentration.
#
# Y-axis:
# AQI value.
#
# color="green":
# Sets the points to green.
#
# alpha=0.7:
# Controls transparency.
#
# alpha ranges from 0 to 1.
# 0 means fully transparent.
# 1 means fully opaque.
#
# The scatter plot helps identify
# whether AQI changes with pollutant levels.


#     plt.title(
#         "AQI vs " + pollutant.upper()
#     )
#
# Sets the graph title according to the pollutant.
#
# Example:
# AQI vs PM25.


#     plt.xlabel(pollutant.upper() + " Level")
# Labels the X-axis with the pollutant name.


#     plt.ylabel("AQI")
# Labels the Y-axis as AQI.


#     plt.grid(True)
# Displays grid lines.


#     plt.tight_layout()
# Adjusts the graph layout.


#     plt.show()
# Displays the scatter plot.


# ============================================================
# 19. BUBBLE CHART FOR AQI AND PM2.5
# ============================================================

# if "pm25" in df.columns:
# Checks whether the PM2.5 column exists.
#
# If PM2.5 is unavailable,
# the bubble chart is skipped.


#     plt.figure(figsize=(8, 5))
# Creates a new graph.


#     plt.scatter(
#         df["pm25"],
#         df[aqi_column],
#         s=df[aqi_column].clip(lower=1) * 2,
#         c=df[aqi_column],
#         cmap="viridis",
#         alpha=0.6
#     )
#
# Creates a bubble chart.
#
# X-axis:
# PM2.5 concentration.
#
# Y-axis:
# AQI value.
#
# s:
# Controls the size of each bubble.
#
# df[aqi_column].clip(lower=1) * 2:
# Uses AQI values to calculate bubble sizes.
#
# clip(lower=1):
# Ensures that the bubble size calculation
# does not use a value below 1.
#
# c=df[aqi_column]:
# Uses AQI values to determine bubble colors.
#
# cmap="viridis":
# Selects the Viridis color map.
#
# alpha=0.6:
# Makes bubbles partially transparent.
#
# Larger bubbles represent higher AQI values.
# Bubble color also represents AQI values.


#     plt.title("Bubble Chart: PM2.5 vs AQI")
# Sets the graph title.


#     plt.xlabel("PM2.5 Level")
# Labels the X-axis.


#     plt.ylabel("AQI")
# Labels the Y-axis.


#     plt.colorbar(label="AQI")
# Displays a color bar.
#
# It explains how colors correspond to AQI values.


#     plt.tight_layout()
# Adjusts the graph layout.


#     plt.show()
# Displays the bubble chart.


# ============================================================
# 20. DISPLAYING AQI STATISTICS
# ============================================================

# print("\nAQI Statistics:")
# Displays the AQI statistics heading.


# print("Maximum AQI:", df[aqi_column].max())
# Finds and displays the highest AQI value.
#
# max():
# Returns the maximum value in the column.


# print("Minimum AQI:", df[aqi_column].min())
# Finds and displays the lowest AQI value.
#
# min():
# Returns the minimum value in the column.


# print("Average AQI:", df[aqi_column].mean())
# Calculates and displays the average AQI.
#
# mean():
# Adds all AQI values and divides by their count.


# ============================================================
# 21. DISPLAYING POLLUTANT STATISTICS
# ============================================================

# print("\nPollutant Statistics:")
# Displays the pollutant statistics heading.


# print(df[pollutants].describe())
# Displays descriptive statistics
# for all available pollutant columns.
#
# It shows:
# Count.
# Mean.
# Standard deviation.
# Minimum.
# 25th percentile.
# Median.
# 75th percentile.
# Maximum.
#
# This helps understand pollutant levels
# in the dataset.


# ============================================================
# 22. CORRELATION ANALYSIS
# ============================================================

# if pollutants:
# Checks whether pollutant columns are available.
#
# Correlation analysis is skipped
# if no pollutant columns exist.


#     correlation = df[
#         [aqi_column] + pollutants
#     ].corr()
#
# Selects AQI and all available pollutant columns.
#
# corr():
# Calculates the correlation between numerical columns.
#
# Correlation values range from -1 to +1.
#
# +1 = Perfect positive correlation.
# -1 = Perfect negative correlation.
#  0 = No linear correlation.
#
# The result is stored in correlation.


#     print("\nCorrelation Matrix:")
# Displays the correlation matrix heading.


#     print(correlation)
# Displays the correlation values.
#
# The matrix helps identify relationships
# between AQI and different pollutants.


# ============================================================
# 23. CORRELATION HEATMAP
# ============================================================

#     plt.figure(figsize=(8, 6))
# Creates a new graph.


#     plt.imshow(
#         correlation,
#         cmap="coolwarm",
#         aspect="auto"
#     )
#
# Displays the correlation matrix as a heatmap.
#
# imshow():
# Converts matrix values into colored cells.
#
# cmap="coolwarm":
# Uses different colors for positive and negative values.
#
# aspect="auto":
# Automatically adjusts the cell aspect ratio.
#
# The heatmap makes relationships easier to understand.


#     plt.colorbar(label="Correlation")
# Displays a color bar representing correlation values.


#     plt.xticks(
#         range(len(correlation.columns)),
#         correlation.columns,
#         rotation=45
#     )
#
# Sets the X-axis labels.
#
# range():
# Creates positions for the column labels.
#
# correlation.columns:
# Provides the column names.
#
# rotation=45:
# Rotates labels by 45 degrees.


#     plt.yticks(
#         range(len(correlation.columns)),
#         correlation.columns
#     )
#
# Sets the Y-axis labels.
#
# The same column names are used
# for both axes because the matrix compares
# every variable with every other variable.


#     plt.title("AQI and Pollutant Correlation")
# Sets the heatmap title.


#     plt.tight_layout()
# Adjusts the graph layout.


#     plt.show()
# Displays the correlation heatmap.


# ============================================================
# 24. EXPORTING THE CLEANED DATASET
# ============================================================

# df.to_csv(
#     "cleaned_air_quality.csv",
#     index=False
# )
#
# Saves the cleaned dataset into a CSV file.
#
# Output file:
# cleaned_air_quality.csv
#
# It contains cleaned and sorted air quality records.
#
# index=False:
# Prevents the DataFrame index from being saved
# as an additional column.


# print("\nCleaned Dataset Saved Successfully!")
# Displays a message confirming that the cleaned
# dataset has been saved.


# print("\nAir Quality Visualization Completed Successfully!")
# Displays the final completion message.
#
# This indicates that the analysis and visualization
# operations have finished.
