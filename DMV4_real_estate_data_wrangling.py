
# AIM: Perform Data Wrangling on Real Estate Market Dataset.

# Installation commands:
# Windows:
# python -m pip install pandas numpy matplotlib scikit-learn
# python DMV4_real_estate_data_wrangling.py
#
# Ubuntu/Linux:
# pip install pandas numpy matplotlib scikit-learn
# python3 DMV4_real_estate_data_wrangling.py


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os

# 1. Import dataset

df = pd.read_csv("RealEstate_Prices.csv")

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

df.columns = [
    re.sub(
        r"_+",
        "_",
        re.sub(
            r"[^a-zA-Z0-9]+",
            "_",
            column.strip().lower()
        )
    ).strip("_")
    for column in df.columns
]

print("\nCleaned Column Names:")

print(df.columns.tolist())

# 4. Remove duplicate records

df = df.drop_duplicates()

# 5. Standardize text columns

for column in df.columns:

    if pd.api.types.is_string_dtype(df[column].dtype):

        df[column] = df[column].str.strip()

        df[column] = df[column].replace("", np.nan)

# 6. Identify important columns

price_candidates = [
    "sale_price",
    "price",
    "saleprice",
    "selling_price",
    "house_price",
    "listprice"
]

price_column = next(
    (
        column for column in price_candidates
        if column in df.columns
    ),
    None
)

if price_column is None:

    raise ValueError(
        "Price column not found. Check the dataset column names."
    )

# Convert price column into numeric format

df[price_column] = pd.to_numeric(
    df[price_column],
    errors="coerce"
)

# Identify date column

date_candidates = [
    "sale_date",
    "date",
    "transaction_date",
    "listingdate"
]

date_column = next(
    (
        column for column in date_candidates
        if column in df.columns
    ),
    None
)

if date_column is not None:

    df[date_column] = pd.to_datetime(
        df[date_column],
        errors="coerce"
    )

# 7. Handle missing values

# Remove records where sale price is missing

df = df.dropna(subset=[price_column])

# Fill missing numerical values with median

numeric_columns = df.select_dtypes(
    include=np.number
).columns

for column in numeric_columns:

    df[column] = df[column].fillna(
        df[column].median()
    )

# Fill missing categorical values with mode

categorical_columns = [
    column for column in df.columns
    if pd.api.types.is_string_dtype(df[column].dtype)
]

for column in categorical_columns:

    if not df[column].mode().empty:

        df[column] = df[column].fillna(
            df[column].mode()[0]
        )

# Fill missing dates by removing records with invalid dates

if date_column is not None:

    df = df.dropna(subset=[date_column])

# 8. Merge additional dataset if available

if os.path.exists("Neighborhood_Info.csv"):

    extra_df = pd.read_csv("Neighborhood_Info.csv")

    extra_df.columns = [
        re.sub(
            r"_+",
            "_",
            re.sub(
                r"[^a-zA-Z0-9]+",
                "_",
                column.strip().lower()
            )
        ).strip("_")
        for column in extra_df.columns
    ]

    if "neighborhood" in df.columns and "neighborhood" in extra_df.columns:

        df = pd.merge(
            df,
            extra_df,
            on="neighborhood",
            how="left"
        )

        print("\nAdditional Neighborhood Data Merged.")

else:

    print("\nNo additional dataset found. Merge skipped.")

# 9. Filter and subset data

# Select a particular location if required

selected_location = None

if selected_location is not None and "location" in df.columns:

    df = df[
        df["location"].str.lower()
        == selected_location.lower()
    ]

# Select a particular property type if required

selected_property_type = None

if selected_property_type is not None and "property_type" in df.columns:

    df = df[
        df["property_type"].str.lower()
        == selected_property_type.lower()
    ]

# Filter by date if date column exists

if date_column is not None:

    df["year"] = df[date_column].dt.year

    selected_year = None

    if selected_year is not None:

        df = df[
            df["year"] == selected_year
        ]

# 10. Feature engineering

# Calculate price per square foot if area is available

area_candidates = [
    "area",
    "square_feet",
    "sqft",
    "area_sqft",
    "gr_liv_area"
]

area_column = next(
    (
        column for column in area_candidates
        if column in df.columns
    ),
    None
)

if area_column is not None:

    df[area_column] = pd.to_numeric(
        df[area_column],
        errors="coerce"
    )

    df["price_per_sqft"] = (
        df[price_column] / df[area_column].replace(0, np.nan)
    )

# 11. Identify and handle outliers using IQR

Q1 = df[price_column].quantile(0.25)

Q3 = df[price_column].quantile(0.75)

IQR = Q3 - Q1

lower_limit = Q1 - 1.5 * IQR

upper_limit = Q3 + 1.5 * IQR

print("\nOutlier Limits:")

print("Lower Limit:", lower_limit)

print("Upper Limit:", upper_limit)

outliers = df[
    (df[price_column] < lower_limit)
    |
    (df[price_column] > upper_limit)
]

print("\nNumber of Outliers:")

print(len(outliers))

# Cap extreme prices within IQR limits

df[price_column] = df[price_column].clip(
    lower_limit,
    upper_limit
)

# 12. Aggregate data

if "neighborhood" in df.columns:

    neighborhood_summary = df.groupby(
        "neighborhood"
    )[price_column].agg(
        ["mean", "min", "max", "count"]
    )

    print("\nAverage Price by Neighborhood:")

    print(neighborhood_summary)

    neighborhood_summary.to_csv(
        "neighborhood_summary.csv"
    )

if "property_type" in df.columns:

    property_summary = df.groupby(
        "property_type"
    )[price_column].agg(
        ["mean", "min", "max", "count"]
    )

    print("\nAverage Price by Property Type:")

    print(property_summary)

    property_summary.to_csv(
        "property_type_summary.csv"
    )

# 13. Encode categorical variables

encoded_df = pd.get_dummies(
    df,
    drop_first=True,
    dtype=int
)

print("\nEncoded Dataset:")

print(encoded_df.head())

# 14. Export cleaned and encoded datasets

df.to_csv(
    "cleaned_real_estate.csv",
    index=False
)

encoded_df.to_csv(
    "encoded_real_estate.csv",
    index=False
)

# 15. Display final results

print("\nFinal Dataset:")

print(df.head())

print("\nFinal Dataset Shape:")

print(df.shape)

print("\nMissing Values After Cleaning:")

print(df.isnull().sum())

print("\nData Wrangling Completed Successfully!")

# 16. Visualize average price by neighborhood

if "neighborhood" in df.columns:

    neighborhood_prices = df.groupby(
        "neighborhood"
    )[price_column].mean()

    plt.figure(figsize=(9, 5))

    neighborhood_prices.plot(kind="bar")

    plt.title("Average House Price by Neighborhood")

    plt.xlabel("Neighborhood")

    plt.ylabel("Average Sale Price")

    plt.xticks(rotation=45)

    plt.tight_layout()

    plt.show()

# 17. Visualize average price by property type

if "property_type" in df.columns:

    property_prices = df.groupby(
        "property_type"
    )[price_column].mean()

    plt.figure(figsize=(8, 5))

    property_prices.plot(kind="bar")

    plt.title("Average House Price by Property Type")

    plt.xlabel("Property Type")

    plt.ylabel("Average Sale Price")

    plt.tight_layout()

    plt.show()

# 18. Visualize price distribution

plt.figure(figsize=(8, 5))

plt.hist(
    df[price_column],
    bins=10,
    edgecolor="black"
)

plt.title("Real Estate Price Distribution")

plt.xlabel("Sale Price")

plt.ylabel("Frequency")

plt.tight_layout()

plt.show()
# ============================================================
# AIM:
# Perform Data Wrangling and Analysis on Real Estate Prices.
#
# Steps:
# 1. Load and explore the dataset.
# 2. Clean column names.
# 3. Remove duplicate records.
# 4. Standardize text columns.
# 5. Identify important columns.
# 6. Handle missing values.
# 7. Merge additional datasets.
# 8. Filter and subset data.
# 9. Perform feature engineering.
# 10. Detect and handle outliers using IQR.
# 11. Perform data aggregation.
# 12. Encode categorical variables.
# 13. Export cleaned and encoded datasets.
# 14. Visualize real estate prices.
# ============================================================


# ============================================================
# 1. INSTALLATION AND EXECUTION COMMANDS
# ============================================================

# Windows:
# python -m pip install pandas numpy matplotlib
# python practical_realestate.py

# Ubuntu/Linux:
# python3 -m pip install pandas numpy matplotlib
# python3 practical_realestate.py

# Keep RealEstate_Prices.csv in the same folder
# as the Python program.

# Optional:
# Keep Neighborhood_Info.csv in the same folder
# if additional neighborhood information is available.


# ============================================================
# 2. IMPORTING LIBRARIES
# ============================================================

# import pandas as pd
# Imports the Pandas library.
# It is used to load, clean, transform and analyze datasets.
# pd is the short name used for Pandas.


# import numpy as np
# Imports the NumPy library.
# It is used for numerical calculations.
# It also provides special values such as np.nan and np.inf.


# import matplotlib.pyplot as plt
# Imports the Pyplot module from Matplotlib.
# It is used to create graphs and visualizations.
# plt is the short name used for Pyplot.


# import re
# Imports the Regular Expression library.
# It is used to search, match and replace text patterns.
# Here, it is used to clean column names.


# import os
# Imports the Operating System library.
# It provides functions for interacting with files and folders.
# Here, it checks whether an additional CSV file exists.


# ============================================================
# 3. IMPORTING THE DATASET
# ============================================================

# df = pd.read_csv("RealEstate_Prices.csv")
# Reads the real estate dataset from a CSV file.
#
# pd.read_csv():
# Loads CSV data into a Pandas DataFrame.
#
# df:
# Stores the complete dataset.
#
# The dataset may contain information such as:
# Property ID.
# Sale price.
# Property type.
# Neighborhood.
# Location.
# Area.
# Sale date.
# Number of bedrooms.
# Number of bathrooms.


# print("First Five Records:")
# Displays a heading for the first five records.


# print(df.head())
# Displays the first five rows of the dataset.
#
# head() displays five records by default.
# It helps us understand the dataset structure.


# ============================================================
# 4. EXPLORING THE DATASET
# ============================================================

# print("\nDataset Shape:")
# Displays the dataset dimensions heading.


# print(df.shape)
# Displays the number of rows and columns.
#
# Example:
# (1000, 10)
#
# This means:
# 1000 rows = 1000 property records.
# 10 columns = 10 attributes.
#
# Actual values depend on the dataset.


# print("\nDataset Information:")
# Displays the dataset information heading.


# df.info()
# Displays:
# Column names.
# Number of non-null values.
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
# For categorical columns:
# count = Number of non-missing values.
# unique = Number of unique values.
# top = Most frequent value.
# freq = Frequency of the most frequent value.
#
# include="all" includes numerical and categorical columns.


# print("\nMissing Values:")
# Displays the missing-value heading.


# print(df.isnull().sum())
# isnull() identifies missing values.
#
# sum() counts the missing values in each column.
#
# This helps identify incomplete property records.


# ============================================================
# 5. CLEANING COLUMN NAMES
# ============================================================

# df.columns = [
#     re.sub(
#         r"_+",
#         "_",
#         re.sub(
#             r"[^a-zA-Z0-9]+",
#             "_",
#             column.strip().lower()
#         )
#     ).strip("_")
#     for column in df.columns
# ]
#
# Cleans and standardizes all column names.
#
# df.columns:
# Accesses the names of all columns.
#
# column.strip():
# Removes unwanted spaces from the beginning
# and end of a column name.
#
# .lower():
# Converts the column name into lowercase.
#
# Example:
# "Sale Price" becomes "sale price".
#
# re.sub(r"[^a-zA-Z0-9]+", "_", ...):
# Replaces one or more characters that are not
# English letters or digits with an underscore.
#
# Example:
# "Sale Price ($)" becomes "sale_price_".
#
# re.sub(r"_+", "_", ...):
# Replaces multiple consecutive underscores
# with a single underscore.
#
# .strip("_"):
# Removes underscores from the beginning and end.
#
# Final example:
# " Sale Price ($) " becomes "sale_price".
#
# for column in df.columns:
# Repeats the cleaning operation for every column.
#
# This entire expression is a list comprehension.
# It creates a new list of cleaned column names.


# print("\nCleaned Column Names:")
# Displays the heading for cleaned column names.


# print(df.columns.tolist())
# Converts the column names into a Python list
# and displays them.


# ============================================================
# 6. REMOVING DUPLICATE RECORDS
# ============================================================

# df = df.drop_duplicates()
# Removes duplicate rows from the dataset.
#
# drop_duplicates():
# Keeps the first occurrence and removes repeated rows.
#
# This prevents the same property record
# from being counted multiple times.


# ============================================================
# 7. STANDARDIZING TEXT COLUMNS
# ============================================================

# for column in df.columns:
# Starts a loop through every column in the dataset.


#     if pd.api.types.is_string_dtype(df[column].dtype):
# Checks whether the current column contains string data.
#
# Examples:
# Neighborhood.
# Property type.
# Location.


#         df[column] = df[column].str.strip()
# Removes unwanted spaces from the beginning
# and end of text values.
#
# Example:
# " Pune " becomes "Pune".


#         df[column] = df[column].replace("", np.nan)
# Replaces empty strings with NaN.
#
# NaN represents a missing value.
#
# This makes empty text values easier to identify
# and handle during data cleaning.


# ============================================================
# 8. IDENTIFYING THE PRICE COLUMN
# ============================================================

# price_candidates = [
#     "sale_price",
#     "price",
#     "saleprice",
#     "selling_price",
#     "house_price",
#     "listprice"
# ]
#
# Creates a list of possible names for the property price column.
#
# Different datasets may use different column names.
#
# Examples:
# sale_price.
# price.
# selling_price.
# house_price.
#
# This makes the program more flexible.


# price_column = next(
#     (
#         column for column in price_candidates
#         if column in df.columns
#     ),
#     None
# )
#
# Finds the first matching price column.
#
# next():
# Returns the next available value from an iterator.
#
# The expression checks each candidate column name.
#
# if column in df.columns:
# Checks whether that column exists in the dataset.
#
# If a matching column is found:
# Its name is stored in price_column.
#
# If no matching column is found:
# None is returned.


# if price_column is None:
# Checks whether no price column was found.


#     raise ValueError(
#         "Price column not found. Check the dataset column names."
#     )
#
# Displays an error and stops the program
# if the dataset does not contain a recognized price column.
#
# This is important because property price
# is the main variable used in the practical.


# ============================================================
# 9. CONVERTING PRICE COLUMN INTO NUMERIC FORMAT
# ============================================================

# df[price_column] = pd.to_numeric(
#     df[price_column],
#     errors="coerce"
# )
#
# Converts the selected price column into numerical values.
#
# pd.to_numeric():
# Converts values into numbers.
#
# errors="coerce":
# Converts invalid values into NaN.
#
# Example:
# "2500000" becomes 2500000.
#
# An invalid price value becomes NaN.
#
# This allows mathematical calculations
# to be performed on property prices.


# ============================================================
# 10. IDENTIFYING DATE COLUMN
# ============================================================

# date_candidates = [
#     "sale_date",
#     "date",
#     "transaction_date",
#     "listingdate"
# ]
#
# Creates a list of possible date column names.
#
# Different datasets may use different names
# for property transaction dates.


# date_column = next(
#     (
#         column for column in date_candidates
#         if column in df.columns
#     ),
#     None
# )
#
# Searches for the first available date column.
#
# If a matching column exists:
# Its name is stored in date_column.
#
# Otherwise:
# date_column becomes None.


# if date_column is not None:
# Checks whether a date column was found.


#     df[date_column] = pd.to_datetime(
#         df[date_column],
#         errors="coerce"
#     )
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
# This allows us to extract information such as
# year, month and day from the date.


# ============================================================
# 11. HANDLING MISSING VALUES
# ============================================================

# df = df.dropna(subset=[price_column])
# Removes records where the property price is missing.
#
# subset:
# Specifies the column to check.
#
# Records without a price cannot be used
# for price analysis.
#
# Therefore, those rows are removed.


# ============================================================
# 12. IDENTIFYING NUMERICAL COLUMNS
# ============================================================

# numeric_columns = df.select_dtypes(
#     include=np.number
# ).columns
#
# Selects all numerical columns from the dataset.
#
# select_dtypes():
# Selects columns according to their data types.
#
# include=np.number:
# Selects integer and floating-point columns.
#
# columns:
# Retrieves the names of the selected columns.
#
# These columns are used to fill missing numerical values.


# ============================================================
# 13. FILLING MISSING NUMERICAL VALUES
# ============================================================

# for column in numeric_columns:
# Starts a loop through all numerical columns.


#     df[column] = df[column].fillna(
#         df[column].median()
#     )
#
# Fills missing numerical values using the median.
#
# median():
# Finds the middle value of the numerical data
# after arranging the values in ascending order.
#
# fillna():
# Replaces missing values with the median.
#
# Median is less affected by extreme values
# than the mean.
#
# This helps preserve numerical records
# instead of deleting every row with missing values.


# ============================================================
# 14. IDENTIFYING CATEGORICAL COLUMNS
# ============================================================

# categorical_columns = [
#     column for column in df.columns
#     if pd.api.types.is_string_dtype(df[column].dtype)
# ]
#
# Finds columns containing string values.
#
# Examples:
# Neighborhood.
# Property type.
# Location.
#
# This is a list comprehension.
#
# It checks every column and stores the names
# of columns having string data types.


# ============================================================
# 15. FILLING MISSING CATEGORICAL VALUES
# ============================================================

# for column in categorical_columns:
# Starts a loop through all categorical columns.


#     if not df[column].mode().empty:
# Checks whether the column has a mode value.
#
# mode():
# Finds the most frequently occurring value.
#
# empty:
# Checks whether the result contains no values.


#         df[column] = df[column].fillna(
#             df[column].mode()[0]
#         )
#
# Fills missing categorical values
# with the most frequently occurring category.
#
# mode()[0]:
# Selects the first mode value.
#
# Example:
# If Pune occurs most frequently in the Location column,
# missing location values are replaced with Pune.


# ============================================================
# 16. HANDLING MISSING DATES
# ============================================================

# if date_column is not None:
# Checks whether a date column exists.


#     df = df.dropna(subset=[date_column])
# Removes rows with missing or invalid dates.
#
# Invalid dates were converted into NaT earlier.
#
# These records are removed because
# date-based analysis requires valid dates.


# ============================================================
# 17. MERGING ADDITIONAL DATASET
# ============================================================

# if os.path.exists("Neighborhood_Info.csv"):
# Checks whether Neighborhood_Info.csv exists
# in the current working directory.
#
# os.path.exists():
# Returns True if the specified file exists.
#
# This allows the program to merge additional information
# only when the file is available.


#     extra_df = pd.read_csv("Neighborhood_Info.csv")
# Reads the additional neighborhood dataset.
#
# Stores it in a separate DataFrame called extra_df.
#
# This file may contain information such as:
# Neighborhood name.
# Average income.
# Nearby facilities.
# Schools.
# Hospitals.
# Transport facilities.


#     extra_df.columns = [
#         re.sub(
#             r"_+",
#             "_",
#             re.sub(
#                 r"[^a-zA-Z0-9]+",
#                 "_",
#                 column.strip().lower()
#             )
#         ).strip("_")
#         for column in extra_df.columns
#     ]
#
# Cleans the column names of the additional dataset.
#
# It performs the same standardization
# used for the main dataset.
#
# This is important because both datasets
# must have matching column names for merging.


#     if "neighborhood" in df.columns and "neighborhood" in extra_df.columns:
# Checks whether both datasets contain
# a common neighborhood column.
#
# The neighborhood column acts as the joining key.


#         df = pd.merge(
#             df,
#             extra_df,
#             on="neighborhood",
#             how="left"
#         )
#
# Merges the main and additional datasets.
#
# pd.merge():
# Combines two DataFrames using a common column.
#
# df:
# Main real estate dataset.
#
# extra_df:
# Additional neighborhood dataset.
#
# on="neighborhood":
# Uses neighborhood as the common matching column.
#
# how="left":
# Performs a left join.
#
# A left join keeps all records from the main dataset.
#
# Matching information from extra_df is added
# wherever a matching neighborhood is found.
#
# If no match is found, the additional columns
# contain missing values.
#
# This helps combine property details
# with neighborhood information.


#         print("\nAdditional Neighborhood Data Merged.")
# Displays a message when the datasets are merged.


# else:
# Executes when the additional CSV file does not exist.


#     print("\nNo additional dataset found. Merge skipped.")
# Displays a message that the merging operation was skipped.


# Important:
# If the file exists but either dataset does not contain
# the neighborhood column, no merge takes place.
# In that situation, this program does not print
# the "merge skipped" message.


# ============================================================
# 18. FILTERING AND SUBSETTING DATA
# ============================================================

# selected_location = None
# Defines an optional location filter.
#
# None means no particular location is selected.
#
# A location can be specified, for example:
# selected_location = "Pune"


# if selected_location is not None and "location" in df.columns:
# Checks whether a location was selected
# and whether the location column exists.


#     df = df[
#         df["location"].str.lower()
#         == selected_location.lower()
#     ]
#
# Filters records according to the selected location.
#
# str.lower():
# Converts location names into lowercase.
#
# This makes the comparison case-insensitive.
#
# Example:
# "PUNE" and "Pune" are treated as the same value.
#
# Only matching property records are retained.


# ============================================================
# 19. FILTERING BY PROPERTY TYPE
# ============================================================

# selected_property_type = None
# Defines an optional property type filter.
#
# None means no property type is selected.
#
# Example:
# selected_property_type = "Apartment"


# if selected_property_type is not None and "property_type" in df.columns:
# Checks whether a property type was selected
# and whether the column exists.


#     df = df[
#         df["property_type"].str.lower()
#         == selected_property_type.lower()
#     ]
#
# Keeps only records matching the selected property type.
#
# Example:
# If Apartment is selected,
# only apartment records remain.


# ============================================================
# 20. FILTERING BY YEAR
# ============================================================

# if date_column is not None:
# Checks whether a date column is available.


#     df["year"] = df[date_column].dt.year
# Extracts the year from each date.
#
# .dt.year:
# Retrieves the year component of a datetime value.
#
# Example:
# 2025-06-15 becomes 2025.
#
# The result is stored in a new column named year.


#     selected_year = None
# Defines an optional year filter.
#
# None means all years are included.


#     if selected_year is not None:
# Checks whether a specific year was selected.


#         df = df[
#             df["year"] == selected_year
#         ]
#
# Filters records for the selected year.
#
# Example:
# If selected_year = 2025,
# only properties sold or listed in 2025 remain.


# ============================================================
# 21. FEATURE ENGINEERING
# ============================================================

# Feature engineering means creating new useful features
# from existing data to improve analysis.


# ============================================================
# 22. IDENTIFYING AREA COLUMN
# ============================================================

# area_candidates = [
#     "area",
#     "square_feet",
#     "sqft",
#     "area_sqft",
#     "gr_liv_area"
# ]
#
# Creates a list of possible names for the property area column.
#
# Different datasets may use different names
# for property size.


# area_column = next(
#     (
#         column for column in area_candidates
#         if column in df.columns
#     ),
#     None
# )
#
# Finds the first matching area column.
#
# If a matching column exists:
# Its name is stored in area_column.
#
# Otherwise:
# area_column becomes None.


# ============================================================
# 23. CALCULATING PRICE PER SQUARE FOOT
# ============================================================

# if area_column is not None:
# Checks whether a property area column was found.


#     df[area_column] = pd.to_numeric(
#         df[area_column],
#         errors="coerce"
#     )
#
# Converts property area values into numeric format.
#
# Invalid values are converted into NaN.


#     df["price_per_sqft"] = (
#         df[price_column] / df[area_column].replace(0, np.nan)
#     )
#
# Creates a new feature called price_per_sqft.
#
# Formula:
#
# Price per Square Foot =
# Property Sale Price / Property Area.
#
# Example:
# Sale Price = 50,00,000.
# Area = 1000 square feet.
#
# Price per square foot = 5000.
#
# replace(0, np.nan):
# Replaces zero area values with NaN.
#
# This prevents division by zero.
#
# Important:
# If the area column contains missing values,
# the corresponding price_per_sqft may also be missing.
# The program does not fill these newly created missing values.


# ============================================================
# 24. IDENTIFYING OUTLIERS USING IQR
# ============================================================

# Q1 = df[price_column].quantile(0.25)
# Calculates the first quartile.
#
# Q1 represents the 25th percentile.
# Approximately 25% of the property prices
# are at or below this value.


# Q3 = df[price_column].quantile(0.75)
# Calculates the third quartile.
#
# Q3 represents the 75th percentile.
# Approximately 75% of the property prices
# are at or below this value.


# IQR = Q3 - Q1
# Calculates the Interquartile Range.
#
# IQR represents the spread of the middle 50%
# of the property prices.


# lower_limit = Q1 - 1.5 * IQR
# Calculates the lower outlier limit.
#
# Values below this limit are considered
# potential lower outliers.


# upper_limit = Q3 + 1.5 * IQR
# Calculates the upper outlier limit.
#
# Values above this limit are considered
# potential upper outliers.


# ============================================================
# 25. DISPLAYING OUTLIER LIMITS
# ============================================================

# print("\nOutlier Limits:")
# Displays the outlier limits heading.


# print("Lower Limit:", lower_limit)
# Displays the calculated lower limit.


# print("Upper Limit:", upper_limit)
# Displays the calculated upper limit.


# ============================================================
# 26. IDENTIFYING OUTLIER RECORDS
# ============================================================

# outliers = df[
#     (df[price_column] < lower_limit)
#     |
#     (df[price_column] > upper_limit)
# ]
#
# Selects records containing potential price outliers.
#
# df[price_column] < lower_limit:
# Identifies prices below the lower limit.
#
# |:
# Represents the OR operator for Pandas conditions.
#
# df[price_column] > upper_limit:
# Identifies prices above the upper limit.
#
# The result contains all records
# satisfying either condition.
#
# These records are stored in outliers.


# print("\nNumber of Outliers:")
# Displays the outlier count heading.


# print(len(outliers))
# len() returns the number of outlier records.
#
# This tells us how many property prices
# fall outside the IQR limits.


# ============================================================
# 27. HANDLING OUTLIERS USING CLIPPING
# ============================================================

# df[price_column] = df[price_column].clip(
#     lower_limit,
#     upper_limit
# )
#
# Handles extreme property prices using clipping.
#
# clip():
# Replaces values outside the specified limits.
#
# Values below lower_limit become lower_limit.
#
# Values above upper_limit become upper_limit.
#
# Values inside the limits remain unchanged.
#
# This method is also called winsorization.
#
# Important:
# Outlier records are not deleted.
# Their price values are capped at the IQR boundaries.


# ============================================================
# 28. AGGREGATING DATA BY NEIGHBORHOOD
# ============================================================

# if "neighborhood" in df.columns:
# Checks whether the neighborhood column exists.


#     neighborhood_summary = df.groupby(
#         "neighborhood"
#     )[price_column].agg(
#         ["mean", "min", "max", "count"]
#     )
#
# Groups property records according to neighborhood.
#
# groupby("neighborhood"):
# Creates a group for each neighborhood.
#
# [price_column]:
# Selects the property price column for analysis.
#
# agg():
# Performs multiple aggregation operations.
#
# mean:
# Calculates the average property price.
#
# min:
# Finds the minimum property price.
#
# max:
# Finds the maximum property price.
#
# count:
# Counts the number of available price records.
#
# The result is stored in neighborhood_summary.


#     print("\nAverage Price by Neighborhood:")
# Displays the neighborhood summary heading.


#     print(neighborhood_summary)
# Displays the average, minimum, maximum
# and count of property prices for each neighborhood.


#     neighborhood_summary.to_csv(
#         "neighborhood_summary.csv"
#     )
#
# Saves the neighborhood summary into a CSV file.
#
# This file can be used for further analysis.


# ============================================================
# 29. AGGREGATING DATA BY PROPERTY TYPE
# ============================================================

# if "property_type" in df.columns:
# Checks whether the property_type column exists.


#     property_summary = df.groupby(
#         "property_type"
#     )[price_column].agg(
#         ["mean", "min", "max", "count"]
#     )
#
# Groups property records according to property type.
#
# Examples:
# Apartment.
# House.
# Villa.
#
# mean:
# Calculates the average price.
#
# min:
# Finds the minimum price.
#
# max:
# Finds the maximum price.
#
# count:
# Counts the number of property records.


#     print("\nAverage Price by Property Type:")
# Displays the property type summary heading.


#     print(property_summary)
# Displays the summary for each property type.


#     property_summary.to_csv(
#         "property_type_summary.csv"
#     )
#
# Saves the property type summary into a CSV file.


# ============================================================
# 30. ENCODING CATEGORICAL VARIABLES
# ============================================================

# encoded_df = pd.get_dummies(
#     df,
#     drop_first=True,
#     dtype=int
# )
#
# Converts categorical columns into numerical columns.
#
# Machine learning algorithms generally require
# numerical input values.
#
# pd.get_dummies():
# Performs one-hot encoding.
#
# Example:
# Property Type:
# Apartment, House, Villa.
#
# One-hot encoding creates separate columns
# representing categories.
#
# drop_first=True:
# Removes the first category from each encoded feature.
#
# dtype=int:
# Stores encoded values as integers, usually 0 and 1.
#
# The encoded dataset is stored in encoded_df.


# print("\nEncoded Dataset:")
# Displays the encoded dataset heading.


# print(encoded_df.head())
# Displays the first five records
# of the encoded dataset.


# ============================================================
# 31. EXPORTING CLEANED AND ENCODED DATASETS
# ============================================================

# df.to_csv(
#     "cleaned_real_estate.csv",
#     index=False
# )
#
# Saves the cleaned real estate dataset.
#
# It contains cleaned original columns
# and newly engineered features.
#
# index=False:
# Prevents the DataFrame index from being saved
# as an additional column.


# encoded_df.to_csv(
#     "encoded_real_estate.csv",
#     index=False
# )
#
# Saves the encoded dataset.
#
# It contains numerical representations
# of categorical variables.
#
# This dataset can be used for machine learning.


# ============================================================
# 32. DISPLAYING FINAL RESULTS
# ============================================================

# print("\nFinal Dataset:")
# Displays the final dataset heading.


# print(df.head())
# Displays the first five rows
# of the cleaned real estate dataset.


# print("\nFinal Dataset Shape:")
# Displays the final dimensions heading.


# print(df.shape)
# Displays the final number of rows and columns.


# print("\nMissing Values After Cleaning:")
# Displays the missing-value heading.


# print(df.isnull().sum())
# Counts missing values in each column
# after the cleaning process.
#
# Some missing values may still remain,
# especially in newly created columns or
# additional merged columns.


# print("\nData Wrangling Completed Successfully!")
# Displays the final completion message.


# ============================================================
# 33. VISUALIZING AVERAGE PRICE BY NEIGHBORHOOD
# ============================================================

# if "neighborhood" in df.columns:
# Checks whether the neighborhood column exists.


#     neighborhood_prices = df.groupby(
#         "neighborhood"
#     )[price_column].mean()
#
# Groups the property records by neighborhood.
#
# Calculates the average property price
# for each neighborhood.
#
# The result is stored in neighborhood_prices.


#     plt.figure(figsize=(9, 5))
# Creates a new graph.
#
# figsize=(9, 5):
# Sets graph width to 9 inches
# and height to 5 inches.


#     neighborhood_prices.plot(kind="bar")
# Creates a bar chart showing average property prices.
#
# Each bar represents one neighborhood.
#
# The height of each bar represents
# the average sale price.


#     plt.title("Average House Price by Neighborhood")
# Sets the graph title.


#     plt.xlabel("Neighborhood")
# Sets the X-axis label.


#     plt.ylabel("Average Sale Price")
# Sets the Y-axis label.


#     plt.xticks(rotation=45)
# Rotates neighborhood labels by 45 degrees.
#
# This improves readability.


#     plt.tight_layout()
# Automatically adjusts graph spacing.


#     plt.show()
# Displays the neighborhood price bar chart.


# ============================================================
# 34. VISUALIZING AVERAGE PRICE BY PROPERTY TYPE
# ============================================================

# if "property_type" in df.columns:
# Checks whether the property_type column exists.


#     property_prices = df.groupby(
#         "property_type"
#     )[price_column].mean()
#
# Groups property records by property type.
#
# Calculates the average price for each property type.


#     plt.figure(figsize=(8, 5))
# Creates a new graph with width 8 inches
# and height 5 inches.


#     property_prices.plot(kind="bar")
# Creates a bar chart showing average property prices.
#
# Each bar represents a property type.


#     plt.title("Average House Price by Property Type")
# Sets the graph title.


#     plt.xlabel("Property Type")
# Sets the X-axis label.


#     plt.ylabel("Average Sale Price")
# Sets the Y-axis label.


#     plt.tight_layout()
# Adjusts the graph layout.


#     plt.show()
# Displays the property type bar chart.


# ============================================================
# 35. VISUALIZING PRICE DISTRIBUTION
# ============================================================

# plt.figure(figsize=(8, 5))
# Creates a new graph with width 8 inches
# and height 5 inches.


# plt.hist(
#     df[price_column],
#     bins=10,
#     edgecolor="black"
# )
#
# Creates a histogram of property prices.
#
# plt.hist():
# Displays the frequency distribution
# of numerical data.
#
# df[price_column]:
# Supplies the property price values.
#
# bins=10:
# Divides the price range into 10 intervals.
#
# edgecolor="black":
# Adds black borders around the histogram bars.
#
# The histogram helps understand
# how property prices are distributed.


# plt.title("Real Estate Price Distribution")
# Sets the graph title.


# plt.xlabel("Sale Price")
# Sets the X-axis label.


# plt.ylabel("Frequency")
# Sets the Y-axis label.
#
# Frequency means the number of property records
# falling within each price interval.


# plt.tight_layout()
# Adjusts graph spacing.


# plt.show()
# Displays the real estate price histogram.


# ============================================================
# 36. CONCLUSION
# ============================================================

# In this practical, the real estate dataset
# is loaded and explored using Pandas.

# Column names are cleaned using regular expressions.

# Duplicate records are removed.

# Text columns are standardized.

# Important price and date columns are identified.

# Missing numerical values are filled using median.
# Missing categorical values are filled using mode.
# Records with missing prices or invalid dates are removed.

# An additional neighborhood dataset is merged
# when the file is available.

# Data is filtered according to location,
# property type and year when required.

# New features such as price per square foot
# are created.

# Outliers are identified and handled using IQR.

# Data aggregation is performed by neighborhood
# and property type.

# Categorical variables are encoded using get_dummies().

# Cleaned and encoded datasets are exported as CSV files.

# Finally, bar charts and a histogram
# are created to visualize real estate prices.

# Therefore, this practical demonstrates
# data wrangling, transformation, aggregation
# and visualization using Python.
