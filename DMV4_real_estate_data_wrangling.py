
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
# EXPLANATION LINE BY LINE
# ============================================================

# import pandas as pd
# Imports Pandas for data loading and manipulation.

# import numpy as np
# Imports NumPy for numerical operations.

# import matplotlib.pyplot as plt
# Imports Matplotlib for data visualization.

# import re
# Imports regular expressions for cleaning column names.

# import os
# Checks whether an additional dataset file exists.

# pd.read_csv()
# Loads the real estate CSV dataset.

# df.head()
# Displays the first five records.

# df.shape
# Displays the number of rows and columns.

# df.info()
# Displays column names, data types and non-null values.

# describe(include="all")
# Displays statistical information about all columns.

# isnull().sum()
# Counts missing values in each column.

# re.sub()
# Replaces unwanted characters in column names.

# str.strip()
# Removes unnecessary spaces.

# str.lower()
# Converts column names into lowercase.

# drop_duplicates()
# Removes duplicate records.

# select_dtypes()
# Selects columns based on their data types.

# pd.api.types.is_string_dtype()
# Checks whether a column contains string data.

# pd.to_numeric()
# Converts the price column into numeric format.

# errors="coerce"
# Converts invalid values into missing values.

# pd.to_datetime()
# Converts the date column into date format.

# dropna()
# Removes records containing missing values in selected columns.

# median()
# Calculates the middle numerical value.

# mode()
# Finds the most frequently occurring value.

# fillna()
# Replaces missing values.

# os.path.exists()
# Checks whether the additional dataset is available.

# pd.merge()
# Combines two datasets using a common column.

# how="left"
# Keeps all records from the main dataset.

# selected_location
# Stores the location used for filtering.

# selected_property_type
# Stores the property type used for filtering.

# dt.year
# Extracts the year from the listing date.

# price_per_sqft
# Calculates the price per square foot.

# quantile()
# Calculates quartiles for outlier detection.

# IQR
# Calculates the interquartile range.

# clip()
# Caps extreme values within the specified limits.

# groupby()
# Groups records based on a selected column.

# agg()
# Calculates mean, minimum, maximum and count.

# to_csv()
# Saves the processed data into CSV files.

# pd.get_dummies()
# Converts categorical values into numerical columns.

# drop_first=True
# Removes one category to avoid redundant dummy columns.

# plt.figure()
# Creates a new graph.

# plot(kind="bar")
# Creates a bar chart.

# plt.hist()
# Creates a histogram showing price distribution.

# plt.title()
# Sets the graph title.

# plt.xlabel()
# Sets the horizontal axis label.

# plt.ylabel()
# Sets the vertical axis label.

# plt.tight_layout()
# Adjusts the graph layout.

# plt.show()
# Displays the graph.
