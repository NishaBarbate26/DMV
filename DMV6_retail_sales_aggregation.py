# AIM: Perform Data Aggregation to Analyze Sales
# Performance by Region in a Retail Company.

# Installation commands:
# Windows:
# python -m pip install pandas numpy matplotlib
# python practical_sales_aggregation.py
#
# Ubuntu/Linux:
# pip install pandas numpy matplotlib
# python3 -m pip install pandas numpy matplotlib
# python3 practical_sales_aggregation.py


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Import dataset
df = pd.read_csv("Retail_Sales_Data.csv", encoding="latin1")

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
    .str.replace("-", "_")
)

print("\nColumn Names:")

print(df.columns.tolist())

# 4. Identify required columns

region_candidates = [
    "region",
    "sales_region"
]

sales_candidates = [
    "sales_amount",
    "sales",
    "total_sales",
    "amount"
]

category_candidates = [
    "product_category",
    "category",
    "product"
]

region_column = next(
    (col for col in region_candidates if col in df.columns),
    None
)

sales_column = next(
    (col for col in sales_candidates if col in df.columns),
    None
)

category_column = next(
    (col for col in category_candidates if col in df.columns),
    None
)

if region_column is None or sales_column is None or category_column is None:

    raise ValueError(
        "Dataset must contain Region, Sales Amount and Product Category columns."
    )

# 5. Handle missing values

df = df.drop_duplicates()

df[region_column] = df[region_column].astype("string").str.strip().str.title()

df[category_column] = df[category_column].astype("string").str.strip().str.title()

df[sales_column] = pd.to_numeric(
    df[sales_column],
    errors="coerce"
)

df = df.dropna(
    subset=[region_column, sales_column, category_column]
)

# Remove invalid negative sales values

df = df[df[sales_column] >= 0]

# 6. Group sales data by region

region_sales = df.groupby(
    region_column
)[sales_column].sum().sort_values(ascending=False)

print("\nTotal Sales by Region:")

print(region_sales)

# 7. Visualize sales distribution using bar chart

plt.figure(figsize=(9, 5))

region_sales.plot(
    kind="bar",
    color="skyblue"
)

plt.title("Total Sales by Region")

plt.xlabel("Region")

plt.ylabel("Total Sales Amount")

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()

# 8. Visualize sales distribution using pie chart

plt.figure(figsize=(7, 7))

region_sales.plot(
    kind="pie",
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Sales Distribution by Region")

plt.ylabel("")

plt.tight_layout()

plt.show()

# 9. Identify top-performing regions

print("\nTop Performing Regions:")

print(region_sales.head(3))

top_region = region_sales.idxmax()

highest_sales = region_sales.max()

print("\nBest Performing Region:", top_region)

print("Highest Sales Amount:", highest_sales)

# 10. Group sales by region and product category

region_category_sales = df.pivot_table(
    index=region_column,
    columns=category_column,
    values=sales_column,
    aggfunc="sum",
    fill_value=0
)

print("\nSales by Region and Product Category:")

print(region_category_sales)

# 11. Create stacked bar chart

region_category_sales.plot(
    kind="bar",
    stacked=True,
    figsize=(10, 6)
)

plt.title("Stacked Sales by Region and Product Category")

plt.xlabel("Region")

plt.ylabel("Total Sales Amount")

plt.legend(
    title="Product Category"
)

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()

# 12. Create grouped bar chart

region_category_sales.plot(
    kind="bar",
    stacked=False,
    figsize=(10, 6)
)

plt.title("Grouped Sales by Region and Product Category")

plt.xlabel("Region")

plt.ylabel("Total Sales Amount")

plt.legend(
    title="Product Category"
)

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()

# 13. Find top-selling product category

category_sales = df.groupby(
    category_column
)[sales_column].sum().sort_values(ascending=False)

print("\nTotal Sales by Product Category:")

print(category_sales)

print("\nTop Selling Category:")

print(category_sales.idxmax())

# 14. Export aggregated data

region_sales.to_csv(
    "region_sales_summary.csv"
)

region_category_sales.to_csv(
    "region_category_sales.csv"
)

category_sales.to_csv(
    "category_sales_summary.csv"
)

print("\nSales Aggregation Completed Successfully!")

# ============================================================
# AIM:
# To analyze and visualize retail sales data.
#
# 1. Import the retail sales dataset.
# 2. Explore the dataset.
# 3. Clean column names.
# 4. Identify region, sales and category columns.
# 5. Handle missing values and duplicates.
# 6. Remove invalid negative sales values.
# 7. Calculate total sales by region.
# 8. Visualize regional sales using bar and pie charts.
# 9. Identify top-performing regions.
# 10. Analyze sales by region and product category.
# 11. Create stacked and grouped bar charts.
# 12. Identify the top-selling product category.
# 13. Export aggregated sales data.
# ============================================================


# ============================================================
# INSTALLATION AND EXECUTION COMMANDS
# ============================================================

# Windows:
# python -m pip install pandas numpy matplotlib
# python practical_retail_sales.py

# Ubuntu/Linux:
# python3 -m pip install pandas numpy matplotlib
# python3 practical_retail_sales.py

# Keep Retail_Sales_Data.csv in the same folder
# as the Python program.


# ============================================================
# 1. IMPORTING LIBRARIES
# ============================================================

# import pandas as pd
# Imports the Pandas library.
# It is used to load, clean, manipulate and analyze data.
# pd is the short name used for Pandas.


# import numpy as np
# Imports the NumPy library.
# It is used for numerical calculations.
# np is the short name used for NumPy.
# Note: NumPy is imported but not directly used
# in this particular program.


# import matplotlib.pyplot as plt
# Imports the Pyplot module from Matplotlib.
# It is used to create graphs and charts.
# plt is the short name used for Pyplot.


# ============================================================
# 2. IMPORTING THE DATASET
# ============================================================

# df = pd.read_csv("Retail_Sales_Data.csv", encoding="latin1")
#
# Reads the retail sales dataset from a CSV file.
#
# pd.read_csv():
# Loads CSV data into a Pandas DataFrame.
#
# df:
# Stores the complete dataset.
#
# encoding="latin1":
# Specifies the character encoding used to read the file.
#
# It helps read special characters that may not be
# supported by the default UTF-8 encoding.
#
# The dataset may contain:
# Region.
# Sales Amount.
# Product Category.
# Product Name.
# Quantity.
# Date.
# Customer Information.


# print("First Five Records:")
# Displays a heading for the first five records.


# print(df.head())
# Displays the first five rows of the dataset.
#
# head():
# Returns the first five rows by default.
#
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
# (1000, 6)
#
# This means:
# 1000 rows = 1000 sales records.
# 6 columns = 6 attributes.
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
# It helps us understand the dataset structure
# and identify missing values.


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
# unique = Number of distinct values.
# top = Most frequent value.
# freq = Frequency of the most frequent value.
#
# include="all":
# Includes both numerical and categorical columns.


# print("\nMissing Values:")
# Displays the missing-value heading.


# print(df.isnull().sum())
# isnull():
# Identifies missing values.
#
# sum():
# Counts missing values in each column.
#
# It helps us identify incomplete sales records.


# ============================================================
# 4. CLEANING COLUMN NAMES
# ============================================================

# df.columns = (
#     df.columns
#     .str.strip()
#     .str.lower()
#     .str.replace(" ", "_")
#     .str.replace("-", "_")
# )
#
# Standardizes all column names.
#
# df.columns:
# Accesses all column names.
#
# .str.strip():
# Removes unwanted spaces from the beginning
# and end of column names.
#
# .str.lower():
# Converts all column names into lowercase.
#
# Example:
# "Region" becomes "region".
#
# .str.replace(" ", "_"):
# Replaces spaces with underscores.
#
# Example:
# "Sales Amount" becomes "sales_amount".
#
# .str.replace("-", "_"):
# Replaces hyphens with underscores.
#
# Example:
# "Product-Category" becomes "product_category".
#
# This makes column names consistent
# and easier to access in Python.


# print("\nColumn Names:")
# Displays the column names heading.


# print(df.columns.tolist())
# Converts column names into a Python list
# and displays them.


# ============================================================
# 5. IDENTIFYING REQUIRED COLUMNS
# ============================================================

# region_candidates = [
#     "region",
#     "sales_region"
# ]
#
# Creates a list of possible region column names.
#
# Different datasets may use different names
# for the region column.


# sales_candidates = [
#     "sales_amount",
#     "sales",
#     "total_sales",
#     "amount"
# ]
#
# Creates a list of possible sales column names.
#
# It allows the program to work with
# different sales column names.


# category_candidates = [
#     "product_category",
#     "category",
#     "product"
# ]
#
# Creates a list of possible product category names.
#
# The category may be stored under different
# column names in different datasets.


# region_column = next(
#     (col for col in region_candidates if col in df.columns),
#     None
# )
#
# Searches for the first matching region column.
#
# next():
# Returns the first available matching value.
#
# if col in df.columns:
# Checks whether the candidate column exists.
#
# If a matching column is found:
# Its name is stored in region_column.
#
# Otherwise:
# None is returned.


# sales_column = next(
#     (col for col in sales_candidates if col in df.columns),
#     None
# )
#
# Searches for the first matching sales column.
#
# The selected column name is stored in sales_column.
#
# If no matching column exists:
# None is returned.


# category_column = next(
#     (col for col in category_candidates if col in df.columns),
#     None
# )
#
# Searches for the first matching product category column.
#
# The selected column name is stored in category_column.
#
# If no matching column exists:
# None is returned.


# if region_column is None or sales_column is None or category_column is None:
#
# Checks whether any required column is missing.
#
# The program requires:
# Region.
# Sales Amount.
# Product Category.
#
# or:
# Means that even one missing required column
# makes the condition True.


#     raise ValueError(
#         "Dataset must contain Region, Sales Amount and Product Category columns."
#     )
#
# Raises an error and stops the program
# if any required column is not found.
#
# This prevents the program from continuing
# with an incomplete dataset.


# ============================================================
# 6. HANDLING MISSING VALUES AND DUPLICATES
# ============================================================

# df = df.drop_duplicates()
#
# Removes duplicate rows from the dataset.
#
# drop_duplicates():
# Keeps the first occurrence and removes repeated rows.
#
# This prevents the same sales record
# from being counted multiple times.


# df[region_column] = df[region_column].astype("string").str.strip().str.title()
#
# Cleans and standardizes the region column.
#
# astype("string"):
# Converts values into Pandas string data type.
#
# str.strip():
# Removes unwanted spaces from the beginning
# and end of text.
#
# str.title():
# Converts the first letter of each word
# into uppercase.
#
# Example:
# " pune " becomes "Pune".
# "mumbai" becomes "Mumbai".
#
# This prevents the same region from being
# treated as different values because of text formatting.


# df[category_column] = df[category_column].astype("string").str.strip().str.title()
#
# Cleans and standardizes the product category column.
#
# astype("string"):
# Converts values into string format.
#
# str.strip():
# Removes unwanted spaces.
#
# str.title():
# Converts category names into title case.
#
# Example:
# " electronics " becomes "Electronics".
# "furniture" becomes "Furniture".


# ============================================================
# 7. CONVERTING SALES INTO NUMERICAL FORMAT
# ============================================================

# df[sales_column] = pd.to_numeric(
#     df[sales_column],
#     errors="coerce"
# )
#
# Converts the sales column into numerical format.
#
# pd.to_numeric():
# Converts values into numbers.
#
# errors="coerce":
# Converts invalid values into NaN.
#
# Example:
# "5000" becomes 5000.
#
# An invalid value such as "unknown"
# becomes NaN.
#
# Numerical values are required for
# addition, aggregation and statistical analysis.


# ============================================================
# 8. REMOVING MISSING VALUES
# ============================================================

# df = df.dropna(
#     subset=[region_column, sales_column, category_column]
# )
#
# Removes rows where any required column is missing.
#
# subset:
# Specifies the columns to check.
#
# The required columns are:
# Region.
# Sales Amount.
# Product Category.
#
# A record without these values cannot be used
# for the required regional and category analysis.


# ============================================================
# 9. REMOVING INVALID NEGATIVE SALES VALUES
# ============================================================

# df = df[df[sales_column] >= 0]
#
# Filters the dataset to retain records
# where sales amount is greater than or equal to zero.
#
# df[sales_column] >= 0:
# Checks whether each sales amount is non-negative.
#
# Negative sales records are removed.
#
# This program assumes negative values
# are invalid sales records.
#
# Note:
# In real retail datasets, negative values may represent
# refunds or returned products.
# They should only be removed if the analysis
# is intended to exclude returns.


# ============================================================
# 10. GROUPING SALES DATA BY REGION
# ============================================================

# region_sales = df.groupby(
#     region_column
# )[sales_column].sum().sort_values(ascending=False)
#
# Calculates total sales for each region.
#
# groupby(region_column):
# Groups all records according to region.
#
# Example:
# Pune.
# Mumbai.
# Nashik.
#
# [sales_column]:
# Selects the sales amount column.
#
# sum():
# Adds all sales values within each region.
#
# sort_values(ascending=False):
# Sorts the total sales in descending order.
#
# ascending=False:
# Places the highest sales value first.
#
# The result is stored in region_sales.


# print("\nTotal Sales by Region:")
# Displays the regional sales heading.


# print(region_sales)
# Displays total sales for each region.
#
# It helps identify which region
# contributes the highest sales.


# ============================================================
# 11. VISUALIZING REGIONAL SALES USING BAR CHART
# ============================================================

# plt.figure(figsize=(9, 5))
# Creates a new graph.
#
# figsize=(9, 5):
# Sets graph width to 9 inches
# and height to 5 inches.


# region_sales.plot(
#     kind="bar",
#     color="skyblue"
# )
#
# Creates a bar chart using the regional sales data.
#
# kind="bar":
# Specifies a vertical bar chart.
#
# color="skyblue":
# Sets the bar color to sky blue.
#
# X-axis:
# Region names.
#
# Y-axis:
# Total sales amount.
#
# The chart makes it easy to compare
# sales across different regions.


# plt.title("Total Sales by Region")
# Sets the graph title.


# plt.xlabel("Region")
# Labels the X-axis as Region.


# plt.ylabel("Total Sales Amount")
# Labels the Y-axis as Total Sales Amount.


# plt.xticks(rotation=45)
# Rotates region names by 45 degrees.
#
# This improves label readability.


# plt.tight_layout()
# Adjusts the graph layout
# to prevent labels from being cut off.


# plt.show()
# Displays the regional sales bar chart.


# ============================================================
# 12. VISUALIZING REGIONAL SALES USING PIE CHART
# ============================================================

# plt.figure(figsize=(7, 7))
# Creates a square graph for the pie chart.


# region_sales.plot(
#     kind="pie",
#     autopct="%1.1f%%",
#     startangle=90
# )
#
# Creates a pie chart showing the sales contribution
# of each region.
#
# kind="pie":
# Specifies a pie chart.
#
# autopct="%1.1f%%":
# Displays percentage values on each pie slice.
#
# %1.1f:
# Displays percentages with one decimal place.
#
# startangle=90:
# Starts the first slice at 90 degrees.
#
# The pie chart shows how much each region
# contributes to total sales.


# plt.title("Sales Distribution by Region")
# Sets the graph title.


# plt.ylabel("")
# Removes the default Y-axis label.
#
# A pie chart does not require a Y-axis label.


# plt.tight_layout()
# Adjusts the graph layout.


# plt.show()
# Displays the regional sales pie chart.


# ============================================================
# 13. IDENTIFYING TOP-PERFORMING REGIONS
# ============================================================

# print("\nTop Performing Regions:")
# Displays the top-performing regions heading.


# print(region_sales.head(3))
# Displays the first three entries
# from the sorted regional sales result.
#
# Since the data is sorted in descending order,
# these are the three regions with the highest sales.
#
# head(3):
# Returns the first three records.


# top_region = region_sales.idxmax()
#
# Finds the region with the highest total sales.
#
# idxmax():
# Returns the index or label associated
# with the maximum value.
#
# In this case, it returns the region name.


# highest_sales = region_sales.max()
#
# Finds the maximum total sales value.
#
# max():
# Returns the highest numerical value.


# print("\nBest Performing Region:", top_region)
# Displays the name of the best-performing region.


# print("Highest Sales Amount:", highest_sales)
# Displays the highest sales amount.


# ============================================================
# 14. GROUPING SALES BY REGION AND PRODUCT CATEGORY
# ============================================================

# region_category_sales = df.pivot_table(
#     index=region_column,
#     columns=category_column,
#     values=sales_column,
#     aggfunc="sum",
#     fill_value=0
# )
#
# Creates a pivot table to summarize sales
# by region and product category.
#
# pivot_table():
# Creates a summarized table from the dataset.
#
# index=region_column:
# Places regions in rows.
#
# Example:
# Pune, Mumbai, Nashik.
#
# columns=category_column:
# Places product categories in columns.
#
# Example:
# Electronics, Furniture, Clothing.
#
# values=sales_column:
# Uses the sales amount for calculations.
#
# aggfunc="sum":
# Adds the sales amounts for each region
# and product category combination.
#
# fill_value=0:
# Replaces missing combinations with zero.
#
# Example:
# If Nashik has no Electronics sales,
# its Electronics value will be 0.


# print("\nSales by Region and Product Category:")
# Displays the pivot table heading.


# print(region_category_sales)
# Displays total sales for every region
# and product category combination.


# ============================================================
# 15. CREATING STACKED BAR CHART
# ============================================================

# region_category_sales.plot(
#     kind="bar",
#     stacked=True,
#     figsize=(10, 6)
# )
#
# Creates a stacked bar chart.
#
# kind="bar":
# Creates a bar chart.
#
# stacked=True:
# Stacks different product category sales
# on top of one another.
#
# figsize=(10, 6):
# Sets graph width to 10 inches
# and height to 6 inches.
#
# Each bar represents one region.
#
# Each colored section represents a product category.
#
# It helps compare total regional sales
# and the contribution of each category.


# plt.title("Stacked Sales by Region and Product Category")
# Sets the graph title.


# plt.xlabel("Region")
# Labels the X-axis as Region.


# plt.ylabel("Total Sales Amount")
# Labels the Y-axis as Total Sales Amount.


# plt.legend(
#     title="Product Category"
# )
#
# Displays the legend.
#
# title:
# Sets the legend heading to Product Category.
#
# The legend identifies each category's color.


# plt.xticks(rotation=45)
# Rotates region names by 45 degrees.


# plt.tight_layout()
# Adjusts the graph layout.


# plt.show()
# Displays the stacked bar chart.


# ============================================================
# 16. CREATING GROUPED BAR CHART
# ============================================================

# region_category_sales.plot(
#     kind="bar",
#     stacked=False,
#     figsize=(10, 6)
# )
#
# Creates a grouped bar chart.
#
# kind="bar":
# Creates a bar chart.
#
# stacked=False:
# Displays category bars side by side
# instead of stacking them.
#
# figsize=(10, 6):
# Sets graph width to 10 inches
# and height to 6 inches.
#
# Each group represents one region.
#
# Each bar within a group represents
# one product category.
#
# It helps compare different categories
# within each region.


# plt.title("Grouped Sales by Region and Product Category")
# Sets the graph title.


# plt.xlabel("Region")
# Labels the X-axis as Region.


# plt.ylabel("Total Sales Amount")
# Labels the Y-axis as Total Sales Amount.


# plt.legend(
#     title="Product Category"
# )
#
# Displays the legend with its heading.


# plt.xticks(rotation=45)
# Rotates region names by 45 degrees.


# plt.tight_layout()
# Adjusts the graph layout.


# plt.show()
# Displays the grouped bar chart.


# ============================================================
# 17. FINDING THE TOP-SELLING PRODUCT CATEGORY
# ============================================================

# category_sales = df.groupby(
#     category_column
# )[sales_column].sum().sort_values(ascending=False)
#
# Calculates the total sales for each product category.
#
# groupby(category_column):
# Groups records according to product category.
#
# [sales_column]:
# Selects the sales amount column.
#
# sum():
# Adds the sales values within each category.
#
# sort_values(ascending=False):
# Sorts category sales from highest to lowest.
#
# The result is stored in category_sales.


# print("\nTotal Sales by Product Category:")
# Displays the category sales heading.


# print(category_sales)
# Displays total sales for each product category.


# print("\nTop Selling Category:")
# Displays the top-selling category heading.


# print(category_sales.idxmax())
# Finds and displays the category
# with the highest total sales.
#
# idxmax():
# Returns the category name associated
# with the maximum sales value.


# ============================================================
# 18. EXPORTING AGGREGATED DATA
# ============================================================

# region_sales.to_csv(
#     "region_sales_summary.csv"
# )
#
# Saves the total sales of each region
# into a CSV file.
#
# Output file:
# region_sales_summary.csv
#
# It contains region names and their total sales.


# region_category_sales.to_csv(
#     "region_category_sales.csv"
# )
#
# Saves the region and product category pivot table.
#
# Output file:
# region_category_sales.csv
#
# It contains sales values for each
# region and product category combination.


# category_sales.to_csv(
#     "category_sales_summary.csv"
# )
#
# Saves the total sales of each product category.
#
# Output file:
# category_sales_summary.csv
#
# It contains category names and their total sales.


# print("\nSales Aggregation Completed Successfully!")
#
# Displays the final completion message.
#
# It indicates that the sales analysis,
# visualization and export operations are completed.
