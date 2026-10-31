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
# EXPLANATION LINE BY LINE
# ============================================================

# import pandas as pd
# Imports Pandas for data loading and aggregation.

# import numpy as np
# Imports NumPy for numerical operations.

# import matplotlib.pyplot as plt
# Imports Matplotlib for creating graphs.

# pd.read_csv()
# Loads the retail sales CSV dataset.

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
# Removes unnecessary spaces.

# str.lower()
# Converts column names into lowercase.

# str.replace()
# Replaces spaces and hyphens in column names.

# next()
# Finds the first matching column name.

# drop_duplicates()
# Removes duplicate records.

# astype("string")
# Converts a column into string data type.

# str.title()
# Converts the first letter of each word into uppercase.

# pd.to_numeric()
# Converts sales values into numeric format.

# errors="coerce"
# Converts invalid numeric values into missing values.

# dropna()
# Removes records with missing required values.

# groupby()
# Groups records based on a selected column.

# sum()
# Calculates the total sales amount.

# sort_values()
# Sorts sales values in descending order.

# plot(kind="bar")
# Creates a bar chart.

# plot(kind="pie")
# Creates a pie chart.

# autopct
# Displays percentage values on the pie chart.

# idxmax()
# Finds the region or category with the highest sales.

# max()
# Finds the highest sales amount.

# head(3)
# Displays the top three regions.

# pivot_table()
# Creates a summary table using region and product category.

# aggfunc="sum"
# Calculates the total sales for each combination.

# fill_value=0
# Replaces missing combinations with zero.

# stacked=True
# Creates a stacked bar chart.

# stacked=False
# Creates a grouped bar chart.

# legend()
# Displays the product category labels.

# to_csv()
# Exports aggregated results into CSV files.

# plt.title()
# Sets the graph title.

# plt.xlabel()
# Sets the horizontal axis label.

# plt.ylabel()
# Sets the vertical axis label.

# plt.xticks()
# Rotates the region names.

# plt.tight_layout()
# Adjusts the graph layout.

# plt.show()
# Displays the graph.