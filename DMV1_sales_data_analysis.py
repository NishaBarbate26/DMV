
# AIM: Analyzing Sales Data from Multiple File Formats
# Dataset: CSV, Excel and JSON Sales Data

# Installation Commands:
#
# Windows:
# python -m pip install pandas numpy matplotlib openpyxl
# python DMV1_sales_data_analysis.py
#
# Ubuntu / Linux:
# sudo apt update
# sudo apt install python3 python3-pip
# pip3 install pandas numpy matplotlib openpyxl
# python3 DMV1_sales_data_analysis.py
#
# For Ubuntu using a normal text editor:
# Save this file as DMV1_sales_data_analysis.py
# Open Terminal in the file's folder and execute the commands above.


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

data = {
    "Order ID": [101, 102, 103, 104, 105, 106],
    "Order Date": [
        "2026-01-05", "2026-01-10",
        "2026-02-12", "2026-02-18",
        "2026-03-15", "2026-03-20"
    ],
    "Customer Name": [
        "Amit", "Riya", "Rahul",
        "Neha", "Sahil", "Priya"
    ],
    "Product": [
        "Laptop", "Mobile", "Chair",
        "Table", "Headphones", "Sofa"
    ],
    "Category": [
        "Electronics", "Electronics", "Furniture",
        "Furniture", "Electronics", "Furniture"
    ],
    "Quantity": [2, 3, 5, 2, 4, 1],
    "Unit Price": [50000, 20000, 3000, 7000, 1500, 25000],
    "Region": ["Pune", "Mumbai", "Pune", "Nashik", "Mumbai", "Pune"]
}

sample_df = pd.DataFrame(data)

if not os.path.exists("sales.csv"):
    sample_df.iloc[:2].to_csv("sales.csv", index=False)

if not os.path.exists("sales.xlsx"):
    sample_df.iloc[2:4].to_excel("sales.xlsx", index=False)

if not os.path.exists("sales.json"):
    sample_df.iloc[4:].to_json("sales.json", orient="records", indent=4)

csv_df = pd.read_csv("sales.csv")
excel_df = pd.read_excel("sales.xlsx")
json_df = pd.read_json("sales.json")

print("\nCSV Data:")
print(csv_df)

print("\nExcel Data:")
print(excel_df)

print("\nJSON Data:")
print(json_df)

print("\nCSV Dataset Information:")
csv_df.info()

print("\nExcel Dataset Information:")
excel_df.info()

print("\nJSON Dataset Information:")
json_df.info()

print("\nMissing Values in CSV:")
print(csv_df.isnull().sum())

print("\nMissing Values in Excel:")
print(excel_df.isnull().sum())

print("\nMissing Values in JSON:")
print(json_df.isnull().sum())

df = pd.concat(
    [csv_df, excel_df, json_df],
    ignore_index=True
)

print("\nCombined Dataset:")
print(df)

df.columns = (
    df.columns.str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

print("\nDuplicate Records:", df.duplicated().sum())
df = df.drop_duplicates()

df["order_date"] = pd.to_datetime(
    df["order_date"], errors="coerce"
)

df["order_id"] = pd.to_numeric(
    df["order_id"], errors="coerce"
)

df["quantity"] = pd.to_numeric(
    df["quantity"], errors="coerce"
)

df["unit_price"] = pd.to_numeric(
    df["unit_price"], errors="coerce"
)

df["customer_name"] = df["customer_name"].fillna("Unknown")
df["product"] = df["product"].fillna("Unknown")
df["category"] = df["category"].fillna("Unknown")
df["region"] = df["region"].fillna("Unknown")

df["quantity"] = df["quantity"].fillna(0)
df["unit_price"] = df["unit_price"].fillna(0)

df = df.dropna(subset=["order_date", "order_id"])

df["total_sales"] = df["quantity"] * df["unit_price"]

df["month"] = df["order_date"].dt.month_name()
df["year"] = df["order_date"].dt.year

name_parts = (
    df["customer_name"]
    .fillna("")
    .astype(str)
    .str.strip()
    .str.partition(" ")
)

df["first_name"] = name_parts[0]
df["last_name"] = name_parts[2]

print("\nCleaned Dataset:")
print(df)

print("\nSales Statistics:")
print(df["total_sales"].describe())

print("\nTotal Sales:", df["total_sales"].sum())
print("Average Sales:", df["total_sales"].mean())
print("Maximum Sales:", df["total_sales"].max())
print("Minimum Sales:", df["total_sales"].min())

category_sales = df.groupby("category")["total_sales"].sum()

print("\nCategory-wise Sales:")
print(category_sales)

region_sales = df.groupby("region")["total_sales"].sum()

print("\nRegion-wise Sales:")
print(region_sales)

month_sales = df.groupby("month")["total_sales"].sum()

print("\nMonth-wise Sales:")
print(month_sales)

product_sales = df.groupby("product")["total_sales"].sum()

print("\nProduct-wise Sales:")
print(product_sales)

df.to_csv("cleaned_sales.csv", index=False)

print("\nCleaned dataset saved as cleaned_sales.csv")

category_sales.plot(kind="bar", title="Category-wise Sales")
plt.xlabel("Category")
plt.ylabel("Total Sales")
plt.tight_layout()
plt.show()

region_sales.plot(
    kind="pie",
    autopct="%1.1f%%",
    title="Region-wise Sales"
)
plt.ylabel("")
plt.tight_layout()
plt.show()

plt.boxplot(df["total_sales"])
plt.title("Sales Distribution")
plt.ylabel("Total Sales")
plt.tight_layout()
plt.show()

month_sales.plot(kind="bar", title="Month-wise Sales")
plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.tight_layout()
plt.show()

product_sales.plot(kind="bar", title="Product-wise Sales")
plt.xlabel("Product")
plt.ylabel("Total Sales")
plt.tight_layout()
plt.show()

print("\nSales Data Analysis Completed Successfully!")

# ============================================================
# AIM:
# Analyze Sales Data from Multiple File Formats
# (CSV, Excel and JSON).
#
# Perform data loading, data cleaning, data transformation,
# data aggregation, statistical analysis and visualization.
# ============================================================


# ============================================================
# 1. INSTALLATION AND EXECUTION COMMANDS
# ============================================================

# Windows:
# python -m pip install pandas numpy matplotlib openpyxl
# python practical_sales.py

# Ubuntu/Linux:
# python3 -m pip install pandas numpy matplotlib openpyxl
# python3 practical_sales.py

# openpyxl is required by Pandas to read and write Excel files.


# ============================================================
# 2. IMPORTING LIBRARIES
# ============================================================

# import pandas as pd
# Imports the Pandas library.
# Pandas is used for data manipulation and analysis.
# It provides DataFrame and Series data structures.
# pd is the short name used for Pandas.


# import numpy as np
# Imports the NumPy library.
# NumPy is used for numerical calculations and array operations.
# np is the short name used for NumPy.
# Note: NumPy is imported but not directly used in this program.


# import matplotlib.pyplot as plt
# Imports the Pyplot module from Matplotlib.
# It is used to create graphs and charts.
# plt is the short name used for Pyplot.


# import os
# Imports the OS module.
# It provides functions to interact with the operating system.
# Here, it is used to check whether files already exist.


# ============================================================
# 3. CREATING SAMPLE SALES DATA
# ============================================================

# data = {
# Creates a Python dictionary named data.
# It contains sample sales information.
# Each dictionary key represents a column name.
# Each dictionary value represents the records in that column.


# "Order ID": [101, 102, 103, 104, 105, 106],
# Stores the unique identification numbers of orders.


# "Order Date": [
#     "2026-01-05", "2026-01-10",
#     "2026-02-12", "2026-02-18",
#     "2026-03-15", "2026-03-20"
# ],
# Stores the dates on which the orders were placed.
# Dates are initially stored as strings.


# "Customer Name": [
#     "Amit", "Riya", "Rahul",
#     "Neha", "Sahil", "Priya"
# ],
# Stores the names of customers who placed the orders.


# "Product": [
#     "Laptop", "Mobile", "Chair",
#     "Table", "Headphones", "Sofa"
# ],
# Stores the names of the purchased products.


# "Category": [
#     "Electronics", "Electronics", "Furniture",
#     "Furniture", "Electronics", "Furniture"
# ],
# Stores the category of each product.
# The two categories are Electronics and Furniture.


# "Quantity": [2, 3, 5, 2, 4, 1],
# Stores the number of units purchased in each order.


# "Unit Price": [50000, 20000, 3000, 7000, 1500, 25000],
# Stores the price of one unit of each product.


# "Region": ["Pune", "Mumbai", "Pune", "Nashik", "Mumbai", "Pune"]
# Stores the location or region of each customer order.


# ============================================================
# 4. CONVERTING DATA INTO A DATAFRAME
# ============================================================

# sample_df = pd.DataFrame(data)
# Converts the Python dictionary into a Pandas DataFrame.
#
# A DataFrame is a two-dimensional table containing rows
# and columns.
#
# Here, the DataFrame contains six sales records
# and eight columns.
#
# It is used as the sample dataset for this practical.


# ============================================================
# 5. CREATING CSV FILE
# ============================================================

# if not os.path.exists("sales.csv"):
# Checks whether a file named sales.csv already exists.
#
# os.path.exists() returns:
# True if the file exists.
# False if the file does not exist.
#
# The file is created only when it does not already exist.


#     sample_df.iloc[:2].to_csv("sales.csv", index=False)
# Selects the first two rows of the DataFrame.
#
# iloc is used for selecting rows and columns
# using their integer positions.
#
# [:2] selects rows at positions 0 and 1.
#
# to_csv() saves the selected records in CSV format.
#
# "sales.csv" is the output filename.
#
# index=False prevents Pandas from saving
# the DataFrame row numbers as an additional column.


# ============================================================
# 6. CREATING EXCEL FILE
# ============================================================

# if not os.path.exists("sales.xlsx"):
# Checks whether the Excel file already exists.
# If it does not exist, the file is created.


#     sample_df.iloc[2:4].to_excel("sales.xlsx", index=False)
# Selects rows at positions 2 and 3.
#
# to_excel() saves the selected records
# in Microsoft Excel format.
#
# "sales.xlsx" is the output filename.
#
# index=False prevents the DataFrame index
# from being saved as an extra column.


# ============================================================
# 7. CREATING JSON FILE
# ============================================================

# if not os.path.exists("sales.json"):
# Checks whether the JSON file already exists.
# If it does not exist, the file is created.


#     sample_df.iloc[4:].to_json("sales.json", orient="records", indent=4)
# Selects the remaining rows starting from position 4.
#
# to_json() saves the selected records in JSON format.
#
# orient="records" stores each row as a separate JSON object.
#
# indent=4 adds indentation to make the JSON file readable.
#
# The output filename is sales.json.


# ============================================================
# 8. READING DATA FROM DIFFERENT FILE FORMATS
# ============================================================

# csv_df = pd.read_csv("sales.csv")
# Reads the CSV file using Pandas.
# Stores the data in a DataFrame named csv_df.
#
# CSV stands for Comma-Separated Values.
# It stores tabular data as text separated by commas.


# excel_df = pd.read_excel("sales.xlsx")
# Reads the Excel file using Pandas.
# Stores the data in excel_df.
#
# Excel files generally use the .xlsx extension.
# They can store data in worksheets.


# json_df = pd.read_json("sales.json")
# Reads the JSON file using Pandas.
# Stores the data in json_df.
#
# JSON stands for JavaScript Object Notation.
# It stores structured data using objects and key-value pairs.


# ============================================================
# 9. DISPLAYING INDIVIDUAL DATASETS
# ============================================================

# print("\nCSV Data:")
# Displays a heading for the CSV dataset.
# \n inserts a new line before the heading.


# print(csv_df)
# Displays the records loaded from the CSV file.


# print("\nExcel Data:")
# Displays a heading for the Excel dataset.


# print(excel_df)
# Displays the records loaded from the Excel file.


# print("\nJSON Data:")
# Displays a heading for the JSON dataset.


# print(json_df)
# Displays the records loaded from the JSON file.


# ============================================================
# 10. DISPLAYING DATASET INFORMATION
# ============================================================

# print("\nCSV Dataset Information:")
# Displays a heading for CSV dataset information.


# csv_df.info()
# Displays a summary of the CSV DataFrame.
#
# It shows:
# Number of rows.
# Column names.
# Number of non-null values.
# Data types of columns.
# Memory usage.


# print("\nExcel Dataset Information:")
# Displays a heading for Excel dataset information.


# excel_df.info()
# Displays the structure and information of the Excel dataset.


# print("\nJSON Dataset Information:")
# Displays a heading for JSON dataset information.


# json_df.info()
# Displays the structure and information of the JSON dataset.


# ============================================================
# 11. CHECKING MISSING VALUES
# ============================================================

# print("\nMissing Values in CSV:")
# Displays a heading for missing-value information.


# print(csv_df.isnull().sum())
# isnull() checks each DataFrame value.
#
# It returns True for missing values and False for other values.
#
# sum() counts the missing values in each column.
#
# This helps identify incomplete records.


# print("\nMissing Values in Excel:")
# Displays a heading for missing values in the Excel dataset.


# print(excel_df.isnull().sum())
# Counts missing values in each Excel column.


# print("\nMissing Values in JSON:")
# Displays a heading for missing values in the JSON dataset.


# print(json_df.isnull().sum())
# Counts missing values in each JSON column.


# ============================================================
# 12. COMBINING ALL DATASETS
# ============================================================

# df = pd.concat(
#     [csv_df, excel_df, json_df],
#     ignore_index=True
# )
#
# pd.concat() combines multiple DataFrames.
#
# csv_df:
# Contains records loaded from the CSV file.
#
# excel_df:
# Contains records loaded from the Excel file.
#
# json_df:
# Contains records loaded from the JSON file.
#
# ignore_index=True:
# Creates a new continuous index for the combined DataFrame.
#
# The three datasets are combined into one DataFrame named df.
#
# Since each file contains two records,
# the combined dataset contains six records.


# print("\nCombined Dataset:")
# Displays the heading for the combined dataset.


# print(df)
# Displays all records combined from CSV, Excel and JSON.


# ============================================================
# 13. CLEANING COLUMN NAMES
# ============================================================

# df.columns = (
#     df.columns.str.strip()
#     .str.lower()
#     .str.replace(" ", "_")
# )
#
# Cleans and standardizes all column names.
#
# df.columns:
# Accesses all column names in the DataFrame.
#
# str.strip():
# Removes unnecessary spaces from the beginning and end
# of each column name.
#
# str.lower():
# Converts all column names into lowercase letters.
#
# str.replace(" ", "_"):
# Replaces spaces with underscores.
#
# Example:
# "Order ID" becomes "order_id".
# "Customer Name" becomes "customer_name".
# "Unit Price" becomes "unit_price".
#
# This makes column names easier to use in Python.


# ============================================================
# 14. CHECKING AND REMOVING DUPLICATE RECORDS
# ============================================================

# print("\nDuplicate Records:", df.duplicated().sum())
# duplicated() checks whether rows are repeated.
#
# sum() counts the total number of duplicate rows.
#
# This displays the number of duplicate records.


# df = df.drop_duplicates()
# Removes duplicate rows from the DataFrame.
#
# Only one copy of each identical row is retained.
#
# This avoids counting the same sales record multiple times.


# ============================================================
# 15. CONVERTING ORDER DATE INTO DATETIME FORMAT
# ============================================================

# df["order_date"] = pd.to_datetime(
#     df["order_date"], errors="coerce"
# )
#
# Converts the order_date column into Pandas datetime format.
#
# pd.to_datetime():
# Converts date values into a date and time data type.
#
# errors="coerce":
# Converts invalid date values into NaT.
#
# NaT means Not a Time.
#
# Datetime format allows date-based operations
# such as extracting month and year.


# ============================================================
# 16. CONVERTING NUMERIC COLUMNS
# ============================================================

# df["order_id"] = pd.to_numeric(
#     df["order_id"], errors="coerce"
# )
#
# Converts order_id values into numeric format.
#
# Invalid values are converted into NaN.


# df["quantity"] = pd.to_numeric(
#     df["quantity"], errors="coerce"
# )
#
# Converts quantity values into numeric format.
# Invalid values become NaN.


# df["unit_price"] = pd.to_numeric(
#     df["unit_price"], errors="coerce"
# )
#
# Converts unit_price values into numeric format.
# Invalid values become NaN.


# ============================================================
# 17. HANDLING MISSING TEXT VALUES
# ============================================================

# df["customer_name"] = df["customer_name"].fillna("Unknown")
# Replaces missing customer names with "Unknown".
#
# fillna() is used to fill missing values.


# df["product"] = df["product"].fillna("Unknown")
# Replaces missing product names with "Unknown".


# df["category"] = df["category"].fillna("Unknown")
# Replaces missing category values with "Unknown".


# df["region"] = df["region"].fillna("Unknown")
# Replaces missing region values with "Unknown".


# ============================================================
# 18. HANDLING MISSING NUMERIC VALUES
# ============================================================

# df["quantity"] = df["quantity"].fillna(0)
# Replaces missing quantity values with zero.
#
# Zero is used because an unknown quantity should not
# be treated as a positive number in this sample calculation.


# df["unit_price"] = df["unit_price"].fillna(0)
# Replaces missing unit prices with zero.
#
# This prevents missing prices from causing
# missing total-sales calculations.


# ============================================================
# 19. REMOVING RECORDS WITH MISSING IMPORTANT VALUES
# ============================================================

# df = df.dropna(subset=["order_date", "order_id"])
# Removes rows where order_date or order_id is missing.
#
# subset specifies the columns that must not contain missing values.
#
# A record without a valid order date or order ID
# is removed because these fields are important
# for identifying and analyzing an order.


# ============================================================
# 20. CALCULATING TOTAL SALES
# ============================================================

# df["total_sales"] = df["quantity"] * df["unit_price"]
# Creates a new column called total_sales.
#
# Total Sales = Quantity x Unit Price.
#
# Example:
# Quantity = 2.
# Unit Price = 50000.
# Total Sales = 2 x 50000 = 100000.
#
# This calculates the sales amount for each order.


# ============================================================
# 21. EXTRACTING MONTH AND YEAR
# ============================================================

# df["month"] = df["order_date"].dt.month_name()
# Extracts the month name from the order date.
#
# dt provides datetime-related operations.
#
# month_name() returns the full month name.
#
# Example:
# 2026-01-05 becomes January.
# 2026-02-12 becomes February.
# 2026-03-15 becomes March.
#
# The result is stored in a new column named month.


# df["year"] = df["order_date"].dt.year
# Extracts the year from the order date.
#
# Example:
# 2026-01-05 becomes 2026.
#
# The result is stored in a new column named year.


# ============================================================
# 22. SPLITTING CUSTOMER NAMES
# ============================================================

# name_parts = (
#     df["customer_name"]
#     .fillna("")
#     .astype(str)
#     .str.strip()
#     .str.partition(" ")
# )
#
# Splits customer names into first name and remaining name.
#
# fillna(""):
# Replaces missing names with an empty string.
#
# astype(str):
# Converts all values into strings.
#
# str.strip():
# Removes unnecessary spaces from the beginning and end.
#
# str.partition(" "):
# Splits each name at the first space.
#
# It returns three parts:
# Part 1: Text before the first space.
# Part 2: The separator space.
# Part 3: Text after the first space.
#
# Example:
# "Amit Sharma" becomes:
# First part = "Amit"
# Separator = " "
# Remaining part = "Sharma"
#
# For a single-word name such as "Amit":
# First part = "Amit"
# Remaining part = "".


# df["first_name"] = name_parts[0]
# Stores the first part of each customer name
# in a new column called first_name.


# df["last_name"] = name_parts[2]
# Stores the remaining part after the first space
# in a new column called last_name.
#
# Note:
# This column contains everything after the first space,
# not necessarily only one surname.
# Single-word names have an empty last_name.


# ============================================================
# 23. DISPLAYING THE CLEANED DATASET
# ============================================================

# print("\nCleaned Dataset:")
# Displays the heading for the cleaned dataset.


# print(df)
# Displays the final cleaned and transformed DataFrame.
#
# It now contains original columns and newly created columns:
# total_sales, month, year, first_name and last_name.


# ============================================================
# 24. CALCULATING SALES STATISTICS
# ============================================================

# print("\nSales Statistics:")
# Displays the heading for statistical analysis.


# print(df["total_sales"].describe())
# describe() calculates descriptive statistics.
#
# It displays:
#
# count:
# Number of sales records.
#
# mean:
# Average sales value.
#
# std:
# Standard deviation, which measures how spread out
# the sales values are.
#
# min:
# Minimum sales value.
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
# Maximum sales value.


# print("\nTotal Sales:", df["total_sales"].sum())
# sum() calculates the total of all sales values.
#
# It gives the total sales amount for all orders.


# print("Average Sales:", df["total_sales"].mean())
# mean() calculates the average sales per order.
#
# Average = Total Sales / Number of Orders.


# print("Maximum Sales:", df["total_sales"].max())
# max() returns the highest individual sales value.


# print("Minimum Sales:", df["total_sales"].min())
# min() returns the lowest individual sales value.


# ============================================================
# 25. CATEGORY-WISE SALES ANALYSIS
# ============================================================

# category_sales = df.groupby("category")["total_sales"].sum()
# Groups records according to their product category.
#
# groupby("category"):
# Creates groups for Electronics and Furniture.
#
# ["total_sales"]:
# Selects the sales column for analysis.
#
# sum():
# Calculates the total sales for each category.
#
# The result is stored in category_sales.


# print("\nCategory-wise Sales:")
# Displays the category-wise sales heading.


# print(category_sales)
# Displays the total sales for each product category.


# ============================================================
# 26. REGION-WISE SALES ANALYSIS
# ============================================================

# region_sales = df.groupby("region")["total_sales"].sum()
# Groups sales records according to region.
#
# Calculates the total sales for each region.
#
# The result is stored in region_sales.


# print("\nRegion-wise Sales:")
# Displays the region-wise sales heading.


# print(region_sales)
# Displays the total sales for Pune, Mumbai and Nashik.


# ============================================================
# 27. MONTH-WISE SALES ANALYSIS
# ============================================================

# month_sales = df.groupby("month")["total_sales"].sum()
# Groups sales records according to month name.
#
# Calculates the total sales for each month.
#
# The result is stored in month_sales.
#
# Note:
# groupby() sorts month names alphabetically by default.
# Therefore, the graph may display April, February, January
# if those months are present, rather than calendar order.
# The sample dataset contains January, February and March.


# print("\nMonth-wise Sales:")
# Displays the month-wise sales heading.


# print(month_sales)
# Displays the total sales for each month.


# ============================================================
# 28. PRODUCT-WISE SALES ANALYSIS
# ============================================================

# product_sales = df.groupby("product")["total_sales"].sum()
# Groups records according to product name.
#
# Calculates the total sales for every product.
#
# The result is stored in product_sales.


# print("\nProduct-wise Sales:")
# Displays the product-wise sales heading.


# print(product_sales)
# Displays the total sales for each product.


# ============================================================
# 29. SAVING THE CLEANED DATASET
# ============================================================

# df.to_csv("cleaned_sales.csv", index=False)
# Saves the cleaned DataFrame into a new CSV file.
#
# to_csv() exports the DataFrame in CSV format.
#
# cleaned_sales.csv is the output filename.
#
# index=False prevents the DataFrame index
# from being stored as an additional column.


# print("\nCleaned dataset saved as cleaned_sales.csv")
# Displays a confirmation message after saving the dataset.


# ============================================================
# 30. CATEGORY-WISE SALES BAR GRAPH
# ============================================================

# category_sales.plot(kind="bar", title="Category-wise Sales")
# Creates a bar graph for category-wise sales.
#
# kind="bar" creates a vertical bar chart.
#
# Each bar represents one product category.
# The bar height represents its total sales.


# plt.xlabel("Category")
# Sets the X-axis label to Category.


# plt.ylabel("Total Sales")
# Sets the Y-axis label to Total Sales.


# plt.tight_layout()
# Adjusts the graph layout.
# Prevents labels and titles from overlapping or being cut off.


# plt.show()
# Displays the category-wise sales bar graph.


# ============================================================
# 31. REGION-WISE SALES PIE CHART
# ============================================================

# region_sales.plot(
#     kind="pie",
#     autopct="%1.1f%%",
#     title="Region-wise Sales"
# )
#
# Creates a pie chart showing region-wise sales.
#
# kind="pie":
# Creates a pie chart.
#
# autopct="%1.1f%%":
# Displays the percentage of total sales
# for each region with one decimal place.
#
# title:
# Sets the chart title.
#
# Each slice represents a region's share of total sales.


# plt.ylabel("")
# Removes the default Y-axis label from the pie chart.


# plt.tight_layout()
# Adjusts the layout to avoid overlapping elements.


# plt.show()
# Displays the region-wise sales pie chart.


# ============================================================
# 32. SALES DISTRIBUTION BOX PLOT
# ============================================================

# plt.boxplot(df["total_sales"])
# Creates a box plot using the total_sales column.
#
# A box plot is used to understand the distribution of data.
#
# It shows:
# Median.
# First quartile.
# Third quartile.
# Whiskers.
# Potential outliers.
#
# It helps identify unusually high or low sales values.


# plt.title("Sales Distribution")
# Sets the title of the box plot.


# plt.ylabel("Total Sales")
# Sets the Y-axis label.


# plt.tight_layout()
# Adjusts the graph layout.


# plt.show()
# Displays the sales distribution box plot.


# ============================================================
# 33. MONTH-WISE SALES BAR GRAPH
# ============================================================

# month_sales.plot(kind="bar", title="Month-wise Sales")
# Creates a bar graph for monthly sales.
#
# Each bar represents one month.
# The bar height represents total sales in that month.


# plt.xlabel("Month")
# Sets the X-axis label to Month.


# plt.ylabel("Total Sales")
# Sets the Y-axis label to Total Sales.


# plt.tight_layout()
# Adjusts the graph layout.


# plt.show()
# Displays the month-wise sales graph.


# ============================================================
# 34. PRODUCT-WISE SALES BAR GRAPH
# ============================================================

# product_sales.plot(kind="bar", title="Product-wise Sales")
# Creates a bar graph for product-wise sales.
#
# Each bar represents one product.
# The height of the bar represents its total sales.


# plt.xlabel("Product")
# Sets the X-axis label to Product.


# plt.ylabel("Total Sales")
# Sets the Y-axis label to Total Sales.


# plt.tight_layout()
# Adjusts the graph layout.


# plt.show()
# Displays the product-wise sales graph.


# ============================================================
# 35. COMPLETION MESSAGE
# ============================================================

# print("\nSales Data Analysis Completed Successfully!")
# Displays the final completion message.
#
# It indicates that the data loading, cleaning,
# analysis and visualization steps have finished.


# ============================================================
# 36. IMPORTANT VIVA QUESTIONS
# ============================================================

# Q1. What is DMV?
# DMV stands for Data Manipulation and Visualization.
# It involves processing, cleaning, analyzing
# and presenting data using tables and graphs.

# Q2. What is Pandas?
# Pandas is a Python library used for data manipulation
# and data analysis.

# Q3. What is a DataFrame?
# A DataFrame is a two-dimensional tabular data structure
# containing rows and columns.

# Q4. What is CSV?
# CSV stands for Comma-Separated Values.
# It stores tabular data as text separated by commas.

# Q5. What is Excel?
# Excel is a spreadsheet format used to store
# data in rows, columns and worksheets.

# Q6. What is JSON?
# JSON stands for JavaScript Object Notation.
# It is a format used to store and exchange structured data.

# Q7. What is data cleaning?
# Data cleaning is the process of identifying
# and correcting missing, duplicate or incorrect data.

# Q8. What is data manipulation?
# Data manipulation means modifying, organizing
# and transforming data into a useful format.

# Q9. What is data visualization?
# Data visualization means representing data
# using graphs, charts and plots.

# Q10. What is pd.read_csv()?
# It reads data from a CSV file into a DataFrame.

# Q11. What is pd.read_excel()?
# It reads data from an Excel file into a DataFrame.

# Q12. What is pd.read_json()?
# It reads data from a JSON file into a DataFrame.

# Q13. What is pd.concat()?
# It combines multiple Pandas objects,
# such as DataFrames, along a particular axis.

# Q14. What is isnull()?
# It identifies missing values in a DataFrame.

# Q15. What is fillna()?
# It replaces missing values with specified values.

# Q16. What is dropna()?
# It removes rows or columns containing missing values.

# Q17. What is drop_duplicates()?
# It removes duplicate records from a DataFrame.

# Q18. What is duplicated()?
# It identifies duplicate rows in a DataFrame.

# Q19. What is groupby()?
# It divides data into groups based on one or more columns
# and allows calculations on each group.

# Q20. What is describe()?
# It provides descriptive statistical information
# such as count, mean, standard deviation, minimum,
# maximum and quartiles.

# Q21. What is mean()?
# It calculates the average value.

# Q22. What is sum()?
# It calculates the total of numerical values.

# Q23. What is to_datetime()?
# It converts date values into Pandas datetime format.

# Q24. What is to_numeric()?
# It converts values into a numeric data type.

# Q25. What is iloc?
# iloc selects rows and columns using integer positions.

# Q26. What is Matplotlib?
# Matplotlib is a Python library used to create
# graphs and visualizations.

# Q27. What is a bar chart?
# A bar chart represents data using rectangular bars.
# It is useful for comparing different categories.

# Q28. What is a pie chart?
# A pie chart represents parts of a whole using slices.
# It is useful for showing percentage contributions.

# Q29. What is a box plot?
# A box plot represents data distribution
# using quartiles, median, whiskers and potential outliers.

# Q30. What is autopct?
# It controls how percentages are displayed
# on a pie chart.

# Q31. What is index=False?
# It prevents the DataFrame index from being
# saved as an additional column in the output file.

# Q32. What is the purpose of os.path.exists()?
# It checks whether a specified file or directory exists.

# Q33. What is the purpose of errors="coerce"?
# It converts invalid date or numeric values
# into missing values instead of raising an error.

# Q34. What is the purpose of tight_layout()?
# It automatically adjusts the spacing of a graph
# to prevent overlapping labels and titles.

# Q35. What are the applications of DMV?
# Sales analysis.
# Business reporting.
# Customer analysis.
# Financial analysis.
# Inventory management.
# Scientific data analysis.


# ============================================================
# 37. CONCLUSION
# ============================================================

# In this practical, sales data is created and stored
# in three different formats: CSV, Excel and JSON.

# The data is loaded using Pandas and combined into
# a single DataFrame.

# Data cleaning is performed by checking missing values,
# removing duplicate records and converting data types.

# New columns such as total_sales, month, year,
# first_name and last_name are created.

# Statistical analysis is performed using sum(),
# mean(), max(), min() and describe().

# Grouping operations are performed to calculate
# category-wise, region-wise, month-wise and product-wise sales.

# Finally, the cleaned dataset is saved as a CSV file
# and the results are displayed using bar charts,
# a pie chart and a box plot.

# Therefore, this practical demonstrates data loading,
# manipulation, cleaning, analysis and visualization
# using Python libraries.
