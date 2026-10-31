
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


# EXPLANATION:
#
# 1. import pandas as pd
#    Imports Pandas for reading, processing and analyzing data.
#
# 2. import numpy as np
#    Imports NumPy for numerical operations.
#
# 3. import matplotlib.pyplot as plt
#    Imports Matplotlib for creating graphs and charts.
#
# 4. import os
#    Provides functions to check whether files exist.
#
# 5. data = {...}
#    Creates sample sales records containing order details,
#    customer names, products, categories, quantity and region.
#
# 6. sample_df = pd.DataFrame(data)
#    Converts the sample records into a Pandas DataFrame.
#
# 7. os.path.exists()
#    Checks whether the CSV, Excel and JSON files already exist.
#
# 8. to_csv()
#    Saves the first two records into a CSV file.
#
# 9. to_excel()
#    Saves the next two records into an Excel file.
#
# 10. to_json()
#     Saves the remaining records into a JSON file.
#
# 11. pd.read_csv()
#     Reads sales data from the CSV file.
#
# 12. pd.read_excel()
#     Reads sales data from the Excel file.
#
# 13. pd.read_json()
#     Reads sales data from the JSON file.
#
# 14. print()
#     Displays the loaded datasets.
#
# 15. info()
#     Displays the number of rows, columns and data types.
#
# 16. isnull().sum()
#     Counts missing values in every column.
#
# 17. pd.concat()
#     Combines all three datasets into one DataFrame.
#
# 18. df.columns
#     Converts column names into lowercase and replaces spaces
#     with underscores.
#
# 19. duplicated().sum()
#     Counts duplicate records in the combined dataset.
#
# 20. drop_duplicates()
#     Removes duplicate records.
#
# 21. pd.to_datetime()
#     Converts the order date column into date format.
#
# 22. pd.to_numeric()
#     Converts order ID, quantity and unit price into numbers.
#
# 23. fillna()
#     Replaces missing values with suitable default values.
#
# 24. dropna()
#     Removes records having invalid order dates or order IDs.
#
# 25. total_sales
#     Calculates total sales by multiplying quantity by unit price.
#
# 26. dt.month_name()
#     Extracts the month name from the order date.
#
# 27. dt.year
#     Extracts the year from the order date.
#
# 28. str.partition()
#     Splits customer names into first name and last name.
#     It also handles names containing only one word.
#
# 29. describe()
#     Displays statistical information about sales.
#
# 30. sum()
#     Calculates the total sales amount.
#
# 31. mean()
#     Calculates the average sales amount.
#
# 32. max()
#     Finds the highest sales amount.
#
# 33. min()
#     Finds the lowest sales amount.
#
# 34. groupby()
#     Groups sales records according to category, region,
#     month and product.
#
# 35. to_csv()
#     Saves the cleaned dataset into cleaned_sales.csv.
#
# 36. plot(kind="bar")
#     Creates bar charts for comparing sales.
#
# 37. plot(kind="pie")
#     Creates a pie chart showing region-wise sales.
#
# 38. plt.boxplot()
#     Creates a box plot showing sales distribution.
#
# 39. plt.xlabel()
#     Sets the label of the X-axis.
#
# 40. plt.ylabel()
#     Sets the label of the Y-axis.
#
# 41. plt.title()
#     Sets the title of the graph.
#
# 42. plt.tight_layout()
#     Adjusts the graph layout to prevent overlapping.
#
# 43. plt.show()
#     Displays the graph.
#
# 44. Final print statement
#     Displays the successful completion message.
