
# AIM: Perform Data Cleaning and Preparation on
# Telecom Customer Churn Dataset.

# Installation Commands:
#
# Windows:
# python -m pip install pandas numpy scikit-learn matplotlib
# python DMV3_customer_churn.py
#
# Ubuntu / Linux:
# sudo apt update
# sudo apt install python3 python3-pip
# python3 -m pip install pandas numpy scikit-learn matplotlib
# python3 DMV3_customer_churn.py
#
# For Ubuntu using a normal text editor:
# Save this file as DMV3_customer_churn.py
# Open Terminal in the file's folder and execute the commands above.


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# 1. Load dataset

df = pd.read_csv("Telecom_Customer_Churn.csv")

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

print("\nDuplicate Records:")
print(df.duplicated().sum())


# 3. Remove duplicate records

df = df.drop_duplicates()


# 4. Standardize column names

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)


# 5. Standardize text values

for column in df.columns:

    if pd.api.types.is_string_dtype(df[column].dtype):

        df[column] = df[column].str.strip()

        df[column] = df[column].replace("", np.nan)

        df[column] = df[column].str.lower()


# 6. Identify target column

target_column = "churn"

if target_column not in df.columns:
    raise ValueError("Dataset must contain a 'Churn' column.")


# Remove customer ID because it is only an identifier

if "customerid" in df.columns:
    df = df.drop(columns=["customerid"])


# Convert total charges into numeric values

if "totalcharges" in df.columns:

    df["totalcharges"] = pd.to_numeric(
        df["totalcharges"],
        errors="coerce"
    )

if "total_charges" in df.columns:

    df["total_charges"] = pd.to_numeric(
        df["total_charges"],
        errors="coerce"
    )


# Convert senior citizen column into numeric values

if "seniorcitizen" in df.columns:

    df["seniorcitizen"] = pd.to_numeric(
        df["seniorcitizen"],
        errors="coerce"
    )


# 7. Handle missing values

# Remove records with missing target values

df = df.dropna(subset=[target_column])

# Standardize churn labels

df[target_column] = df[target_column].replace(
    {
        "yes": "yes",
        "no": "no",
        "1": "yes",
        "0": "no",
        "true": "yes",
        "false": "no"
    }
)

# Keep only valid churn labels

df = df[
    df[target_column].isin(["yes", "no"])
].copy()

# Store churn distribution for visualization

churn_distribution = df[target_column].value_counts()

# Fill missing values according to data type

for column in df.columns:

    if column == target_column:
        continue

    if pd.api.types.is_numeric_dtype(df[column]):

        df[column] = df[column].fillna(
            df[column].median()
        )

    else:

        mode_values = df[column].mode()

        if not mode_values.empty:

            df[column] = df[column].fillna(
                mode_values[0]
            )

        else:

            df[column] = df[column].fillna("unknown")


# 8. Identify and handle outliers using IQR

numeric_columns = df.select_dtypes(
    include=np.number
).columns.tolist()

for column in numeric_columns:

    Q1 = df[column].quantile(0.25)

    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR

    upper_limit = Q3 + 1.5 * IQR

    df[column] = df[column].clip(
        lower_limit,
        upper_limit
    )


# 9. Feature engineering

# Create average monthly charge

if "monthlycharges" in df.columns and "tenure" in df.columns:

    df["average_charge_per_month"] = df["monthlycharges"]


# Create estimated customer lifetime value

if "monthlycharges" in df.columns and "tenure" in df.columns:

    df["estimated_lifetime_value"] = (
        df["monthlycharges"] * df["tenure"]
    )


# Create tenure groups

if "tenure" in df.columns:

    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, np.inf],
        labels=[
            "new",
            "medium",
            "long",
            "very_long"
        ]
    )


# 10. Encode categorical columns

df = pd.get_dummies(
    df,
    drop_first=True,
    dtype=int
)


# 11. Separate features and target

target_columns = [
    column for column in df.columns
    if column == "churn_yes"
]

if not target_columns:
    raise ValueError("Unable to encode churn target.")

X = df.drop(columns=target_columns)

y = df["churn_yes"]


# 12. Split dataset

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 13. Normalize numerical features

scaler = StandardScaler()

X_train = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=X.columns,
    index=X_train.index
)

X_test = pd.DataFrame(
    scaler.transform(X_test),
    columns=X.columns,
    index=X_test.index
)


# 14. Export cleaned dataset

cleaned_data = pd.concat(
    [X, y.rename("churn")],
    axis=1
)

cleaned_data.to_csv(
    "cleaned_telecom_churn.csv",
    index=False
)


# Export training and testing data

X_train.to_csv("X_train.csv", index=False)

X_test.to_csv("X_test.csv", index=False)

y_train.to_csv("y_train.csv", index=False)

y_test.to_csv("y_test.csv", index=False)


# 15. Display final results

print("\nCleaned Dataset:")
print(cleaned_data.head())

print("\nFinal Dataset Shape:")
print(cleaned_data.shape)

print("\nTraining Data Shape:")
print(X_train.shape)

print("\nTesting Data Shape:")
print(X_test.shape)

print("\nTraining Target Distribution:")
print(y_train.value_counts())

print("\nTesting Target Distribution:")
print(y_test.value_counts())


# 16. Visualize churn distribution

plt.figure(figsize=(6, 5))

churn_distribution.plot(kind="bar")

plt.title("Customer Churn Distribution")

plt.xlabel("Churn")

plt.ylabel("Number of Customers")

plt.tight_layout()

plt.show()


# 17. Visualize missing values after cleaning

plt.figure(figsize=(8, 4))

missing_values = cleaned_data.isnull().sum()

missing_values.plot(kind="bar")

plt.title("Missing Values After Cleaning")

plt.xlabel("Columns")

plt.ylabel("Missing Values")

plt.tight_layout()

plt.show()


print("\nData Cleaning and Preparation Completed Successfully!")


# ============================================================
# EXPLANATION LINE BY LINE
# ============================================================

# import pandas as pd
# Imports Pandas for loading and processing the dataset.

# import numpy as np
# Imports NumPy for numerical calculations.

# import matplotlib.pyplot as plt
# Imports Matplotlib for creating graphs.

# train_test_split
# Divides the dataset into training and testing data.

# StandardScaler
# Standardizes numerical features.

# pd.read_csv()
# Loads the telecom customer churn CSV dataset.

# df.head()
# Displays the first five records.

# df.shape
# Displays the number of rows and columns.

# df.info()
# Displays column names, data types and non-null values.

# describe()
# Displays statistical information about the dataset.

# isnull().sum()
# Counts missing values in every column.

# duplicated().sum()
# Counts duplicate records.

# drop_duplicates()
# Removes duplicate records.

# df.columns
# Accesses all column names.

# str.strip()
# Removes unnecessary spaces from column names and text values.

# str.lower()
# Converts text values into lowercase.

# replace()
# Replaces empty strings with missing values.

# target_column
# Identifies the churn column as the target variable.

# drop(columns=["customerid"])
# Removes customer ID because it is a unique identifier.

# pd.to_numeric()
# Converts total charges and senior citizen values into numbers.

# errors="coerce"
# Converts invalid numeric values into missing values.

# dropna()
# Removes records with missing churn values.

# value_counts()
# Counts the number of customers in each churn category.

# pd.api.types.is_numeric_dtype()
# Checks whether a column contains numeric data.

# median()
# Calculates the middle value of a numerical column.

# mode()
# Finds the most frequently occurring value in a column.

# fillna()
# Replaces missing values with suitable values.

# select_dtypes()
# Selects numerical columns from the dataset.

# quantile()
# Calculates quartiles for detecting outliers.

# IQR
# Calculates the interquartile range.

# clip()
# Replaces extreme values with the lower and upper limits.

# Feature engineering
# Creates new features using existing customer information.

# average_charge_per_month
# Stores the customer's monthly charges.

# estimated_lifetime_value
# Estimates customer value using monthly charges and tenure.

# pd.cut()
# Divides customer tenure into different groups.

# pd.get_dummies()
# Converts categorical columns into numerical columns.

# drop_first=True
# Removes one category to avoid redundant dummy columns.

# X
# Stores input features.

# y
# Stores the target churn values.

# train_test_split()
# Divides the dataset into 80% training and 20% testing data.

# stratify=y
# Maintains the churn distribution in both datasets.

# StandardScaler()
# Creates a standardization object.

# fit_transform()
# Learns scaling values from training data and transforms it.

# transform()
# Applies the same scaling to testing data.

# pd.concat()
# Combines input features and target values.

# to_csv()
# Exports cleaned data into CSV files.

# plt.figure()
# Creates a new figure.

# plot(kind="bar")
# Creates a bar chart.

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

# Final print statement
# Displays the successful completion message.
