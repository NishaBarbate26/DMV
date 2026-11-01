
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
# AIM:
# Perform Data Cleaning and Data Preparation on
# Telecom Customer Churn Dataset.
#
# Steps:
# 1. Load and explore the dataset.
# 2. Remove duplicate records.
# 3. Standardize column names and text values.
# 4. Handle missing values.
# 5. Detect and handle outliers using IQR.
# 6. Perform feature engineering.
# 7. Encode categorical columns.
# 8. Split data into training and testing sets.
# 9. Normalize numerical features.
# 10. Export cleaned data.
# 11. Visualize churn distribution and missing values.
# ============================================================


# ============================================================
# 1. INSTALLATION AND EXECUTION COMMANDS
# ============================================================

# Windows:
# python -m pip install pandas numpy matplotlib scikit-learn
# python practical_telecom.py

# Ubuntu/Linux:
# python3 -m pip install pandas numpy matplotlib scikit-learn
# python3 practical_telecom.py

# Keep Telecom_Customer_Churn.csv in the same folder
# as the Python program.


# ============================================================
# 2. IMPORTING LIBRARIES
# ============================================================

# import pandas as pd
# Imports the Pandas library.
# It is used for loading, cleaning and manipulating datasets.
# pd is the short name used for Pandas.


# import numpy as np
# Imports the NumPy library.
# It is used for numerical operations.
# It also provides special values such as np.nan and np.inf.


# import matplotlib.pyplot as plt
# Imports the Pyplot module from Matplotlib.
# It is used to create graphs and visualizations.
# plt is the short name used for Pyplot.


# from sklearn.model_selection import train_test_split
# Imports train_test_split from Scikit-learn.
# It is used to divide the dataset into training and testing data.


# from sklearn.preprocessing import StandardScaler
# Imports StandardScaler from Scikit-learn.
# It is used to standardize numerical features.
# It brings numerical features to a comparable scale.


# ============================================================
# 3. LOADING THE DATASET
# ============================================================

# df = pd.read_csv("Telecom_Customer_Churn.csv")
# Reads the telecom customer churn CSV file.
#
# pd.read_csv():
# Loads data from a CSV file into a DataFrame.
#
# df:
# Stores the complete dataset.
#
# The dataset contains customer information such as:
# Customer ID.
# Gender.
# Senior citizen status.
# Tenure.
# Monthly charges.
# Total charges.
# Internet service.
# Contract type.
# Churn status.


# print("First Five Records:")
# Displays a heading for the first five records.


# print(df.head())
# Displays the first five rows of the dataset.
#
# head() displays five rows by default.
# It helps us understand the dataset structure.


# ============================================================
# 4. EXPLORING THE DATASET
# ============================================================

# print("\nDataset Shape:")
# Displays the heading for dataset dimensions.


# print(df.shape)
# Displays the number of rows and columns.
#
# Example:
# (7043, 21)
#
# This means:
# 7043 rows = 7043 customer records.
# 21 columns = 21 attributes.
#
# Actual dimensions depend on the supplied dataset.


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
# mean = Average.
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
# include="all" includes both numerical
# and categorical columns.


# print("\nMissing Values:")
# Displays the missing-value heading.


# print(df.isnull().sum())
# isnull() identifies missing values.
#
# sum() counts missing values in every column.
#
# This helps identify incomplete customer records.


# print("\nDuplicate Records:")
# Displays the duplicate-record heading.


# print(df.duplicated().sum())
# duplicated() identifies repeated rows.
#
# sum() counts the total number of duplicate records.
#
# Duplicate records can affect analysis
# because the same customer information may be counted twice.


# ============================================================
# 5. REMOVING DUPLICATE RECORDS
# ============================================================

# df = df.drop_duplicates()
# Removes duplicate rows from the dataset.
#
# drop_duplicates() keeps the first occurrence
# and removes repeated rows.
#
# The cleaned result is stored back in df.
#
# This helps avoid repeated records during analysis.


# ============================================================
# 6. STANDARDIZING COLUMN NAMES
# ============================================================

# df.columns = (
#     df.columns
#     .str.strip()
#     .str.lower()
#     .str.replace(" ", "_")
# )
#
# Standardizes all column names.
#
# df.columns:
# Accesses the names of all DataFrame columns.
#
# str.strip():
# Removes unwanted spaces from the beginning
# and end of column names.
#
# str.lower():
# Converts all column names into lowercase.
#
# str.replace(" ", "_"):
# Replaces spaces with underscores.
#
# Example:
# "Customer ID" becomes "customer_id".
#
# "Monthly Charges" becomes "monthly_charges".
#
# This makes column names easier to use in Python.


# ============================================================
# 7. STANDARDIZING TEXT VALUES
# ============================================================

# for column in df.columns:
# Starts a loop through every column in the dataset.


#     if pd.api.types.is_string_dtype(df[column].dtype):
# Checks whether the current column has a string data type.
#
# String columns contain text values.
# Examples: Male, Female, Yes, No, Month-to-month.


#         df[column] = df[column].str.strip()
# Removes unwanted spaces from the beginning
# and end of text values.
#
# Example:
# " Yes " becomes "Yes".


#         df[column] = df[column].replace("", np.nan)
# Replaces empty strings with NaN.
#
# NaN represents a missing value.
#
# This helps identify empty text fields
# as missing data.


#         df[column] = df[column].str.lower()
# Converts all text values into lowercase.
#
# Example:
# "YES" becomes "yes".
# "Female" becomes "female".
#
# This avoids treating different capitalizations
# as separate categories.


# ============================================================
# 8. IDENTIFYING THE TARGET COLUMN
# ============================================================

# target_column = "churn"
# Defines churn as the target column.
#
# The target column contains the result
# that we want to study or predict.
#
# In this dataset:
# yes = Customer has left the service.
# no = Customer has not left the service.


# if target_column not in df.columns:
# Checks whether the churn column exists
# in the dataset.


#     raise ValueError("Dataset must contain a 'Churn' column.")
# Displays an error and stops execution
# if the required target column is missing.
#
# This prevents further processing
# with an incorrect dataset.


# ============================================================
# 9. REMOVING CUSTOMER ID
# ============================================================

# if "customerid" in df.columns:
# Checks whether customerid is present
# in the DataFrame.


#     df = df.drop(columns=["customerid"])
# Removes the customerid column.
#
# Customer ID is only an identifier.
# It does not normally provide useful information
# about whether a customer will leave the service.
#
# Removing it helps prevent the model
# from learning meaningless patterns from ID values.


# ============================================================
# 10. CONVERTING TOTAL CHARGES INTO NUMERIC VALUES
# ============================================================

# if "totalcharges" in df.columns:
# Checks whether the totalcharges column exists.


#     df["totalcharges"] = pd.to_numeric(
#         df["totalcharges"],
#         errors="coerce"
#     )
#
# Converts totalcharges into numerical values.
#
# pd.to_numeric():
# Converts data into numeric format.
#
# errors="coerce":
# Converts invalid values into NaN.
#
# This is useful because some datasets
# may contain blank spaces in the TotalCharges column.
#
# Example:
# "100.50" becomes 100.50.
# An empty or invalid value becomes NaN.


# ============================================================
# 11. HANDLING ALTERNATIVE TOTAL CHARGES COLUMN NAME
# ============================================================

# if "total_charges" in df.columns:
# Checks whether the dataset uses total_charges
# instead of totalcharges.


#     df["total_charges"] = pd.to_numeric(
#         df["total_charges"],
#         errors="coerce"
#     )
#
# Converts total_charges into numeric format.
#
# This makes the program compatible with datasets
# that use either of the two column names.


# ============================================================
# 12. CONVERTING SENIOR CITIZEN COLUMN
# ============================================================

# if "seniorcitizen" in df.columns:
# Checks whether seniorcitizen is present.


#     df["seniorcitizen"] = pd.to_numeric(
#         df["seniorcitizen"],
#         errors="coerce"
#     )
#
# Converts the seniorcitizen column into numeric format.
#
# In some telecom datasets:
# 0 = Not a senior citizen.
# 1 = Senior citizen.
#
# Numeric conversion makes this column suitable
# for numerical analysis and scaling.


# ============================================================
# 13. HANDLING MISSING VALUES IN TARGET COLUMN
# ============================================================

# df = df.dropna(subset=[target_column])
# Removes rows where the churn target value is missing.
#
# subset specifies the column to check.
#
# A record without a churn label cannot be used
# for supervised machine learning training.
#
# Therefore, those rows are removed.


# ============================================================
# 14. STANDARDIZING CHURN LABELS
# ============================================================

# df[target_column] = df[target_column].replace(
#     {
#         "yes": "yes",
#         "no": "no",
#         "1": "yes",
#         "0": "no",
#         "true": "yes",
#         "false": "no"
#     }
# )
#
# Converts different representations of churn
# into two standard labels.
#
# "yes" and "1" and "true" represent churn.
#
# "no" and "0" and "false" represent no churn.
#
# This creates consistent target values.
#
# Note:
# The earlier text-cleaning step converts strings
# to lowercase, which makes these replacements consistent.


# ============================================================
# 15. KEEPING ONLY VALID CHURN LABELS
# ============================================================

# df = df[
#     df[target_column].isin(["yes", "no"])
# ].copy()
#
# Keeps only rows containing valid churn labels.
#
# isin(["yes", "no"]):
# Checks whether each target value is either yes or no.
#
# Any other target value is removed.
#
# .copy():
# Creates an independent copy of the filtered DataFrame.
#
# This helps avoid unintended modification warnings.


# ============================================================
# 16. STORING CHURN DISTRIBUTION
# ============================================================

# churn_distribution = df[target_column].value_counts()
# Counts the number of customers in each churn category.
#
# value_counts():
# Counts the occurrences of each unique value.
#
# Example:
# no = 5000 customers.
# yes = 2000 customers.
#
# These are only example values.
# Actual counts depend on the dataset.
#
# The result is stored for later visualization.


# ============================================================
# 17. FILLING MISSING VALUES
# ============================================================

# for column in df.columns:
# Starts a loop through every column in the DataFrame.


#     if column == target_column:
#         continue
#
# Checks whether the current column is the target column.
#
# continue skips the remaining statements
# in the current loop iteration.
#
# The target column is skipped because
# its missing values have already been removed.


#     if pd.api.types.is_numeric_dtype(df[column]):
# Checks whether the current column contains numerical data.
#
# Examples:
# Tenure.
# Monthly charges.
# Total charges.
# Senior citizen status.


#         df[column] = df[column].fillna(
#             df[column].median()
#         )
#
# Fills missing numerical values using the median.
#
# median():
# Finds the middle value after arranging
# numerical values in ascending order.
#
# fillna():
# Replaces missing values with the median.
#
# Median is useful because it is less affected
# by extreme values than the mean.


#     else:
#         mode_values = df[column].mode()
#
# Executes when the current column is not numerical.
#
# mode():
# Finds the most frequently occurring value
# in a column.
#
# The result is stored in mode_values.


#         if not mode_values.empty:
# Checks whether at least one mode value exists.
#
# empty returns True if the result has no values.


#             df[column] = df[column].fillna(
#                 mode_values[0]
#             )
#
# Fills missing categorical values
# using the most frequent category.
#
# mode_values[0]:
# Selects the first most frequent value.


#         else:
#             df[column] = df[column].fillna("unknown")
#
# If no mode value is available,
# missing values are replaced with "unknown".
#
# This ensures that missing categorical values
# do not remain in the dataset.


# ============================================================
# 18. IDENTIFYING NUMERICAL COLUMNS
# ============================================================

# numeric_columns = df.select_dtypes(
#     include=np.number
# ).columns.tolist()
#
# Selects all numerical columns from the DataFrame.
#
# select_dtypes():
# Selects columns based on their data types.
#
# include=np.number:
# Selects numerical data types such as integers
# and floating-point numbers.
#
# columns:
# Retrieves the selected column names.
#
# tolist():
# Converts the column names into a Python list.
#
# This list is used for outlier detection.


# ============================================================
# 19. HANDLING OUTLIERS USING IQR
# ============================================================

# for column in numeric_columns:
# Starts a loop through every numerical column.


#     Q1 = df[column].quantile(0.25)
# Calculates the first quartile.
#
# Q1 represents the 25th percentile.
#
# Approximately 25% of the observations
# are at or below this value.


#     Q3 = df[column].quantile(0.75)
# Calculates the third quartile.
#
# Q3 represents the 75th percentile.
#
# Approximately 75% of the observations
# are at or below this value.


#     IQR = Q3 - Q1
# Calculates the Interquartile Range.
#
# IQR represents the spread of the middle 50%
# of the numerical data.


#     lower_limit = Q1 - 1.5 * IQR
# Calculates the lower outlier limit.
#
# Values below this limit are considered
# potential lower outliers.


#     upper_limit = Q3 + 1.5 * IQR
# Calculates the upper outlier limit.
#
# Values above this limit are considered
# potential upper outliers.


#     df[column] = df[column].clip(
#         lower_limit,
#         upper_limit
#     )
#
# Handles outliers using the clipping method.
#
# clip():
# Replaces values outside the specified limits.
#
# Values below the lower limit become the lower limit.
#
# Values above the upper limit become the upper limit.
#
# Values within the limits remain unchanged.
#
# This method is also called winsorization.
#
# Important:
# It does not delete outlier records.
# It reduces the effect of extreme values
# by replacing them with boundary values.


# ============================================================
# 20. FEATURE ENGINEERING
# ============================================================

# Feature engineering means creating new useful columns
# from existing columns to improve data analysis
# and machine learning.


# ============================================================
# 21. CREATING AVERAGE MONTHLY CHARGE
# ============================================================

# if "monthlycharges" in df.columns and "tenure" in df.columns:
# Checks whether both monthlycharges and tenure
# columns are available.


#     df["average_charge_per_month"] = df["monthlycharges"]
# Creates a new column named average_charge_per_month.
#
# It copies the values from monthlycharges.
#
# Important:
# This is not a new calculated average.
# It is simply a duplicate of the monthlycharges column.
#
# It may be unnecessary because the original column
# already contains the same information.


# ============================================================
# 22. CREATING ESTIMATED CUSTOMER LIFETIME VALUE
# ============================================================

# if "monthlycharges" in df.columns and "tenure" in df.columns:
# Checks whether monthlycharges and tenure are available.


#     df["estimated_lifetime_value"] = (
#         df["monthlycharges"] * df["tenure"]
#     )
#
# Creates a new feature called estimated_lifetime_value.
#
# Multiplies monthly charges by customer tenure.
#
# Formula:
# Estimated Lifetime Value =
# Monthly Charges × Tenure in Months.
#
# Example:
# Monthly Charges = 100.
# Tenure = 12 months.
# Estimated Lifetime Value = 100 × 12 = 1200.
#
# This is a simple estimated revenue measure.
# It is not necessarily the customer's true lifetime value.


# ============================================================
# 23. CREATING TENURE GROUPS
# ============================================================

# if "tenure" in df.columns:
# Checks whether the tenure column exists.


#     df["tenure_group"] = pd.cut(
#         df["tenure"],
#         bins=[-1, 12, 24, 48, np.inf],
#         labels=[
#             "new",
#             "medium",
#             "long",
#             "very_long"
#         ]
#     )
#
# Divides customers into groups based on tenure.
#
# pd.cut():
# Converts numerical values into categories
# using specified intervals.
#
# bins:
# Defines the boundaries of the groups.
#
# -1 to 12:
# New customers.
#
# More than 12 to 24:
# Medium-tenure customers.
#
# More than 24 to 48:
# Long-tenure customers.
#
# More than 48:
# Very-long-tenure customers.
#
# np.inf:
# Represents positive infinity.
# It allows the last group to include all values above 48.
#
# labels:
# Assigns a name to each interval.
#
# The resulting categories are stored
# in the tenure_group column.


# ============================================================
# 24. ENCODING CATEGORICAL COLUMNS
# ============================================================

# df = pd.get_dummies(
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
# Gender values:
# male, female.
#
# After encoding:
# gender_male = 0 or 1.
#
# drop_first=True:
# Removes the first category from each encoded feature.
#
# This helps avoid redundant dummy variables
# for a category represented by two values.
#
# dtype=int:
# Stores encoded values as integers, usually 0 and 1.
#
# Important:
# The target churn column is also encoded here.
# The resulting target column is expected to be churn_yes.


# ============================================================
# 25. SEPARATING FEATURES AND TARGET
# ============================================================

# target_columns = [
#     column for column in df.columns
#     if column == "churn_yes"
# ]
#
# Finds the encoded target column named churn_yes.
#
# This is a list comprehension.
#
# It checks every column name in the DataFrame.
#
# If a column is named churn_yes,
# it is included in target_columns.


# if not target_columns:
# Checks whether the churn_yes column was found.


#     raise ValueError("Unable to encode churn target.")
# Stops the program if the encoded target column
# could not be found.
#
# This may happen if the original target labels
# or dataset format are different from expected.


# X = df.drop(columns=target_columns)
# Creates the feature dataset.
#
# X contains all input columns except the target.
#
# Examples:
# Tenure.
# Monthly charges.
# Internet service.
# Contract type.
# Payment method.
#
# These features are used to predict customer churn.


# y = df["churn_yes"]
# Creates the target variable.
#
# y contains the churn result.
#
# 1 represents yes, meaning the customer churned.
#
# 0 represents no, meaning the customer did not churn.
#
# X = Input features.
# y = Output or target variable.


# ============================================================
# 26. SPLITTING DATASET INTO TRAINING AND TESTING DATA
# ============================================================

# X_train, X_test, y_train, y_test = train_test_split(
#     X,
#     y,
#     test_size=0.2,
#     random_state=42,
#     stratify=y
# )
#
# Divides the dataset into training and testing sets.
#
# X:
# Contains input features.
#
# y:
# Contains target values.
#
# test_size=0.2:
# Uses 20% of the data for testing.
# The remaining 80% is used for training.
#
# random_state=42:
# Ensures the same random split is produced
# when the program is run again with the same data.
#
# stratify=y:
# Maintains approximately the same churn class proportions
# in both training and testing sets.
#
# X_train:
# Input features used for training.
#
# X_test:
# Input features used for testing.
#
# y_train:
# Target values used for training.
#
# y_test:
# Target values used for testing.


# ============================================================
# 27. NORMALIZING NUMERICAL FEATURES
# ============================================================

# scaler = StandardScaler()
# Creates a StandardScaler object.
#
# StandardScaler standardizes numerical features.
#
# Formula:
#
# Standardized Value = (X - Mean) / Standard Deviation
#
# After standardization:
# Mean is approximately 0.
# Standard deviation is approximately 1.
#
# This helps numerical features
# become comparable in scale.


# ============================================================
# 28. FITTING AND TRANSFORMING TRAINING DATA
# ============================================================

# X_train = pd.DataFrame(
#     scaler.fit_transform(X_train),
#     columns=X.columns,
#     index=X_train.index
# )
#
# fit_transform():
# First learns the mean and standard deviation
# from the training data.
#
# Then uses those values to standardize
# the training data.
#
# pd.DataFrame():
# Converts the scaled array back into a DataFrame.
#
# columns=X.columns:
# Restores the original feature column names.
#
# index=X_train.index:
# Restores the original row indexes.
#
# The standardized training data is stored in X_train.


# ============================================================
# 29. TRANSFORMING TESTING DATA
# ============================================================

# X_test = pd.DataFrame(
#     scaler.transform(X_test),
#     columns=X.columns,
#     index=X_test.index
# )
#
# transform():
# Standardizes the testing data using
# the mean and standard deviation learned
# from the training data.
#
# Important:
# We use transform(), not fit_transform(),
# on testing data.
#
# This prevents information from the test dataset
# from influencing the scaling parameters.
#
# The result is converted back into a DataFrame
# with the original column names and indexes.


# ============================================================
# 30. EXPORTING CLEANED DATASET
# ============================================================

# cleaned_data = pd.concat(
#     [X, y.rename("churn")],
#     axis=1
# )
#
# Combines the feature dataset and target variable.
#
# pd.concat():
# Joins DataFrames or Series together.
#
# X:
# Contains all input features.
#
# y.rename("churn"):
# Renames the target Series to churn.
#
# axis=1:
# Combines the data column-wise.
#
# The result is stored in cleaned_data.


# cleaned_data.to_csv(
#     "cleaned_telecom_churn.csv",
#     index=False
# )
#
# Saves the cleaned dataset as a CSV file.
#
# index=False:
# Does not save the DataFrame index
# as an additional column.


# ============================================================
# 31. EXPORTING TRAINING AND TESTING DATA
# ============================================================

# X_train.to_csv("X_train.csv", index=False)
# Saves the standardized training features
# into X_train.csv.


# X_test.to_csv("X_test.csv", index=False)
# Saves the standardized testing features
# into X_test.csv.


# y_train.to_csv("y_train.csv", index=False)
# Saves the training target values
# into y_train.csv.


# y_test.to_csv("y_test.csv", index=False)
# Saves the testing target values
# into y_test.csv.


# ============================================================
# 32. DISPLAYING FINAL CLEANED DATASET
# ============================================================

# print("\nCleaned Dataset:")
# Displays the heading for the cleaned dataset.


# print(cleaned_data.head())
# Displays the first five rows of the cleaned dataset.


# print("\nFinal Dataset Shape:")
# Displays the final dataset dimensions heading.


# print(cleaned_data.shape)
# Displays the final number of rows and columns
# after cleaning, transformation and encoding.


# ============================================================
# 33. DISPLAYING TRAINING DATA SHAPE
# ============================================================

# print("\nTraining Data Shape:")
# Displays the training data heading.


# print(X_train.shape)
# Displays the number of rows and columns
# in the training feature dataset.
#
# Approximately 80% of the records are used for training.


# ============================================================
# 34. DISPLAYING TESTING DATA SHAPE
# ============================================================

# print("\nTesting Data Shape:")
# Displays the testing data heading.


# print(X_test.shape)
# Displays the number of rows and columns
# in the testing feature dataset.
#
# Approximately 20% of the records are used for testing.


# ============================================================
# 35. DISPLAYING TRAINING TARGET DISTRIBUTION
# ============================================================

# print("\nTraining Target Distribution:")
# Displays the training target heading.


# print(y_train.value_counts())
# Counts the number of churn and non-churn customers
# in the training target.
#
# 0 = No churn.
# 1 = Churn.
#
# This helps understand the class distribution.


# ============================================================
# 36. DISPLAYING TESTING TARGET DISTRIBUTION
# ============================================================

# print("\nTesting Target Distribution:")
# Displays the testing target heading.


# print(y_test.value_counts())
# Counts the number of churn and non-churn customers
# in the testing target.
#
# stratify=y helps maintain a similar class distribution
# in the training and testing datasets.


# ============================================================
# 37. VISUALIZING CUSTOMER CHURN DISTRIBUTION
# ============================================================

# plt.figure(figsize=(6, 5))
# Creates a new graph with width 6 inches
# and height 5 inches.


# churn_distribution.plot(kind="bar")
# Creates a bar chart using the churn distribution.
#
# kind="bar":
# Specifies a bar chart.
#
# The graph compares:
# Number of customers who churned.
# Number of customers who did not churn.


# plt.title("Customer Churn Distribution")
# Sets the graph title.


# plt.xlabel("Churn")
# Sets the X-axis label.
#
# The categories are yes and no.


# plt.ylabel("Number of Customers")
# Sets the Y-axis label.
#
# It represents the number of customers in each category.


# plt.tight_layout()
# Automatically adjusts graph spacing.


# plt.show()
# Displays the customer churn distribution chart.


# ============================================================
# 38. VISUALIZING MISSING VALUES AFTER CLEANING
# ============================================================

# plt.figure(figsize=(8, 4))
# Creates a new graph with width 8 inches
# and height 4 inches.


# missing_values = cleaned_data.isnull().sum()
# Checks for missing values in the cleaned dataset.
#
# isnull():
# Identifies missing values.
#
# sum():
# Counts missing values in every column.
#
# The result is stored in missing_values.


# missing_values.plot(kind="bar")
# Creates a bar chart showing missing values
# in each column.


# plt.title("Missing Values After Cleaning")
# Sets the graph title.


# plt.xlabel("Columns")
# Sets the X-axis label.
#
# Each category represents a dataset column.


# plt.ylabel("Missing Values")
# Sets the Y-axis label.
#
# It shows the number of missing values.


# plt.tight_layout()
# Adjusts the graph layout.


# plt.show()
# Displays the missing-value bar chart.
#
# If cleaning was successful,
# all bars should have a value of zero.


# ============================================================
# 39. FINAL COMPLETION MESSAGE
# ============================================================

# print("\nData Cleaning and Preparation Completed Successfully!")
#
# Displays a completion message.
#
# It indicates that the dataset has been cleaned,
# transformed, split, standardized and exported.


# ============================================================
# 40. CONCLUSION
# ============================================================

# In this practical, the Telecom Customer Churn dataset
# is loaded using Pandas.

# Dataset exploration is performed using head(),
# shape, info(), describe() and isnull().

# Duplicate records are removed.

# Column names and text values are standardized.

# Missing values are handled using median and mode.

# Outliers are handled using the IQR method.

# New features are created using feature engineering.

# Categorical variables are converted into numerical
# values using one-hot encoding.

# The dataset is divided into training and testing sets.

# Numerical features are standardized using StandardScaler.

# Cleaned data, training data and testing data
# are saved as CSV files.

# Finally, churn distribution and missing values
# are visualized using bar charts.

# Therefore, this practical demonstrates the complete
# data cleaning and preparation process for telecom
# customer churn analysis.
