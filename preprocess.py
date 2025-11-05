import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 1) Load data
df = pd.read_csv("supply_chain_resilience_dataset.csv")

# 2) Basic exploration
df["Disruption_Type"] = df["Disruption_Type"].fillna("None")
df["Disruption_Severity"] = df["Disruption_Severity"].fillna("None")
print("Shape (rows, columns):", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nData Types:\n", df.dtypes)
print("\nMissing values:\n", df.isnull().sum())
print("\nSummary Statistics (numeric):\n", df.describe())
print("\nSummary Statistics (non-numeric):\n", df.describe(include='object'))

# 3) Drop ID-like / unwanted columns
cols_to_drop = [
    "Order_ID",
    "Data_Sharing_Consent",
    "Federated_Round",
    "Parameter_Change_Magnitude",
    "Communication_Cost_MB",
    "Energy_Consumption_Joules"
]
df.drop(columns=cols_to_drop, inplace=True, errors="ignore")



# 4) Convert date columns to datetime
date_cols = ["Order_Date", "Dispatch_Date", "Delivery_Date"]
df[date_cols] = df[date_cols].apply(pd.to_datetime)

# 5) Convert selected columns to categorical
cat_cols = [
    "Product_Category",
    "Shipping_Mode",
    "Disruption_Type",
    "Disruption_Severity"
]
df[cat_cols] = df[cat_cols].astype("category")

# 6) Output final schema + preview
print("\nFinal Data Types:\n", df.dtypes)
print("\nDataFrame preview:\n", df.head())
print("\nFinal shape:", df.shape)

df.to_csv("preprocessed_supply_chain_resilience_dataset.csv", index=False)