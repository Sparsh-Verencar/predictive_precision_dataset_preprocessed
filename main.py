import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# ===============================
# 1) Load data
# ===============================
df = pd.read_csv("supply_chain_resilience_dataset.csv")
print("before dropping:")
print(df.dtypes)
print(df.shape)


# ===============================
# 2) Drop useless cols
# ===============================
cols_to_drop = [
    "Order_ID", 
    "Data_Sharing_Consent", 
    "Federated_Round", 
    "Parameter_Change_Magnitude", 
    "Communication_Cost_MB", 
    "Energy_Consumption_Joules"
]
df.drop(columns=cols_to_drop, inplace=True)
print("\nafter dropping:")
print(df.dtypes)
print(df.shape)


# ===============================
# 3) Fill categorical NA
# ===============================
df["Disruption_Type"].fillna(df["Disruption_Type"].mode()[0], inplace=True)
df["Disruption_Severity"].fillna(df["Disruption_Severity"].mode()[0], inplace=True)
print("\nnull values:")
print(df.isnull().sum())


# ===============================
# 4) Convert date columns → int
# ===============================
date_cols = ["Order_Date", "Dispatch_Date", "Delivery_Date"]

for c in date_cols:
    df[c] = pd.to_datetime(df[c], errors="coerce")
    df[c] = (df[c] - pd.Timestamp("1970-01-01")) // pd.Timedelta("1D")   # convert to days


print("\nafter date conversion:")
print(df.dtypes)


# ===============================
# 5) Detect outliers
# ===============================
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

def find_outliers(col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[col] < lower) | (df[col] > upper)]

outlier_dict = {}

for col in numeric_cols:
    outlier_dict[col] = find_outliers(col)
    print(f"\nColumn: {col}")
    print("Outlier count:", len(outlier_dict[col]))


# ===============================
# 6) Encode categorical
# ===============================
categorical_cols = df.select_dtypes(include="object").columns

le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])


# ===============================
# 7) Correlation Heatmap
# ===============================
corr = df.corr(numeric_only=True)

plt.figure(figsize=(10, 6))
sns.heatmap(corr, annot=True)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=300)


# ===============================
# 8) Scale data
# ===============================
scaler = MinMaxScaler()
scaled_df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

print("\nScaled DataFrame:")
print(scaled_df.head())


# ===============================
# 9) Save processed CSV
# ===============================

#scaled_df.to_csv("processed_dataset.csv", index=False)


# ===============================
# 10) Train XGBoost (w Delay)
# ===============================
target = "Supply_Risk_Flag"

X1 = scaled_df.drop(columns=[target])
y = scaled_df[target]

X1_train, X1_test, y_train, y_test = train_test_split(
    X1, y, test_size=0.2, random_state=42
)

modelWithDelay = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss"
)

modelWithDelay.fit(X1_train, y_train)
pred1 = modelWithDelay.predict(X1_test)

print("\n=========== Model 1 (WITH Delay_Days) ===========")
print("Accuracy:", accuracy_score(y_test, pred1))
print(classification_report(y_test, pred1))


# ===============================
# 11) Train XGBoost (w/o Delay)
# ===============================
X2 = scaled_df.drop(columns=[target, "Delay_Days"])

X2_train, X2_test, y_train, y_test = train_test_split(
    X2, y, test_size=0.2, random_state=42
)

corr = X2.corr(numeric_only=True)

plt.figure(figsize=(10, 6))
sns.heatmap(corr, annot=True)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("X2_correlation_heatmap.png", dpi=300)

modelWithoutDelay = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss"
)

modelWithoutDelay.fit(X2_train, y_train)
pred2 = modelWithoutDelay.predict(X2_test)

print("\n=========== Model 2 (WITHOUT Delay_Days) ===========")
print("Accuracy:", accuracy_score(y_test, pred2))
print(classification_report(y_test, pred2))


# ===============================
# 12) Compare results
# ===============================
acc1 = accuracy_score(y_test, pred1)
acc2 = accuracy_score(y_test, pred2)

print("\nModel-1 Accuracy (With Delay_Days):", acc1)
print("Model-2 Accuracy (Without Delay_Days):", acc2)

# Save
with open("xgb_model_with_delay.pkl", "wb") as f:
    pickle.dump(modelWithDelay, f)

with open("xgb_model_without_delay.pkl", "wb") as f:
    pickle.dump(modelWithoutDelay, f)

print("✅ Models saved using pickle!")


