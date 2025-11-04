import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 1) Load data
df = pd.read_csv("supply_chain_resilience_dataset.csv")


""" df["Supply_Risk_Flag"].value_counts().plot(kind="bar")
plt.title("Supply Risk Flag Count")
plt.xlabel("Risk")
plt.ylabel("Frequency")
plt.savefig("supplySupply_Risk_Flag_bar_graph.png")
plt.close()

num_cols = [
    "Order_Value_USD",
    "Quantity_Ordered",
    "Historical_Disruption_Count",
    "Supplier_Reliability_Score",
    "Delay_Days"
]

for col in num_cols:
    df[col].plot(kind="hist")
    plt.title(f"{col} Distribution")
    plt.xlabel(col)
    plt.savefig(f"{col}_hist_graph.png")
    plt.close()

for col in num_cols:
    df.boxplot(column=col, by="Supply_Risk_Flag")
    plt.title(f"{col} vs Risk")
    plt.suptitle("")
    plt.xlabel("Risk")
    plt.ylabel(col)
    plt.savefig(f"{col}_box_plot.png")
    plt.close()
 """

""" import numpy as np

corr = df.corr(numeric_only=True)
plt.imshow(corr)
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.colorbar()
plt.title("Correlation Heatmap")
plt.savefig(f"ticks.png")
plt.close() """

""" for col in ["Product_Category", "Shipping_Mode", "Disruption_Type", "Disruption_Severity"]:
    df[col].value_counts().plot(kind="bar")
    plt.title(f"{col} Distribution")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.savefig(f"{col}_bar_graph.png")
    plt.close() """

""" for col in ["Product_Category", "Shipping_Mode", "Disruption_Type", "Disruption_Severity"]:
    pd.crosstab(df[col], df["Supply_Risk_Flag"]).plot(kind="bar", stacked=True)
    plt.title(f"{col} vs Supply Risk")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.savefig(f"{col}_bar_graph.png")
    plt.close() """


 
