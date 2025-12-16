#%%
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

filepath = "archive/CMaps/train_FD001.txt"

df = pd.read_csv(filepath, header=None, sep=r"\s+")

# Creating column names and sorting existing colums. 
# Unit number, cycle and operation setting numb 1, 2 and 3. 
# Also adding the total of 21 sensors used in this. 
cols = (["unit_number", "current_cycle"] + [f"operation_setting_{i}" for i in range (1, 4)] + [f"sensor_{i}" for i in range (1, 22)])
df.columns = cols


# Creates max_cycle which groups the engine number with the maximum number of cycles for each engine.   
df["max_cycle"] = df.groupby("unit_number")["current_cycle"].transform("max")
#Creates "Remaining Usefull Life (RUL)" which calculates how many cycles are left. max_c - current_c
df["RUL"] = df["max_cycle"] - df["current_cycle"]
# Dropping max_cycle beacuse it is not needed anymore
df = df.drop(columns = ["max_cycle"])

#Set at threashold for 30 cycles remaining
# if the RUL values is below 30 it will classify as 1 and flagged ass Failure_risk
threshold = 30
df["Failure_risk"] = np.where(df["RUL"] <= threshold, 1, 0)

# checks whole cols and adding all operation_settings# and sensor# to feature_cols. Set these as X
# y = column "Failure risk" 
feature_cols = [c for c in cols if c.startswith("operation_setting_") or c.startswith("sensor_")]
X = df[feature_cols]
y = df["Failure_risk"]

# Train test split
# 25% for testing and rest for training
# Keeps same class distriubution from y with stratify = y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.25, random_state= 1337, stratify=y)

#300 treas seems like a good match for such a big dataset, can be alterad
model = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=1337)

# traning on X and y train
model.fit(X_train, y_train)
# Predicting with unseen data
y_pred = model.predict(X_test)

X_all = df[feature_cols]

df["pred_proba"] = model.predict_proba(X_all)[:, 1]
#Reusing threashold variable for probability and same type of comparison as Failure risk and RUL
threshold = 0.7
df["pred_label"] = np.where(df["pred_proba"] > threshold, 1, 0)

# Getting the latest cycle of each engine
# Added and ofsett so we can look at earlier cycles than the last one and get a more interesting result
ofsett = 30 # Amount of cyles before the last one. 
latest = (df.sort_values("current_cycle")).groupby("unit_number").tail(1)
latest_wofsett = (df.sort_values("current_cycle").groupby("unit_number")).nth(-ofsett)
# Creating engine at risk
engline_at_risk = latest_wofsett[latest_wofsett["pred_label"] == 1]

plt.figure(figsize=(15, 5))

colors = np.where(latest_wofsett["pred_label"] == 1, "red", "green")

latest_wofsett_sorted = latest_wofsett.sort_values("unit_number")

plt.bar(latest_wofsett_sorted["unit_number"].astype(str), latest_wofsett_sorted["pred_proba"], color=colors)
plt.xticks(rotation=90)
plt.ylabel("Predicted failure risk")
plt.xlabel("Engine (unit_number)")
plt.title("Predicted risk per engine")
plt.tight_layout()

plt.show()

# %%
