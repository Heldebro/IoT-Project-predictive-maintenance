#%%
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

filepath = "archive/CMaps/train_FD001.txt"

df = pd.read_csv(filepath, delim_whitespace=True, header=None)

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
threashold = 30
df["Failure_risk"] = np.where(df["RUL"] <= threashold, 1, 0)

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

#Compare real value with predicted
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
# %%
