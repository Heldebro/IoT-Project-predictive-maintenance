#%%
import matplotlib.pyplot as plt
import pandas as pd
import kagglehub 
from kagglehub import KaggleDatasetAdapter
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Set the path to the file you'd like to load
file_path = "machine_failure_dataset.csv"

# Load the latest version
df = kagglehub.dataset_load(
  KaggleDatasetAdapter.PANDAS,
  "mujtabamatin/dataset-for-machine-failure-detection",
  file_path,
  # Provide any additional arguments like 
  # sql_query or pandas_kwargs. See the 
  # documenation for more information:
  # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
)

x = df[["Temperature", "Vibration", "Power_Usage", "Humidity", "Machine_Type"]]
y = df["Failure_Risk"]

x = pd.get_dummies(x, columns=["Machine_Type"], drop_first=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(x)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.25, random_state=1337
)

model = LogisticRegression(class_weight="balanced", max_iter=1000, solver="liblinear")
model.fit(X_train, y_train)

y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

colors = {0: "green", 1: "red"}

def model_scatter():
  plt.scatter(
      df["Temperature"],
      df["Vibration"],
      c=df["Failure_Risk"].map(colors),
      alpha=0.6
  )

  plt.xlabel("Temperature")
  plt.ylabel("Vibration")

  # Legenda
  for value, color in colors.items():
      plt.scatter([], [], c=color, label=f"Failure = {value}")

  plt.legend()
  plt.show()

def model_report(): 
  print("Accuracy:", accuracy_score(y_test, y_pred))
  print(confusion_matrix(y_test, y_pred))
  print(classification_report(y_test, y_pred))

model_scatter()
model_report()

# print(df["Failure_Risk"].value_counts())

# plt.scatter(df["Temperature"], df["Failure_Risk"], alpha=0.3)
# plt.xlabel("Temperature")
# plt.ylabel("Failure_Risk")
# plt.show()



# print(df.head())

# plt.scatter(x, y)
# plt.show() 
# %%
