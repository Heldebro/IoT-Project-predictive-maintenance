# Dataset:
# https://www.kaggle.com/datasets/palbha/cmapss-jet-engine-simulated-data
import time
import json
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# Loading and labling data
def load_cmapss_train(filepath: str):

    df = pd.read_csv(filepath, header=None, sep=r"\s+")

    cols = (["unit_number", "current_cycle"] + [f"operation_setting_{i}" for i in range(1, 4)] + [f"sensor_{i}" for i in range(1, 22)])

    df.columns = cols
    df.columns = [c.strip() for c in df.columns]  # Removes potential whitespace in the col-names

    # Listing al operating_settings and sensor cols
    feature_cols = [] 
    for c in df.columns:
        if c.startswith("operation_setting_") or c.startswith("sensor_"):
            feature_cols.append(c) # Adding them to the list
    return df, feature_cols


def add_rul_and_label(df: pd.DataFrame, rul_threshold: int = 30): # number of cycles before failure is the threashold for "dangerzone"

    df = df.copy()
    df["max_cycle"] = df.groupby("unit_number")["current_cycle"].transform("max") # Adding max_cycles for each engine
    df["RUL"] = df["max_cycle"] - df["current_cycle"] #Remaing usefull life for each engine
    df["Failure_risk"] = (df["RUL"] <= rul_threshold).astype(int) #Compares RUL with rul_threshold int
    return df


#Splitting the data to X and y
def train_rf_classifier(df: pd.DataFrame, feature_cols):

    X = df[feature_cols]
    y = df["Failure_risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=1337, stratify=y # stratify on y to preserve class imblanace in train and test. Get representative splits 
    )
    #n_estimator= number of trees, class_weight = favor true/pos instead of false/neg, randome_state = ELITE
    model = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=1337)
    model.fit(X_train, y_train)

    #using my model to predict y_pred based on X_test
    y_pred = model.predict(X_test)
    print("\n=== Classification report ===")
    print(classification_report(y_test, y_pred))

    return model


#Creating the somulation
def snapshot_at_cycle(df: pd.DataFrame, t: int):
    snap = df[df["current_cycle"] == t].copy() # making a copy of each engine on simulated time t, adding to df snap

    
    max_cycle = df.groupby("unit_number")["current_cycle"].max()
    #Creating two lists
    failed_units = max_cycle[max_cycle < t].index.tolist() # Units who have max_cycle lower than t are added to failed_units list
    just_failed = max_cycle[max_cycle == t].index.tolist() # Compare units current cycle with max_cycle. Displaying only the units who failed this cycle

    return snap, failed_units, just_failed


def top_k_risk(snap: pd.DataFrame, feature_cols, model, k: int = 5, exclude_units=None):

    #Failsafe to give return even when the snap list is empty, t is to low or to high
    if snap.empty:
        return [] #Function needs a list to return, otherwise ValuError

    snap = snap.copy()

    if exclude_units: # Standard none units to exclude but is replaced by just_failed list when it's filled. None = false and will therefore be skipped until just_failed is filled
        snap = snap[~snap["unit_number"].isin(exclude_units)].copy() # Keeps rows which are NOT in exclude_units. Comparing cols. 
        if snap.empty: # failsafe if every row if filtered away
            return [] # ex. There was engines in snap but they were al a part of exclude_units
    snap["pred_proba"] = model.predict_proba(snap[feature_cols])[:, 1] #Prediction is made on each engine, proba between 0 and 1. Simple bolean would not be good for ranking

    top = snap.sort_values("pred_proba", ascending=False).head(k) # Sorting snap list and showing top k 

    # JSON-friendly list, might get Scaler error but int and float shoudl be accepted. Can be ignored
    out = []
    for r in top.itertuples(index=False): #Does not include index#
        out.append({
            "unit_number": int(r.unit_number), 
            "cycle": int(r.current_cycle),
            "pred_proba": float(r.pred_proba),
            "rul": int(r.RUL) if hasattr(r, "RUL") else None
        })
    return out

# Parameters for simluation of real time stream of data, can be changed but will be overwritten by the main() paramters
def simulate_stream(
    df: pd.DataFrame,
    feature_cols,
    model,
    start_t: int = 1,
    end_t: int | None = None,
    tick_seconds: float = 1.0, #Tic time in sec
    top_k: int = 5,
    print_pretty: bool = False
):
    # set the max time of simlutation t, if not max value is set then run intill data is out.   
    max_t = int(df["current_cycle"].max()) if end_t is None else int(end_t)

    for t in range(int(start_t), max_t + 1): # Set to run from start_t to max_t, for each t se which engines are active, which are already dead and which died this t
        snap, failed_units, just_failed = snapshot_at_cycle(df, t)

        top_risk = top_k_risk(
            snap, feature_cols, model,
            k=top_k,
            exclude_units=set(just_failed) # Excluding the engines that die at this exact tic to only show engines that are alive
        )

        payload = {
            "sim_cycle": int(t), # Cycle number t
            "top_risk": top_risk, # runs top_risk function
            "failed": [int(u) for u in failed_units], # prints all failed engines
            "just_failed": [int(u) for u in just_failed], # prints engines that failed this tic
        }

        if print_pretty:
            print(json.dumps(payload, indent=2)) # Ads indentation to payload print, easier to read
        else:
            print(payload)

        time.sleep(tick_seconds)


#main function to give parameters
def main():
    filepath = "archive/CMaps/train_FD001.txt"

    # Threshold for RUL
    rul_threshold = 80

    # Simulation settings, values that will overwrite the simulation
    top_k = 5
    tick_seconds = 1.0  # Can be changed for faster and slower tic
    start_t = 1
    end_t = None  # Can be changed for ha shorter demo

    df, feature_cols = load_cmapss_train(filepath)
    df = add_rul_and_label(df, rul_threshold=rul_threshold)

    model = train_rf_classifier(df, feature_cols)

    print("\n=== Starting simulation ===")
    simulate_stream(
        df, feature_cols, model,
        start_t=start_t,
        end_t=end_t,
        tick_seconds=tick_seconds,
        top_k=top_k,
        print_pretty=False
    )


if __name__ == "__main__":
    main()
