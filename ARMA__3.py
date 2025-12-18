# ARMA
# Autoregressive Moving Average Model

# import Libraries
import inspect
import time
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns
from pymongo import MongoClient
from sklearn.metrics import mean_absolute_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")
#................................................................
# connect to MongoDB
client = MongoClient("mongodb://127.0.0.1:27017")
# Assign database
db = client["air-quality"]
# Assign collection
nairobi = db["nairobi"]
#................................................................
# Wrangle Function
def wrangle(collection, resample_rule="h"):
    results = collection.find(
        {"metadata.site": 29, "metadata.measurement": "P2"},
        projection={"P2": 1, "timestamp": 1, "_id": 0},
    )

    df = pd.DataFrame(list(results))
    # Convert timestamp column to datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    # Set as index
    df = df.set_index("timestamp")
    # If timezone-naive → localize
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    
    # Localize timezone
    df.index = df.index.tz_convert("UTC").tz_convert("Africa/Nairobi")

    # Remove outliers
    df = df[df["P2"] < 500]

    # Resample and forward-fill
    y = df.resample(resample_rule).mean().ffill()

    return y
# call wrangle function
y = wrangle(nairobi)
print(y.shape)
y.head()
#...............................................................
# exploratory Data Analysis
fig, ax = plt.subplots(figsize=(15, 6))
plot_acf(y, ax=ax)
ax.set_ylim(bottom= -0.2)
plt.xlabel("Lags [Hours]")
plt.ylabel("Correlation Coefficient")
plt.tight_layout()
plt.show()

# Partial Autocorrelation Plot
fig, ax = plt.subplots(figsize=(15, 6))
plot_pacf(y, ax=ax)
ax.set_ylim(bottom= -0.2)
plt.xlabel("Lags [Hours]")
plt.ylabel("Correlation Coefficient")
plt.tight_layout()
plt.show()
#...............................................................
# Train-Test Split
# Training set: all readings from October 2018
y_train = y['2018-10-01':'2018-10-31']

# Test set: readings from 1 November 2018
y_test = y['2018-11-01':'2018-11-01']

# check
print(y_train.shape)
y_train.head()

print(y_test.shape)
y_test.head()
#...............................................................
# Building the Model
# Baseline
y_train_mean = y_train.mean()
y_pred_baseline = [y_train_mean] * len(y_train)
mae_baseline = mean_absolute_error(y_train, y_pred_baseline)
print("Mean P2 Reading:", round(y_train_mean[0], 2))
print("Baseline MAE:", round(mae_baseline, 2))
#...............................................................
# Iterate
# ARMA Model
# Create ranges for possible p and q hyperparameters
p_params = range(0, 25, 8)
q_params = range(0, 3, 1)

list(p_params)
list(q_params) 
#.................................
for p in p_params:
    for q in q_params:
        order = (p, 0, q)
        print(order)
        model = ARIMA(y_train, order=order).fit()
        print(f"Trained ARMA {order}")
#.........................................
# Add timing
for p in p_params:
    for q in q_params:
        order = (p, 0, q)
        start_time = time.time()
        print(order)
        model = ARIMA(y_train, order=order).fit()
        elapsed_time = round(time.time() - start_time, 2)
        
        print(f"Trained ARMA {order} in {elapsed_time} seconds.")
        # Generate in-sample (training) predictions
        y_pred = model.predict() 
        # Calculate training MAE
        mae = mean_absolute_error(y_train, y_pred)
        print(f"Training MAE: {round(mae, 2)}")
#...............................................................
# Grid to store MAEs
# Create dictionary to store MAEs
mae_grid = dict()
# Outer loop: Iterate through possible values for `p`
for p in p_params:
    # Create key-value pair in dict. Key is `p`, value is empty list.
    mae_grid[p] = list()
    # Inner loop: Iterate through possible values for `q`
    for q in q_params:
        # Combination of hyperparameters for model
        order = (p, 0, q)
        # Note start time
        start_time = time.time()
        # Train model
        model = ARIMA(y_train, order=order).fit()
        # Calculate model training time
        elapsed_time = round(time.time() - start_time, 2)
        print(f"Trained ARIMA {order} in {elapsed_time} seconds.")
        # Generate in-sample (training) predictions
        y_pred = model.predict() 
        # Calculate training MAE
        mae = mean_absolute_error(y_train, y_pred)
        # Append MAE to list in dictionary
        mae_grid[p].append(mae)

print()
print(mae_grid)
#...............................................................
mae_grid
# Convert to DataFrame for better visualization
# Convert dictionary to DataFrame
mae_df = pd.DataFrame(mae_grid)
mae_df.round(4)
#...............................................................
# heatmap of MAE values
sns.heatmap(mae_df, annot=True, fmt=".4g", cmap="YlGnBu")
plt.xlabel("AR Order (p values)")
plt.ylabel("MA Order (q values)")
plt.title("ARMA Grid Search (Criterion: MAE)")    
# Note: Lower MAE values the better the model performance
#...............................................................
# Residual Analysis
fig, ax = plt.subplots(figsize=(15, 12))
model.plot_diagnostics(fig=fig);
#...............................................................
# Evaluate
# using the best model (8,0,1) with lowest training time and relatively low MAE
y_pred_wfv = pd.Series(dtype=float)
history = y_train.copy()

for i in range(len(y_test)):

    model = ARIMA(history, order=(8, 0, 1)).fit()
    next_pred = model.forecast()

    # Append prediction
    y_pred_wfv = pd.concat([y_pred_wfv, next_pred])

    # Append the actual next value from y_test
    history = pd.concat([history, y_test.loc[next_pred.index]])
#...............................................................
test_mae = mean_absolute_error(y_test, y_pred_wfv)
print("Test MAE (walk forward validation):", round(test_mae, 2))

#...............................................................
# Communicate Results
df_predictions = pd.DataFrame({"y_test": y_test, "y_pred_wfv": y_pred_wfv})
fig = px.line(df_predictions, labels={"value": "PM2.5"})
fig.show()
#...............................................................
# Make both 1-D
y_test = y_test.squeeze()
y_pred_wfv = y_pred_wfv.squeeze()

# Align indices
y_pred_wfv.index = y_test.index

# Combine
df_predictions = pd.DataFrame({
    "y_test": y_test,
    "y_pred_wfv": y_pred_wfv
})

# Plot
fig = px.line(df_predictions, labels={"value": "PM2.5"})
fig.show()
#...............................................................
# Matplotlib Plot for results
# Ensure 1-D series
y_test = y_test.squeeze()
y_pred_wfv = y_pred_wfv.squeeze()

# Align prediction index with test index
y_pred_wfv.index = y_test.index

# Plot
plt.figure(figsize=(14, 6))

plt.plot(y_test.index, y_test.values, label="Actual", linewidth=2)
plt.plot(y_pred_wfv.index, y_pred_wfv.values, label="Predicted", linewidth=2)

plt.xlabel("Time")
plt.ylabel("PM2.5")
plt.title("Actual vs Predicted PM2.5 Readings (Walk-Forward Validation)")

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
