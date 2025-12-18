# Autoregressive Models
# import libraries
import warnings
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
from pymongo import MongoClient
from sklearn.metrics import mean_absolute_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg
warnings.simplefilter(action="ignore", category=FutureWarning)
#.................................................................
# connect to MongoDB
client = MongoClient("mongodb://127.0.0.1:27017")
# Assign database
db = client["air-quality"]
# Assign collection
nairobi = db["nairobi"]
#...................................................................
# Wrangle Function from MongoDB_1
def wrangle(collection):
    results = collection.find(
        {"metadata.site": 29, "metadata.measurement": "P2"},
        projection={"P2": 1, "timestamp": 1, "_id": 0}
    )
    df = pd.DataFrame(results)
    # Convert timestamp column to datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    # Set as index
    df = df.set_index("timestamp")
    # If timezone-naive → localize
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    
    # localised timezone
    df.index = df.index.tz_convert("UTC").tz_convert("Africa/Nairobi")
    
    #Remove Outliers
    df = df[df["P2"] < 500]
    
    # Resample to 1 hour window, fill missing values
    df = df["P2"].resample("h").mean().ffill() # updated syntax

    return df
#...................................................
y = wrangle(nairobi)
print(y.shape)
y.head()
#...................................................
# Exploratory Data Analysis
y.corr(y)
y.corr(y.shift(2))
# as the lag increases, the correlation decreases
# or predictability decreases
#...................................................
#  ACF plot for the data in y
fig, ax = plt.subplots(figsize=(15, 6))
plot_acf(y, ax=ax)
plt.xlabel("Lag [hours]")
plt.ylabel("Correlation Coefficient");
#...................................................
y.shift(1).corr(y.shift(2))
# PACF plot for the data in y
fig, ax = plt.subplots(figsize=(15, 6))
plot_pacf(y, ax=ax)
plt.xlabel("Lag [hours]")
plt.ylabel("Correlation Coefficient");
#...................................................
# Train-Test Split
int(len (y) * 0.95)

cutoff_test = int(len (y) * 0.95)

y_train = y[:cutoff_test]
y_test = y[cutoff_test:]
# check lengths
len(y_train) + len(y_test) == len(y)
#...................................................
# Building the Model
# Basline
y_train_mean = y_train.mean()
y_pred_baseline = [y_train_mean] * len(y_train)
mae_baseline = mean_absolute_error(y_train, y_pred_baseline)

print("Mean P2 Reading:", round(y_train_mean, 2))
print("Baseline MAE:", round(mae_baseline, 2))
#...................................................
# Iterate
model = AutoReg(y_train, lags=26).fit()
model.predict().isnull().sum()
# the 26 NaN values correspond to the 26 lags
#...................................................
y_pred = model.predict().dropna()
training_mae = mean_absolute_error(y_train.iloc[26:], y_pred)
print("Training MAE:", training_mae)
#...................................................
# Residual Analysis
y_train_resid = y_train- y_pred # or y_train_resid = model.resid
y_train_resid.tail()
#...................................................
# Time Series Plot of Residuals
fig, ax = plt.subplots(figsize=(15, 6))
y_train_resid.plot(ylabel= "residual Value", ax=ax)
# Histogram of Residuals
y_train_resid.hist()
# ACF plot of y_train_resid
fig, ax = plt.subplots(figsize=(15, 6))
plot_acf(y_train_resid.iloc[26:], ax=ax);
#...................................................
# Evaluate
y_test.tail()
y_pred_test = model.predict(y_test.index.min(), y_test.index.max())
test_mae = mean_absolute_error(y_test, y_pred_test)
print("Test MAE:", test_mae)
#...................................................
df_pred_test = pd.DataFrame(
    {"y_test": y_test, "y_pred": y_pred_test}, index=y_test.index
)
df_pred_test.head()
print(df_pred_test.shape)

fig = px.line(df_pred_test, labels={"value": "P2"})
fig.show()
#...................................................
# Walk-Forward Validation
y_pred_wfv = pd.Series()
history = y_train.copy()
for i in range(len(y_test)):
    pass
#....................................................
history.tail(1)
y_test.head(1)
# fit model
model = AutoReg(history, lags=26).fit()
# Out-of-sample prediction
model.forecast()
#.....................................................
# Old code
y_pred_wfv = pd.Series()
history = y_train.copy()
for i in range(len(y_test)):
    model = AutoReg(history, lags=26).fit()
    next_pred = model.forecast()
    y_pred_wfv = y_pred_wfv.append(next_pred)
    history = history.append(y_test[next_pred.index])
#.....................................................
# New code
y_pred_wfv = pd.Series(dtype=float)   # specify dtype
history = y_train.copy()

for i in range(len(y_test)):
    model = AutoReg(history, lags=26).fit()
    next_pred = model.forecast()

    # concatenate instead of append
    y_pred_wfv = pd.concat([y_pred_wfv, next_pred])

    # update history safely
    history = pd.concat([history, y_test[next_pred.index]])
#........................................................
# test MAE for WFV
test_mae = mean_absolute_error(y_test, y_pred_wfv)
print("Test MAE (WFV):", round(test_mae, 2))
#........................................................
print(model.params)
print(model.summary())
#........................................................
df_pred_test = pd.DataFrame(
    {"y_test": y_test, "y_pred_wfv": y_pred_wfv},
)
fig = px.line(df_pred_test, labels={"value": "P2"})
fig.show()
#........................................................