# import libraries
from pprint import PrettyPrinter
import pandas as pd
from pymongo import MongoClient
#.............................................................................
pp = PrettyPrinter(indent=2)
# connect to MongoDB
client = MongoClient("mongodb://127.0.0.1:27017")
# Assign database
db = client["air-quality"]
# Assign collection
pp.pprint(list(db.list_collection_names()))
dar = db["dar-es-salaam"]
#.............................................................................
# Explore
dar.count_documents({})
resutlt = dar.find_one({})
pp.pprint(resutlt)
dar.distinct("metadata.site")
#.............................................................................
# covert data from string to dict error while loading data into MongoDB
import ast
from pymongo import UpdateOne

batch_size = 1000
operations = []

for i, doc in enumerate(dar.find()):
    if isinstance(doc['metadata'], str):
        operations.append(
            UpdateOne(
                {"_id": doc["_id"]},
                {"$set": {"metadata": ast.literal_eval(doc["metadata"])}}
            )
        )
    # Execute in batches
    if len(operations) == batch_size:
        dar.bulk_write(operations)
        operations = []
        print(f"{i+1} documents processed")

# Execute remaining operations
if operations:
    dar.bulk_write(operations)
    print(f"{i+1} documents processed")
#.............................................................................
# Explore after conversion
resutlt = dar.find_one({})
pp.pprint(resutlt)
sites = dar.distinct("metadata.site")
sites
#.............................................................................
# Aggregation Framework
# count number of documents per site
result = dar.aggregate([
    {
        "$group": {
            "_id": "$metadata.site",
            "count": {"$sum": 1}
        }
    }
])
readings_per_site = list(result)
readings_per_site
#.....................................
dar.distinct("metadata.measurement")
#.............................................................................
# IMPORT
# retrieve the PM 2.5 readings from site 29
# limit your results to 3 records only
# use the projection argument to limit the results to the "P2" and "timestamp" keys only
result = dar.find(
    {"metadata.measurement": 11, "metadata.measurement": "P2"},
)
pp.pprint(result.next())


# Change Projection
result = dar.find({"metadata.site": 11, "metadata.measurement": "P2"},
    projection={"P2": 1, "timestamp": 1, "_id": 0})

pp.pprint(result.next())
#.............................................................................
# Wrangling Function
def wrangle(collection):
    result = collection.find({"metadata.site": 11, "metadata.measurement": "P2"},
    projection={"P2": 1, "timestamp": 1, "_id": 0})

    df = pd.DataFrame(result)

    # Convert timestamp column to datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # Set as index
    df = df.set_index("timestamp")

    # If timezone-naive → localize
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    
    # localised timezone
    df.index = df.index.tz_convert("UTC").tz_convert("Africa/Dar_es_Salaam")
    
    # Remove outliers
    df = df[df["P2"] < 100]
     
    # Resample to 1 hour window, fill missing values
    df = df["P2"].resample("h").mean().ffill().to_frame()
    
   
    return df
# call wrangle function
df = wrangle(dar)
print(df.shape)
df.head()
#.............................................................................
# Exploratory Data Analysis
import matplotlib.pyplot as plt
import pandas as pd

#
fig, ax = plt.subplots(figsize=(15, 6))
df["P2"].plot(xlabel="Date", ylabel="PM2.5 Level", title="Dar es Salaam PM2.5 Levels", ax=ax)
plt.tight_layout()
plt.show()

# Line plot 
fig, ax = plt.subplots(figsize=(15, 6))
df["P2"].rolling(168).mean().plot(ax=ax, ylabel="PM2.5 Level", title= "Dar es Salaam PM2.5 Levels, 7-Day Rolling Average")
plt.tight_layout()
plt.show()

# ACF Plot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
fig, ax = plt.subplots(figsize=(15, 6))
plot_acf(df, ax=ax)
ax.set_ylim(bottom= -0.2)
plt.xlabel("Lags [Hours]")
plt.ylabel("Correlation Coefficient")
plt.title("Dar es Salaam PM2.5 Readings, ACF")
plt.tight_layout()
plt.show()

# PACF Plot
fig, ax = plt.subplots(figsize=(15, 6))
plot_pacf(df, ax=ax)
ax.set_ylim(bottom= -0.2)
plt.xlabel("Lags [Hours]")
plt.ylabel("Correlation Coefficient")
plt.tight_layout()
plt.show()
#.............................................................................
# Train-Test Split
df = df["P2"].resample("h").mean().ffill() # make y as series for splitting

cutoff_test = int(len (df) * 0.9)

y_train = df[:cutoff_test]
y_test = df[cutoff_test:]

print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
#.............................................................................
# Building the Model
# Basline
from sklearn.metrics import mean_absolute_error
y_train_mean = y_train.mean()
y_pred_baseline = [y_train_mean] * len(y_train)
mae_baseline = mean_absolute_error(y_train, y_pred_baseline)

print("Mean P2 Reading:", round(y_train_mean, 2))
print("Baseline MAE:", round(mae_baseline, 2))
#.............................................................................
# Iterate
# Instantiate  AutoReg
from statsmodels.tsa.ar_model import AutoReg

model = AutoReg(y_train, lags=30).fit()
model.predict().isnull().sum() # the 30 NaN values correspond to the 30 lags

y_pred = model.predict().dropna()
training_mae = mean_absolute_error(y_train.iloc[30:], y_pred)
print("Training MAE:", round(training_mae, 2))
#.............................................................................
# Create range to test different lags
p_params = range(1, 31)

# Create empty list to hold mean absolute error scores
maes = []

# Iterate through all values of p in `p_params`
for p in p_params:
    # Build model
    model = AutoReg(y_train, lags=p).fit()

    # Make predictions on training data, dropping null values caused by lag
    y_pred = model.predict().dropna()

    # Calculate mean absolute error for training data vs predictions
    mae = mean_absolute_error(y_train.iloc[p:], y_pred)

    # Append `mae` to list `maes`
    maes.append(mae)

# Put list `maes` into Series with index `p_params`
mae_series = pd.Series(maes, name="mae", index=p_params)

# Inspect head of Series
mae_series.head()
#..............................................................................
# Determine best p
best_p = mae_series.idxmin()
best_model = AutoReg(y_train, lags=best_p).fit()
best_mae = mae_series.min()
print(f"Best p: {best_p}, with MAE: {round(best_mae, 2)}")
#.............................................................................
# Personal review
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))
plt.plot(mae_series.index, mae_series.values, marker='o')
plt.title("MAE for Different Lag Values (p = 1 to 30)")
plt.xlabel("Lag Value (p)")
plt.ylabel("Mean Absolute Error (MAE)")

# Highlight the best lag
best_p = mae_series.idxmin()
best_mae = mae_series.min()

plt.scatter(best_p, best_mae, s=120)
plt.text(best_p, best_mae, f"  Best p = {best_p}", fontsize=12)

plt.grid(True)
plt.show()
#.............................................................................
# Residual Analysis
y_train_resid = y_train - best_model.dropna()
y_train_resid.name = "residuals"
y_train_resid

#...........................................
# Get model fitted values (in-sample predictions)
fitted = best_model.fittedvalues

# Align y_train to match prediction index
y_aligned = y_train.loc[fitted.index]

# Compute residuals
y_train_resid = y_aligned - fitted
y_train_resid.name = "residuals"
y_train_resid
#...........................................................................
# Residual Plots
fig, ax = plt.subplots()
y_train_resid.hist(ax=ax)
ax.set_title("Best Model, Training Residuals")
ax.set_xlabel("Residual")
ax.set_ylabel("Frequency")
plt.tight_layout()
plt.show()
#.............................................................................
# ACF Plot of Residuals
fig, ax = plt.subplots(figsize=(15, 6))
plot_acf(y_train_resid, ax=ax)
ax.set_ylim(bottom= -0.2)
plt.xlabel("Lag [Hours]")
plt.ylabel("Correlation Coefficient")
plt.title("Dar es Salaam, Training Residuals ACF")
plt.tight_layout()
#.........................................................................
# Walk-Forward Validation
y_pred_wfv = pd.Series(dtype="float64")
history = y_train.copy()
for i in range(len(y_test)):
    model = AutoReg(history, lags=best_p).fit()
    next_pred = model.forecast()

    # Append prediction
    y_pred_wfv = pd.concat([y_pred_wfv, next_pred])

    # Append the actual next value from y_test
    history = pd.concat([history, y_test.loc[next_pred.index]])
        
y_pred_wfv.name = "prediction"
y_pred_wfv.index.name = "timestamp"
y_pred_wfv
#...........................
# Evaluate Walk-Forward Validation
test_mae = mean_absolute_error(y_test, y_pred_wfv)
print("Test MAE (walk forward validation):", round(test_mae, 2))
#...........................................................................
# Communicate Results
import plotly.express as px

df_pred_test =  pd.DataFrame({
    "y_test": y_test,
    "y_pred_wfv": y_pred_wfv
})
fig = px.line(df_pred_test)
fig.update_layout(
    title="Dar es Salaam, WFV Predictions",
    xaxis_title="Date",
    yaxis_title="PM2.5 Level",
)
#...........................................................................
# Create the figure and axis
fig, ax = plt.subplots(figsize=(15, 6))

# Plot the actual values
ax.plot(y_test.index, y_test.values, label="y_test", color="blue")

# Plot the predicted values
ax.plot(y_pred_wfv.index, y_pred_wfv.values, label="y_pred_wfv", color="red")

# Set title and axis labels
ax.set_title("Dar es Salaam, WFV Predictions")
ax.set_xlabel("Date")
ax.set_ylabel("PM2.5 Level")

# Show legend
ax.legend()

# Adjust layout to fit everything
plt.tight_layout()

# Show plot
plt.show()
#...........................................................................