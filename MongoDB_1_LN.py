# import libraries
from pprint import PrettyPrinter
import pandas as pd
from pymongo import MongoClient
#.............................................................................
pp = PrettyPrinter(indent=2)
# connect to MongoDB
client = MongoClient("mongodb://127.0.0.1:27017")
#.........................................
client.list_databases()
#Sample
from sys import getsizeof
lt =[0,1,2,3,4]
for i in lt:
    print(i)

rang = range(0,5) # Iterator

# Important difference between list and range
# Print the size of list and range in bytes
getsizeof(lt)  # size of list
getsizeof(rang) # size of range

# The Ilterator is more memory efficient
# bcos for large dataset it bring all the dataset into memory at once
# but gives what you need on demand
#.........................................
list(client.list_databases())
pp.pprint(list(client.list_databases()))
# Assign database
db = client["air-quality"]
# list collections in the database
list(db.list_collection_names()) [0]
pp.pprint(list(db.list_collection_names()))
# also you can use
for c in db.list_collections():
    print(c["name"])
# Assign collection
nairobi = db["nairobi"]
# count documents in the collection
nairobi.count_documents({})
# Find one document
resutlt = nairobi.find_one({})
pp.pprint(resutlt)
# unique site in the dataset
nairobi.distinct("metadata.site")

# count number of unique sites
nairobi.count_documents({"metadata.site": 6})

print("Documents from site 6:", nairobi.count_documents({"metadata.site": 6}))
print("Documents from site 29:", nairobi.count_documents({"metadata.site": 29}))
#.............................................
# Aggregation Framework
# count number of documents per site
result = nairobi.aggregate([
    {
        "$group": {
            "_id": "$metadata.site",
            "count": {"$sum": 1}
        }
    }
])
pp.pprint(list(result))

# or
result =  nairobi.aggregate([
    {
        "$group": {
            "_id": "$metadata.site",
            "count": {"$count": {}}}}
])
pp.pprint(list(result))
#.............................................
nairobi.distinct("metadata.measurement")
#.............................................
from pymongo import UpdateOne
import ast

batch_size = 1000
bulk_ops = []

cursor = nairobi.find({}, {"metadata": 1})

for i, doc in enumerate(cursor, 1):
    try:
        metadata_fixed = ast.literal_eval(doc["metadata"])
        bulk_ops.append(
            UpdateOne(
                {"_id": doc["_id"]},
                {"$set": {"metadata": metadata_fixed}}
            )
        )
    except:
        continue

    # Execute every batch
    if i % batch_size == 0:
        nairobi.bulk_write(bulk_ops)
        bulk_ops = []
        print(f"{i} documents processed...")

# Run remaining operations
if bulk_ops:
    nairobi.bulk_write(bulk_ops)

print("✅ Metadata conversion complete")
#.............................................
# Retrieve PM 2.5 measurements from all site
resutlt = nairobi.find({"metadata.measurement": "P2"}).limit(3)
pp.pprint(list(resutlt))
#.............................................
# Aggregation metadata.measurement
# number o obs broken down by the type of reading ['P1', 'P2', 'humidity', 'temperature']
result =  nairobi.aggregate([
    {"$match": {"metadata.site": 6}},
    {"$group": {"_id": "$metadata.measurement", "count": {"$count": {}}}}
])
pp.pprint(list(result))
# for site 29
result =  nairobi.aggregate([
    {"$match": {"metadata.site": 29}},
    {"$group": {"_id": "$metadata.measurement", "count": {"$count": {}}}}
])
pp.pprint(list(result))
#..............................................
# IMPORT
# retrieve the PM 2.5 readings from site 29
# limit your results to 3 records only
# use the projection argument to limit the results to the "P2" and "timestamp" keys only
result = nairobi.find(
    {"metadata.measurement": 29, "metadata.measurement": "P2"},
)
pp.pprint(result.next())
# change projection
result = nairobi.find(
    {"metadata.measurement": 29, "metadata.measurement": "P2"},
    projection = {"P2": 1, "timestamp": 1}
)
pp.pprint(result.next())
# remove _id from projection
result = nairobi.find(
    {"metadata.measurement": 29, "metadata.measurement": "P2"},
    projection = {"P2": 1, "timestamp": 1, "_id": 0}
)
pp.pprint(result.next())
#..............................................
# load data into pandas dataframe
df = pd.DataFrame(result)
df.head()
# make timestamp as index
df = pd.DataFrame(result).set_index("timestamp")
df.head()
#.................................................................................................
# Linear Regression with Time Series Data (MongoDB)
# import libraries
import pandas as pd
from pymongo import MongoClient
import matplotlib.pyplot as plt
import plotly.express as px
import pytz
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
#.............................................................................
# connect to MongoDB
client = MongoClient("mongodb://127.0.0.1:27017")
# Assign database
db = client["air-quality"]
# Assign collection
nairobi = db["nairobi"]
#...................................................
# Wrangle Function
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
    df = df["P2"].resample("h").mean().ffill().to_frame() # updated syntax

    # Create a time Lag feature
    df["P2.Lag1"] = df["P2"].shift(1)
    
    # drop missing values
    df.dropna(inplace=True)

    return df
#...................................................
df = wrangle(nairobi)
print(df.shape)
df.head(10)
#...................................................
# Timezone Handling
df.index[:5]
# localize timestamp to UTC
df.index.tz_localize("UTC")[:5]
# convert timestamp to Nairobi timezone
df.index.tz_convert("UTC").tz_convert("Africa/Nairobi")[:5]
#.............................................................................
# Visualize the data
# box plot
fig, ax = plt.subplots(figsize=(15, 6))
df["P2"].plot(kind="box", vert=False, title="Distribution of P2 Readings",ax=ax)
# pd.plot
fig, ax = plt.subplots(figsize=(15, 6))
df["P2"].plot(xlabel="Time", ylabel="PM2.5", title="PM 2.5 Time Series", ax=ax)
#...............................................................................
# Resample data to hourly and fill missing values
df["P2"].resample("H").mean().fillna(method="ffill").head().isnull().sum()
df["P2"].resample("h").mean().ffill().to_frame().head() # updated syntax
#...............................................................................
# Plot rolling average of PM2.5 readings
df["P2"].rolling(168).mean().isnull().sum()

fig, ax = plt.subplots(figsize=(15, 6))
df["P2"].rolling(168).mean().plot(ax=ax, ylabel="PM2.5", title= "weekly rolling avearage")
#...........................................................................
# Create a time Lag
df["P2.Lag1"] = df["P2"].shift(1)
# drop missing values
df.dropna().head()
#...................................................
# scatter plot that shows PM 2.5 mean reading 
# for each hour as a function of the mean reading from the previous hour.
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(x=df["P2.Lag1"], y=df["P2"], alpha=0.5)
ax.plot([0,120], [0,120], color="red", linestyle="--")
plt.xlabel("P2 Lag 1")
plt.ylabel("P2")
plt.title("PM 2.5 Autocorrelation Scatter Plot")
plt.show()
#...........................................................................
# Split data
target = "P2"
y = df[target]
X = df.drop(columns=[target])

y.head()
X.head()

# get the 80th percentile index
cutoff = int(0.8 * len(df))
# training data 
X_train, y_train = X.iloc[:cutoff], y.iloc[:cutoff]
# testing data
X_test, y_test = X.iloc[cutoff:], y.iloc[cutoff:]

# check lengths fo X
len(X_train), len(X_test)
len(X_train) + len(X_test) == len(X)
# check lengths of y
len(y_train), len(y_test)
len(y_train) + len(y_test) == len(y)
#...........................................................................
# Baseline Model
# Always predict the mean of the training data
y_pred_baseline = [y_train.mean()] * len(y_train)
# Compute MAE
mae_baseline = mean_absolute_error(y_train, y_pred_baseline)
# Print results
print("Mean P2 Reading:", round(y_train.mean(), 2))
print("Baseline MAE:", round(mae_baseline, 2))
#...........................................................................
# Linear Regression Model
# Instantiate the model
model = LinearRegression()
# Fit the model
model.fit(X_train, y_train)
# Make predictions
y_pred_train = model.predict(X_train)
y_pred_train[:5]
#...............................................
# Evaluate the model on training data
train_mae = mean_absolute_error(y_train, y_pred_train)
test_mae = mean_absolute_error(y_test, model.predict(X_test))
# Print results
print("Training MAE:", round(train_mae, 2))
print("Testing MAE:", round(test_mae, 2))
# ...............................................
# Communicate results
intercept = model.intercept_.round(2)
coefficient = model.coef_.round(2)
print(f"P2 = {intercept} + ({coefficient} * P2.L1)")
#...............................................
df_pred_test = pd.DataFrame(
    {
        "y_test": y_test,
        "y_pred": model.predict(X_test)
    }
)
df_pred_test.head()
#...............................................
# Visualize predictions vs actuals
fig = px.line(df_pred_test, labels={"value": "P2"})
legend_names = {
    "y_test": "Actual P2",
    "y_pred": "Predicted P2"
}
fig.for_each_trace(
    lambda trace: trace.update(name=legend_names[trace.name])
)
fig.show()
#...............................................

#...............................................
fig = px.line(df_pred_test, labels={"value": "P2"})

legend_names = {
    "y_test": "Actual P2",
    "y_pred": "Predicted P2"
}
fig.for_each_trace(
    lambda trace: trace.update(name=legend_names.get(trace.name, trace.name))
)

# Move legend above and set a normal width
fig.update_layout(
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.05,
        xanchor="center",
        x=0.5
    ),
    width=700,        # <--- Adjust width here
    height=350        # <--- Optional height
)

fig.show()

#................................................
# the purple line is the actual P2 readings
# while the red line is the predicted P2 readings
#...............................................